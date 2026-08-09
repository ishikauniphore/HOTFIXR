"""
Offline merger for verl FSDP2 sharded checkpoints.

verl 0.4.0's FSDPCheckpointManager saves per-rank sharded state dicts (DTensors)
via ShardedStateDictConfig. This script reconstructs the full HuggingFace model
by combining the local shards from each rank file, mirroring the logic in
verl/third_party/vllm/.../dtensor_weight_loaders.py:redistribute_dtensor().
"""

import argparse
import os
import glob

import torch
import torch.distributed.tensor
from torch.distributed._tensor import DTensor, Replicate, Shard
from transformers import AutoConfig, AutoModelForCausalLM


def merge_dtensor_shards(shards: list[DTensor]) -> torch.Tensor:
    """Reconstruct a full tensor from per-rank DTensor local shards."""
    placements = shards[0].placements
    assert len(placements) == 1, f"Expected 1D mesh, got {placements}"
    placement = placements[0]

    if isinstance(placement, Replicate):
        return shards[0].to_local()
    elif isinstance(placement, Shard):
        dim = placement.dim
        return torch.cat([s.to_local() for s in shards], dim=dim)
    else:
        raise ValueError(f"Unsupported placement type: {type(placement)}")


def load_and_merge(checkpoint_dir: str, target_dir: str):
    # Find all rank shard files
    rank_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_world_size_*_rank_*.pt")))
    if not rank_files:
        raise FileNotFoundError(f"No model shard files found in {checkpoint_dir}")

    world_size = int(rank_files[0].split("world_size_")[1].split("_")[0])
    print(f"Found {len(rank_files)} shards (world_size={world_size})")

    # Load all rank shards
    shards = []
    for rank in range(world_size):
        path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"Loading {path} ...")
        shards.append(torch.load(path, map_location="cpu", weights_only=False))

    # Reconstruct full state dict
    full_state_dict = {}
    for key in shards[0].keys():
        tensors = [shards[rank][key] for rank in range(world_size)]
        if isinstance(tensors[0], DTensor):
            full_state_dict[key] = merge_dtensor_shards(tensors)
        else:
            full_state_dict[key] = tensors[0]

    print(f"Merged {len(full_state_dict)} parameters")

    # Load model config from the checkpoint dir (saved by rank 0 during training)
    config = AutoConfig.from_pretrained(checkpoint_dir)
    print(f"Model config: {config.model_type}, arch={config.architectures}")

    # Instantiate empty model and fill weights
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.load_state_dict(full_state_dict, strict=True)

    # Save HuggingFace model
    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)

    # Copy tokenizer files from checkpoint dir
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
        "generation_config.json",
    ]
    import shutil
    for fname in tokenizer_files:
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_dir, fname))

    print(f"Saved merged model to {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, help="Path to actor checkpoint dir (contains model_world_size_*_rank_*.pt)")
    parser.add_argument("--target_dir", required=True, help="Output directory for merged HuggingFace model")
    args = parser.parse_args()
    load_and_merge(args.local_dir, args.target_dir)
