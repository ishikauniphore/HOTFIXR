import os
from pathlib import Path

def require_env(name):
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Export it in your shell profile before using HOTFIXR "
            f"(same env var the underlying scripts already require)."
        )
    return val

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirrors the hardcoded Hydra overrides in run_verl.sh. Keys that vary per
# call (train_files, model.path, reward path, job/trainer naming, GPU count)
# are filled in by train_generator() itself.
GRPO_DEFAULTS = {
    "algorithm.adv_estimator": "grpo",
    "data.train_batch_size": 12,
    "data.max_prompt_length": 4096,
    "data.max_response_length": 1024,
    "data.filter_overlong_prompts": True,
    "data.truncation": "error",
    "actor_rollout_ref.model.use_remove_padding": True,
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    "actor_rollout_ref.actor.optim.lr": "1e-6",
    "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.entropy_coeff": 0,
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
    "+actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
    "actor_rollout_ref.rollout.n": 4,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7,
    "actor_rollout_ref.rollout.max_model_len": 4096,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.fsdp_config.param_offload": True,
    "algorithm.use_kl_in_reward": False,
    "trainer.logger": '["console","wandb"]',
    "trainer.critic_warmup": 0,
    "trainer.project_name": "acquisition_synthesis",
    "trainer.save_freq": 20,
    "trainer.test_freq": -1,
    "trainer.total_epochs": 1,
    "trainer.max_actor_ckpt_to_keep": 1,
    "trainer.val_before_train": False,
    "custom_reward_function.name": "compute_score",
}
