import os
import shutil
import sys
from pathlib import Path

from ._proc import run
from ._service import ensure_reward_service_running, stop_reward_service
from .config import REPO_ROOT, GRPO_DEFAULTS, require_env

# Must match prompts.py's CUSTOM_PROMPT_ENV (the two can't import each other:
# prompts.py is loaded standalone by subprocess scripts via sys.path tricks).


def set_prompt(prompt):
    """Override the question prompt template used by generate_data,
    train_student, and evaluate_student, in place of prompts.py's built-in
    stem/math/chat/nemotron templates.

    `prompt` must contain a literal "{q}" placeholder for the question text,
    e.g. "Answer briefly.\n\n<question> {q} </question>.". Pass prompt=None
    to clear the override and restore the defaults.

    Takes effect for pipeline calls made afterward in this process: stored
    as an env var (rather than passed in-process) because generate_data/
    train_student/evaluate_student each shell out to a script
    (data_gen_cluster.py/sft.py/model_inference*.py) that re-imports
    prompts.py in a fresh subprocess; env vars are what actually reaches it.
    """
    if prompt is None:
        os.environ.pop("HOTFIXR_CUSTOM_PROMPT", None)
        return
    if "{q}" not in prompt:
        raise ValueError('prompt must contain a literal "{q}" placeholder for the question')
    os.environ["HOTFIXR_CUSTOM_PROMPT"] = prompt

def set_variables(variables):
    for key, value in variables.items():
        os.environ[f"HOTFIXR_{key}"] = value


def train_generator(dataset_path, base_model, *, reward="lingualdeficit", trained_generator_name=None,
                     num_gpu=6, generator_cuda_visible_devices="0,1,2,3,4,5",
                     service_cuda_visible_devices=None, push=True,
                     stop_service=True, dry_run=False, **grpo_overrides):
    """GRPO-train an acquisition/generator model with verl, merge the FSDP
    checkpoint, and (by default) push it to the Hub. Mirrors run_verl.sh +
    run_upload_model.sh. Returns the pushed repo id, or the local merged
    path if push=False.

    Starts services/all.py in the background if it isn't already running
    (equivalent of `source run_services.sh`), and by default shuts it down
    again once training finishes (`stop_service=False` to leave it up for a
    follow-up train_generator() call).

    `cuda_visible_devices` is for the verl training process; `service_cuda_visible_devices`
    is for the reward service (services/all.py) and must name disjoint GPUs
    from `cuda_visible_devices` since both run concurrently. Defaults to
    services/all.py's own default ("6,7") if left unset.
    """
    dataset_name = os.path.basename(str(dataset_path).rstrip("/"))
    if trained_generator_name is None:
        trained_generator_name = f"generator_{base_model.split('/')[-1]}_{dataset_name}_{reward}"

    localhost = "localhost"
    ckpt_root = Path(f"/dev/shm/grpo_synthesis_models/{trained_generator_name}")

    if dry_run:
        print(f"[dry_run] would rm -rf {ckpt_root}")
    else:
        shutil.rmtree(ckpt_root, ignore_errors=True)

    ensure_reward_service_running(dry_run=dry_run,
                                   cuda_visible_devices=service_cuda_visible_devices)

    if dry_run:
        print(f"[dry_run] would POST http://{localhost}:5145/start_service "
              f"service={reward!r} kwargs={{'model_name': {base_model!r}}}")
    else:
        import requests
        resp = requests.post(
            f"http://{localhost}:5145/start_service",
            json={"service": reward, "kwargs": {"model_name": base_model}},
        )
        resp.raise_for_status()

    overrides = dict(GRPO_DEFAULTS)
    overrides.update({
        "data.train_files": f"{REPO_ROOT}/data/{dataset_name}/train.parquet",
        "data.val_files": f"{REPO_ROOT}/data/{dataset_name}/test.parquet",
        "actor_rollout_ref.model.path": base_model,
        "custom_reward_function.path": f"{REPO_ROOT}/rewards/{reward}.py",
        "trainer.experiment_name": trained_generator_name,
        "trainer.n_gpus_per_node": num_gpu,
        "trainer.nnodes": 1,
        "trainer.default_local_dir": str(ckpt_root),
    })
    overrides.update(grpo_overrides)

    cmd = ["python3", "-m", "verl.trainer.main_ppo"] + [f"{k}={v}" for k, v in overrides.items()]
    run(cmd, cwd=REPO_ROOT, dry_run=dry_run, env={
        "CUDA_VISIBLE_DEVICES": generator_cuda_visible_devices,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    })

    run(["ray", "stop", "--force"], dry_run=dry_run)

    hf_username = os.environ.get("HF_USERNAME")
    merged_dir = ckpt_root / "merged"

    if dry_run:
        print(f"[dry_run] would merge FSDP checkpoint from {ckpt_root}/global_step_*/actor "
              f"into {merged_dir}")
        if stop_service:
            stop_reward_service(dry_run=dry_run)
        if push:
            repo_id = f"{hf_username or '<HF_USERNAME>'}/{trained_generator_name}"
            print(f"[dry_run] would push merged model to {repo_id}")
            return repo_id
        return str(merged_dir)

    sys.path.insert(0, str(REPO_ROOT))
    from merge_fsdp_model import load_and_merge

    latest_ckpt = max(ckpt_root.glob("global_step_*"), key=lambda p: p.stat().st_mtime)
    load_and_merge(str(latest_ckpt / "actor"), str(merged_dir))

    if stop_service:
        stop_reward_service()

    if not push:
        return str(merged_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_username = require_env("HF_USERNAME")
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = f"{hf_username}/{trained_generator_name}"

    gen_model = AutoModelForCausalLM.from_pretrained(str(merged_dir))
    gen_tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    gen_model.push_to_hub(repo_id, token=hf_token)
    gen_tokenizer.push_to_hub(repo_id, token=hf_token)
    return repo_id


def generate_data(dataset_path, question_generation_model, size, *, answer_model="Qwen/Qwen2.5-32B-Instruct",
                   k=4, output_file=None, cuda_visible_devices="0,1,2,3",
                   dry_run=False, **kwargs):
    """Generate synthetic question/answer/reasoning triples with vLLM.
    Wraps generating_data/data_gen_cluster.py unchanged. Returns the path to
    the output parquet file.
    """
    dataset_name = os.path.basename(str(dataset_path).rstrip("/"))

    if output_file is None:
        output_file = str(REPO_ROOT / "training_data" /
                           f"{question_generation_model.split('/')[-1]}_{dataset_name}_{size}.parquet")

    cmd = [
        "python", str(REPO_ROOT / "generating_data" / "data_gen_cluster.py"),
        "--dataset_name", dataset_name,
        "--question_generation_model", question_generation_model,
        "--answer_model_name", answer_model,
        "--size", str(size),
        "--k", str(k),
        "--output_file", output_file,
    ]
    for key, val in kwargs.items():
        cmd += [f"--{key}", str(val)]

    if not dry_run:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    run(cmd, cwd=REPO_ROOT, dry_run=dry_run, env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices})
    return output_file


def train_student(base_student_model, data_path, *, num_gpus=4, cuda_visible_devices="0,1,2,3",
                   num_epochs=3, trained_student_name=None, push=True, dry_run=False, **sft_overrides):
    """SFT-train a student model on generator-produced data, then merge the
    LoRA adapter. Wraps evaluation/sft.py (via torchrun) + evaluation/merge.py.
    Returns the pushed repo id, or the local merged path if push=False.
    """
    evaluation_dir = REPO_ROOT / "evaluation"
    sft_output_root = "/dev/shm/sft_models/"
    run_name_resolved = trained_student_name or ("student_" + Path(str(data_path)).name.split(".")[0])

    sft_cmd = [
        "torchrun", f"--nproc_per_node={num_gpus}", "sft.py",
        "--base_student_model", base_student_model,
        "--data_path", str(data_path),
        "--output_dir", sft_output_root,
        "--num_epochs", str(num_epochs),
        "--trained_student_name", run_name_resolved,
    ]
    for key, val in sft_overrides.items():
        sft_cmd += [f"--{key}", str(val)]

    run(sft_cmd, cwd=evaluation_dir, dry_run=dry_run,
        env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    sft_output_dir = str(Path(sft_output_root) / run_name_resolved)

    merge_cmd = ["python", "merge.py", "--model_path", sft_output_dir, "--base_student_model", base_student_model]
    if not push:
        merge_cmd.append("--skip_push")
    run(merge_cmd, cwd=evaluation_dir, dry_run=dry_run)

    if not push:
        return sft_output_dir

    hf_username = os.environ.get("HF_USERNAME") if dry_run else require_env("HF_USERNAME")
    return f"{hf_username or '<HF_USERNAME>'}/{run_name_resolved}"


def evaluate_student(student_model, *, cuda_visible_devices="0,1,2,3", dry_run=False):
    """Run the eval suite (inference, embedding similarity, ROUGE-L, LLM-judge)
    against a trained student model. Wraps evaluation/run_eval.sh's four
    scripts, which hand state between each other via an
    evaluation/eval_{shorthand}.pkl file (removed by model_laj.py at the
    end) and append one summary line per experiment to evaluation/results.txt.

    Returns a list of per-experiment metric dicts parsed from the new lines
    model_laj.py appends to results.txt, e.g.:
        [{"experiment": "student_x_nemotron_stem", "embed_sim": 61.2,
          "rouge_l": 48.9, "judge": 77.5}, ...]
    """
    evaluation_dir = REPO_ROOT / "evaluation"
    results_path = Path(os.path.join(os.environ.get("HOTFIXR_SAVE_RESULTS_DIRECTORY"), "results.txt"))
    env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices, "VLLM_USE_FLASHINFER_SAMPLER": "0"}

    lines_before = len(results_path.read_text().splitlines()) if results_path.exists() else 0

    for script in ("model_inference.py", "model_embed.py", "model_rouge.py", "model_laj.py"):
        run(["python", script, "--model_name", student_model], cwd=evaluation_dir,
            dry_run=dry_run, env=env)

    if dry_run:
        print(f"[dry_run] would parse new lines appended to {results_path}")
        return []

    new_lines = results_path.read_text().splitlines()[lines_before:]
    metrics = []
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        exp_name, *fields = line.split("\t")
        row = {"experiment": exp_name}
        for field in fields:
            key, val = field.split("=")
            row[key] = float(val)
        metrics.append(row)
    return metrics
