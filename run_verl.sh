# Safer GRPO config to prevent CUDA OOMs
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export VERL_LOGGING_LEVEL=DEBUG

NUM_GPU=6

MODEL_NAME=$1
REWARD_PATH=$2
JOB_NAME=$3
DATASET=$4
SERVICE_NAME=$5
KWARGS_JSON=$6 

curl -X POST http://$localhost:5145/start_service \
    -H "Content-Type: application/json" \
    -d "{\"service\": \"$SERVICE_NAME\", \"kwargs\": $KWARGS_JSON}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/$DATASET/train.parquet \
    data.val_files=./data/$DATASET/test.parquet \
    data.train_batch_size=12 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.critic_warmup=0 \
    trainer.project_name="acquisition_synthesis" \
    trainer.experiment_name=$JOB_NAME \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/dev/shm/grpo_synthesis_models/$JOB_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=False \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score

ray stop --force

# curl -X POST http://$localhost:5145/end_service \
#     -H "Content-Type: application/json"

source run_upload_model.sh $JOB_NAME

