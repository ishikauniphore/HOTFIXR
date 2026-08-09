LATEST_CKPT=$(ls -dt /dev/shm/grpo_synthesis_models/$1/global_step_* | head -1)

python3 merge_fsdp_model.py \
    --local_dir $LATEST_CKPT/actor \
    --target_dir /dev/shm/grpo_synthesis_models/$1/merged

py upload_to_hf.py --model_name "/dev/shm/grpo_synthesis_models/$1/merged" --shorthand "generator_$1"