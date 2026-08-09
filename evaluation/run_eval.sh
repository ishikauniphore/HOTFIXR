export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL_NAME=$1

py model_inference.py --model ${MODEL_NAME}
py model_embed.py --model ${MODEL_NAME}
py model_rouge.py --model ${MODEL_NAME}
py model_laj.py --model ${MODEL_NAME}
