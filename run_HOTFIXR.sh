export VLLM_USE_V1=0
DATASET="nemotron"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORTHAND="qwen7bins"


### STEP 1: Acquisition training
REWARD="lingualdeficit"
rm -rf /dev/shm/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"${MODEL_NAME}\", \"dataset_name\": \"/home/ubuntu/HOTFIXR/data/${DATASET}/train.parquet\"}"
source run_verl.sh "${MODEL_NAME}" "rewards/${REWARD}.py" "${MODEL_SHORTHAND}_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"


export CUDA_VISIBLE_DEVICES=0,1,2,3
# STEP 2: Dataset generations
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_${MODEL_SHORTHAND}_${DATASET}_${REWARD}" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/HOTFIXR/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet" \
    --size 5000 --k 4

# # ### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "${MODEL_NAME}" \
    --file_name "/home/ubuntu/HOTFIXR/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet"
py merge.py --model_path "/dev/shm/sft_models/${MODEL_SHORTHAND}_${DATASET}_${REWARD}" --base_model "${MODEL_NAME}"

source run_eval.sh "${HF_USERNAME}/student_${MODEL_SHORTHAND}_${DATASET}_${REWARD}"
cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3
source keep_alive.sh