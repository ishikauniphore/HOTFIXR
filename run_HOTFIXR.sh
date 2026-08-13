#########
###
### Please remember to run `source run_services.sh` before running this script to start the reward function API.
###
#########

export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

HF_USERNAME="ishikauniphore"

### STEP 1: Acquisition training
REWARD="lingualdeficit"
rm -rf /dev/shm/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"Qwen/Qwen2.5-7B-Instruct\", \"dataset_name\": \"/home/ubuntu/HOTFIXR/data/nemotron/train.parquet\"}"
source run_verl.sh "Qwen/Qwen2.5-7B-Instruct" "rewards/${REWARD}.py" "generator_nemotron_qwen7b" "nemotron" "${REWARD}" "$GRPO_KWARGS"


# STEP 2: Dataset generations
py generating_data/data_gen_cluster.py \
    --dataset_name "nemotron" \
    --question_generation_model "${HF_USERNAME}/generator_nemotron_qwen7b" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/data_nemotron_qwen7b.parquet" \
    --size 5000 --k 4

# # ### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --base_student_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "/home/ubuntu/data_nemotron_qwen7b.parquet" \
    --trained_student_name "student_nemotron_qwen7b"
py merge.py --model_path "/dev/shm/sft_models/student_nemotron_qwen7b" --base_student_model "${HF_USERNAME}/student_nemotron_qwen7b"

source run_eval.sh "${HF_USERNAME}/student_nemotron_qwen7b"
cd ..

notify "experiment done with round 2"