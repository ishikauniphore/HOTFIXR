import HOTFIXR
import os

REPO_ROOT = "/home/ubuntu/"

HOTFIXR.set_variables({
    "CUSTOM_PROMPT": "Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{} </answer> tags.\n\n<question> {q} </question>.", # set this if you have a custom dataset (besides "nemotron")
    "HF_USERNAME": "ishikauniphore",
    "REPO_ROOT": REPO_ROOT,
    "SAVE_RESULTS_DIRECTORY": os.path.join(REPO_ROOT, "result_files/"),
})

generator_model = HOTFIXR.train_generator(
    dataset_path="/home/ubuntu/HOTFIXR/data/nemotron",
    base_generator_model="Qwen/Qwen2.5-7B-Instruct",
    trained_generator_name="generator_nemotron_qwen7b_round0",
    num_gpu=4,
    cuda_visible_devices="0,1,2,3",
)

sft_data = HOTFIXR.generate_data(
    dataset_path="/home/ubuntu/HOTFIXR/data/nemotron",
    output_file="data_nemotron_qwen7b_round0.parquet",
    question_generation_model=generator_model,
    size=5000
)

student_model = HOTFIXR.train_student(
    base_student_model="Qwen/Qwen2.5-7B-Instruct",
    data_path="data_nemotron_qwen7b_round0.parquet",
    trained_student_name="student_nemotron_qwen7b_round0"
)

metrics = HOTFIXR.evaluate_student(student_model=student_model)
print(metrics)
