import HOTFIXR

HOTFIXR.set_variables({
    "CUSTOM_PROMPT": "Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{} </answer> tags.\n\n<question> {q} </question>.",
    "SAVE_RESULTS_DIRECTORY": "result_files/",
    "HF_USERNAME": "ishikauniphore",
    "REPO_ROOT": "/home/ubuntu/"
})

generator_model = HOTFIXR.train_generator(dataset_path="/home/ubuntu/HOTFIXR/data/nemotron", model="Qwen/Qwen2.5-7B-Instruct", job_name="generator_nemotron_qwen7bins_iter0")
sft_data = HOTFIXR.generate_data(dataset_path="/home/ubuntu/HOTFIXR/data/nemotron", output_file="data_nemotron_qwen7bins_iter0.parquet", model=generator_model, size=5000)
student_model = HOTFIXR.train_student(model_path="Qwen/Qwen2.5-7B-Instruct", data="data_nemotron_qwen7bins_iter0.parquet", run_name="student_nemotron_qwen7bins_iter0")
student_model = "ishikauniphore/student_nemotron_qwen7bins_iter0"
metrics = HOTFIXR.evaluate_student(student_model=student_model)
