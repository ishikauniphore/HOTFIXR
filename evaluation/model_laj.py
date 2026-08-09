import os
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import re
import torch
import pandas as pd
import pickle
import numpy as np

PROMETHEUS_PROMPT = lambda q, a, o: f"""###Task Description:
A question and a response will be given. Evaluate the correctness and accuracy of the response based on the reference answer. You MUST give a score between 1 and 5. Respond strictly with only: [RESULT] (score)

###Question:
{q}

###Response to Evaluate:
{a}

###Reference Answer (Score 5):
{o}

###Score Rubrics:
[Is the response correct and accurate?]
Score 1: The response is completely incorrect or irrelevant.
Score 2: The response is mostly incorrect with minor correct elements.
Score 3: The response is partially correct but lacks accuracy or completeness.
Score 4: The response is mostly correct with minor errors.
Score 5: The response is completely correct and accurate.

###Feedback:"""
def evaluate_laj(questions, answers, outputs, llm, sampling_params):

    judge_prompts = [PROMETHEUS_PROMPT(q, a, o) for q, a, o in zip(questions, answers, outputs)]
    judge_outputs = llm.generate(judge_prompts, sampling_params)
    judge_outputs = [o.outputs[0].text.strip() for o in judge_outputs]

    def parse_score(text):
        match = re.search(r'\[RESULT\]\s*(\d)', text)
        return int(match.group(1)) if match else 1.0

    judge_scores = [parse_score(o) for o in judge_outputs]

    return judge_scores

def determine_accuracy(scores, task):
    return [1.0 if s >= 4.0 else 0.0 for s in scores]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="ishikauniphore/student_3bT-7bS-v2_nemotron_stem_mcot")
    args = parser.parse_args()

    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'rb') as f:
        experiments = pickle.load(f)

    
    judge_model_name = 'prometheus-eval/prometheus-7b-v2.0'
    judge_llm = LLM(judge_model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7)
    judge_params = SamplingParams(temperature=0.0, max_tokens=256)
    for exp in experiments:
        exp['judge_score'] = evaluate_laj(exp['questions'], exp['answers'], exp['outputs'], judge_llm, judge_params)
        exp['judge_score_accuracy'] = determine_accuracy(exp['judge_score'], exp['task'])
    
    os.remove(f"eval_{args.model_name.split('/')[-1]}.pkl")

    ######## RECORDING EVALUATION ########
    for exp in experiments:
        exp_name = f"{args.model_name.split('/')[-1]}_{exp['experiment_name']}"
        prompts = [p.replace("\n", " ") for p in exp['questions']]
        answers = [a.replace("\n", " ") for a in exp['answers']]
        outputs = [o.replace("\n", " ") for o in exp['outputs']]

        results_dir = os.environ.get("HOTFIXR_SAVE_RESULTS_DIRECTORY")
        os.makedirs(results_dir, exist_ok=True)
        pd.DataFrame.from_dict({
            "question": prompts,
            "predictions": outputs,
            "references": answers,
            "embed_sim": exp['embed_sim'],
            "embed_sim_accuracy": exp['embed_sim_accuracy'],
            "rouge_l": exp['rouge_l'],
            "rouge_l_accuracy": exp['rouge_l_accuracy'],
            "judge_score": exp['judge_score'],
            "judge_score_accuracy": exp['judge_score_accuracy'],
        }).to_csv(f"{results_dir}/{exp_name}.csv", sep="|")


        with open(f"{results_dir}/results.txt", 'a+') as f:
            f.write(f"{exp_name}\tembed_sim={np.array(exp['embed_sim_accuracy']).mean()*100:.4f}\trouge_l={np.array(exp['rouge_l_accuracy']).mean()*100:.4f}\tjudge={np.array(exp['judge_score_accuracy']).mean()*100:.4f}\n")
    with open(f"{results_dir}/results.txt", 'a+') as f:
        f.write("\n\n\n")
    ######## RECORDING EVALUATION ########
