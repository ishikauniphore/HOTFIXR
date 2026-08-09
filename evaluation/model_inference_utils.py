import os
import sys
import gc
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from argparse import ArgumentParser
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_prompt_template

n = 500

def perform_cheating(llm, sampling_params, dataset_name="/home/ubuntu/HOTFIXR/training_data/selected_nemotron_stem.parquet", english_reasoning="off", only_questions=False):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed['question'])[:n]
    answers = list(grounding_seed['answer'])[:n]

    prompt_template = get_prompt_template("stem", english_reasoning=(english_reasoning == "on"))
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    if not only_questions:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
        outputs = [answer if answer in output[:4] else output for output, answer in zip(outputs, answers)]
    answers = [answer.split("\\boxed{")[-1].split("}")[0].strip() for answer in answers]

    return [{
        "experiment_name": "cheating" if english_reasoning == "off" else "nemotron_stem_english_reasoning",
        "task": "classification",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_nemotron_stem_inference(llm, sampling_params, dataset_name="/home/ubuntu/HOTFIXR/data/nemotron_stem/test.parquet", english_reasoning="off", only_questions=False):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    prompt_template = get_prompt_template("stem", english_reasoning=(english_reasoning == "on"))
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    answers = [answer.split("\\boxed{")[-1].split("}")[0].strip() for answer in answers]
    if not only_questions:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
        outputs = [output[:1].upper() if len(output) > 2 and output[1] == ":" else output for output, answer in zip(outputs, answers)]
    
    
    return [{
        "experiment_name": "nemotron_stem" if english_reasoning == "off" else "nemotron_stem_english_reasoning",
        "task": "classification",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_nemotron_math_inference(llm, sampling_params, dataset_name="/home/ubuntu/HOTFIXR/data/nemotron_math/test.parquet", english_reasoning="off", only_questions=False):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    if english_reasoning == "on":
        prompt_template = lambda q: f"Answer the following question. Output your reasoning in ENGLISH in the <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    else:
        prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    if not only_questions:
        outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip()[7:-1] for output in outputs]
    answers = [answer.split("\\boxed{")[-1].split("}")[0].strip() for answer in answers]
    
    return [{
        "experiment_name": "nemotron_math" if english_reasoning == "off" else "nemotron_math_english_reasoning",
        "task": "math",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_nemotron_chat_inference(llm, sampling_params, dataset_name="/home/ubuntu/HOTFIXR/data/nemotron_chat/test.parquet", english_reasoning="off", only_questions=False):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    if english_reasoning == "on":
        prompt_template = lambda q: f"Answer the following question. Output your reasoning in ENGLISH in the <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    else:
        prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    if not only_questions:
        outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]
    
    return [{
        "experiment_name": "nemotron_chat" if english_reasoning == "off" else "nemotron_chat_english_reasoning",
        "task": "open-ended",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_mhotpot_inference(llm, sampling_params, data_dir="/home/ubuntu/HOTFIXR/data/m_hotpotqa", english_reasoning="off", only_questions=False):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        queries = df['query']
        contexts = df['context']

        if english_reasoning == "on":
            prompt_template = lambda c, q: f"Given some context, the task is to answer the question. Output your reasoning in ENGLISH in <reasoning> </reasoning> tags, then give your final answer in <answer> </answer> tags. Keep the final answer short: 2-4 words only.\n\n<context> {c} </context>\n<question> {q} </question>."
        else:
            prompt_template = lambda c, q: f"Given some context, the task is to answer the question. Output your reasoning in <reasoning> </reasoning> tags, then give your final answer in <answer> </answer> tags. Keep the final answer short: 2-4 words only.\n\n<context> {c} </context>\n<question> {q} </question>."
        prompts = [prompt_template(c, q) for c, q in zip(contexts, queries)]
        outputs = []

        if not only_questions:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            outputs = [output.outputs[0].text.strip() for output in outputs]
        
            outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]
        questions = [f"Context: {c}\nQuestion: {q}" for c, q in zip(contexts, queries)]

        experiments.append({
            "experiment_name": f"mhotpot_{csv_file.split('/')[-1].split('.')[0]}" if english_reasoning == "off" else f"mhotpot_{csv_file.split('/')[-1].split('.')[0]}_english_reasoning",
            "task": "open-ended",
            "questions": questions,
            "contexts": contexts,
            "answers": df['output'],
            "outputs": outputs
        })

    return experiments


def perform_mmmlu_inference(llm, sampling_params, data_dir="/home/ubuntu/HOTFIXR/data/mmmlu", english_reasoning="off", only_questions=False):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        questions = df['questions']
        answers = list(df['answers'])

        prompt_template = get_prompt_template("stem", english_reasoning=(english_reasoning == "on"))
        prompts = [prompt_template(q) for q in questions]
        outputs = []

        if not only_questions:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            outputs = [output.outputs[0].text.strip() for output in outputs]
        
            outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
            outputs = [output[:1].upper() if len(output) > 2 and output[1] == ":" else output for output, answer in zip(outputs, answers)]

        experiments.append({
            "experiment_name": f"mmmlu_{csv_file.split('/')[-1].split('.')[0]}" if english_reasoning == "off" else f"mmmlu_{csv_file.split('/')[-1].split('.')[0]}_english_reasoning",
            "task": "classification",
            "questions": questions,
            "answers": answers,
            "outputs": outputs
        })

    return experiments

def perform_opus_inference(llm, sampling_params, data_dir="/home/ubuntu/HOTFIXR/data/opus-100", english_reasoning="off", only_questions=False):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        questions = df['questions']

        prompt_template = lambda q: f"Answer the question and output your final answer in <answer> </answer> tags.\n<question> {q} </question>."
        prompts = [prompt_template(q) for q in questions]
        outputs = []

        if not only_questions:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            outputs = [output.outputs[0].text.strip() for output in outputs]
        
            outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]

        experiments.append({
            "experiment_name": f"opus_{csv_file.split('/')[-1].split('.')[0]}" if english_reasoning == "off" else f"opus_{csv_file.split('/')[-1].split('.')[0]}_english_reasoning",
            "task": "open-ended",
            "questions": questions,
            "answers": df['answers'],
            "outputs": outputs
        })

    return experiments
