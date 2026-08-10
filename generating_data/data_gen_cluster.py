import os
import sys
import argparse
import json
import re
import gc

from config import TRAINING_DATA_DIR
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter, defaultdict
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_prompt_template

LANGUAGE_SET = ['English', 'French', 'Spanish', 'Arabic', 'Portuguese', 'Italian']
def parse_reasoning(text):
    try:
        sample_pattern = re.compile(
            r'<reasoning>(.*?)</reasoning>\s*',
            re.DOTALL | re.IGNORECASE
        )
        
        match = sample_pattern.search(text)
        if not match:
            return None
        
        return match.group(1).strip()
    except:
        return None
    
def parse_answer(text):
    try:
        sample_pattern = re.compile(
            r'<answer>(.*?)</answer>\s*',
            re.DOTALL | re.IGNORECASE
        )
        
        match = sample_pattern.search(text)
        if not match:
            return ""
        
        return match.group(1).strip()
    except:
        return ""

def parse_html(generated_samples, synthetic_data, questions):
    for text in generated_samples:
        try:
            sample_pattern = re.compile(
                r'<question>(.*?)</question>\s*',
                re.DOTALL | re.IGNORECASE
            )
            
            match = sample_pattern.search(text)
            if not match:
                continue
            q = match.group(1).strip()
            if len(q) >= 2 * 4096: continue
            synthetic_data.append({"question": q})
            questions.append(q)
        except:
            continue
    
    return synthetic_data, questions

def generate_questions(model_name, dataset_name, size):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    prompts = list(grounding_seed.apply(lambda row: row['prompt'][0]['content'], axis=1))
    prompts = prompts * ((size // len(prompts)) + 1)

    synthetic_data = []
    questions = []

    model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, max_model_len=5096)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=2048)

    while len(synthetic_data) < size:
        outputs = model.generate(prompts, sampling_params)
        generated_samples = [output.outputs[0].text.strip() for output in outputs]
        synthetic_data, questions = parse_html(generated_samples, synthetic_data, questions)
        print(f"{len(synthetic_data)}/{size} questions collected...")

    del model, sampling_params
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()

    return [d['question'] for d in synthetic_data][:size]


# ── Step 2: Generate answers for each question ────────────────────────────────

def apply_chat_template(model_name, prompts, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Reuse the exact same instruction/tag/\boxed{} format used at SFT and eval
    # time (prompts.py), so the labels this script produces are in the format
    # the student is actually trained and scored on.
    mcot_instruction = get_prompt_template(dataset_name, multilingual=True)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": mcot_instruction(p)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

def generate_k_responses(
    questions: list[dict],
    model_name: str,
    dataset_name: str,
    k: int = 16,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    seed: int = 42,
) -> list[dict]:
    """Generate K diverse responses per question. Adds 'generations' field to each dict."""
    prompts = apply_chat_template(model_name, [q["question"] for q in questions], dataset_name)

    print(f"  [vllm] Loading answer model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        disable_custom_all_reduce=True,
        max_model_len=4096,
        enforce_eager=True,
    )
    sp = SamplingParams(n=k, temperature=temperature, max_tokens=max_tokens, seed=seed)

    print(f"  [vllm] Generating {len(prompts)} x {k} = {len(prompts) * k} responses...")
    outputs = llm.generate(prompts, sp)

    for q, out in zip(questions, outputs):
        q["generations"] = [out.outputs[i].text.strip() for i in range(k)]

    # Retry questions where no generation contains <reasoning>
    max_retries = 3
    for attempt in range(max_retries):
        bad_indices = [
            i for i, q in enumerate(questions)
            if any(parse_reasoning(g) is None for g in q["generations"])
        ]

        if not bad_indices:
            break
        print(f"  [vllm] Retry {attempt + 1}: {len(bad_indices)} questions missing <reasoning>, regenerating...")
        retry_prompts = [prompts[i] for i in bad_indices]
        retry_outputs = llm.generate(retry_prompts, sp)
        for idx, out in zip(bad_indices, retry_outputs):
            questions[idx]["generations"] = [out.outputs[i].text.strip() for i in range(k)]

    del llm, sp
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()
    return questions


def cluster_and_pick(
    questions: list[dict],
    dataset_name: str,
    embed_model_name: str = "all-MiniLM-L6-v2",
    distance_threshold: float = 0.35,
) -> list[dict]:
    """
    Embed K generations per question, cluster by cosine similarity,
    pick medoid of largest cluster as pseudo-label answer.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  [cluster] Loading embedding model: {embed_model_name}")
    encoder = SentenceTransformer(embed_model_name)

    for q in tqdm(questions, desc="  [cluster]"):
        all_generations = q.pop("generations")
        full_answers = [g for g in all_generations if parse_reasoning(g) is not None] or all_generations
        texts = [parse_answer(t) for t in full_answers]
        k = len(texts)

        if len(set(texts)) == 1:
            q["answer"] = texts[0]
            q["reasoning"] = parse_reasoning(full_answers[0])
            continue

        embeddings = encoder.encode(texts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, None)
        np.fill_diagonal(dist_matrix, 0.0)

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(dist_matrix)

        counter = Counter(labels)
        counter.pop(-1, None)

        if len(counter) == 0:
            centroid = embeddings.mean(axis=0)
            dists = np.linalg.norm(embeddings - centroid, axis=1)
            best_idx = int(np.argmin(dists))
        else:
            best_label = counter.most_common(1)[0][0]
            mask = labels == best_label
            sub_indices = np.where(mask)[0]
            sub_embeddings = embeddings[mask]
            sub_dist = np.clip(1.0 - cosine_similarity(sub_embeddings), 0.0, None)
            medoid_local = int(np.argmin(sub_dist.mean(axis=1)))
            best_idx = int(sub_indices[medoid_local])

        q["answer"] = texts[best_idx]
        q["reasoning"] = parse_reasoning(full_answers[best_idx])

    return questions


def generate_answers(
    model_name: str,
    questions: list[str],
    dataset_name: str,
    k: int = 16,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    embed_model: str = "all-MiniLM-L6-v2",
    distance_threshold: float = 0.35,
) -> list[str]:
    """
    For each question string, generate K answers, cluster semantically,
    and return the medoid of the largest cluster as the final answer.
    """
    q_dicts = [{"index": i, "question": q} for i, q in enumerate(questions)]
    q_dicts = generate_k_responses(q_dicts, model_name, dataset_name, k=k, temperature=temperature, max_tokens=max_tokens)
    q_dicts = cluster_and_pick(q_dicts, dataset_name, embed_model_name=embed_model, distance_threshold=distance_threshold)
    return [q["answer"] for q in q_dicts], [q["reasoning"] for q in q_dicts]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic questions then label them with answers using vLLM."
    )
    parser.add_argument("--dataset_name", type=str, default="nemotron_stem",
                        help="Path to seed parquet dataset.")
    parser.add_argument("--question_generation_model", type=str, default="ishikauniphore/generator_3bT-7bS-v3_nemotron_stem_mcot",
                        help="HuggingFace model name used for question generation.")
    parser.add_argument("--answer_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name used for answer generation. Defaults to --model_name.")
    parser.add_argument("--size", type=int, default=5,
                        help="Number of question-answer pairs to produce.")
    parser.add_argument("--output_file", type=str, default="data.csv",
                        help="Path for the output CSV. Auto-generated from model/dataset names if not provided.")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of diverse responses to generate per question for clustering.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for answer generation (must be >0 when k>1).")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Sentence-transformer model for semantic clustering.")
    parser.add_argument("--distance_threshold", type=float, default=0.35,
                        help="Agglomerative clustering distance threshold.")
    parser.add_argument("--questions_file", type=str, default=None)
    args = parser.parse_args()

    answer_model = args.answer_model_name

    if args.output_file:
        output_file = args.output_file
    elif "acquisition" in args.question_generation_model:
        output_file = (TRAINING_DATA_DIR,
                       f"AS_{args.question_generation_model.split('/')[-1]}_{args.dataset_name}_{args.size}.csv")
    else:
        output_file = (TRAINING_DATA_DIR,
                       f"Base_{args.question_generation_model.split('/')[-1]}_{args.dataset_name}_{args.size}.csv")

    # if not os.path.exists(output_file):
    if args.questions_file is None:
        print("=== Step 1: Generating questions ===")
        questions = generate_questions(args.question_generation_model, f"/home/ubuntu/HOTFIXR/data/{args.dataset_name}/valid.parquet", args.size)
    else:
        print(f"=== Step 1: Parsing questions from {args.questions_file} ===")
        questions = pd.read_parquet(args.questions_file, engine='pyarrow')['question']
    
    pd.DataFrame({"question": questions}).to_parquet('training_data/questions.parquet', engine='pyarrow')

    print("=== Step 2: Generating answers ===")
    answers, reasonings = generate_answers(
        answer_model, questions, args.dataset_name,
        k=args.k,
        temperature=args.temperature,
        embed_model=args.embed_model,
        distance_threshold=args.distance_threshold,
    )

    pd.DataFrame({"question": questions, "answer": answers, "reasoning": reasonings}).to_parquet(output_file)
    print(f"Saved {len(questions)} Q&A pairs to {output_file}")
    # else:
    #     print(f"Output already exists: {output_file}")


if __name__ == "__main__":
    main()