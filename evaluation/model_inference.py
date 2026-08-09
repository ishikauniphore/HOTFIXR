import os
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
from model_inference_utils import *
import pickle

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="ishikauniphore/student_qwen7bins_nemotron_stem_answerdiff")
    parser.add_argument("--english_reasoning", default="off", type=str)
    args = parser.parse_args()

    ######## LLM INFERENCE ########
    llm = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8, trust_remote_code=True, enforce_eager=True, max_model_len=8096)
    sampling_params = SamplingParams(temperature=0.2, max_tokens=2048)
    experiments = []
    experiments.extend(perform_nemotron_stem_inference(llm, sampling_params, english_reasoning=args.english_reasoning))
    experiments.extend(perform_nemotron_math_inference(llm, sampling_params, english_reasoning=args.english_reasoning))
    experiments.extend(perform_nemotron_chat_inference(llm, sampling_params, english_reasoning=args.english_reasoning))
    experiments.extend(perform_opus_inference(llm, sampling_params))
    experiments.extend(perform_mmmlu_inference(llm, sampling_params, english_reasoning=args.english_reasoning))
    experiments.extend(perform_mhotpot_inference(llm, sampling_params, english_reasoning=args.english_reasoning))

    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'wb+') as f:
        pickle.dump(experiments, f)
