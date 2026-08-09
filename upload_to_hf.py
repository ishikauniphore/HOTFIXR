import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
HF_TOKEN = os.getenv("HF_TOKEN")
USERNAME = os.getenv("HF_USERNAME")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--shorthand", required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model.push_to_hub(f"{USERNAME}/{args.shorthand}", token=HF_TOKEN)
tokenizer.push_to_hub(f"{USERNAME}/{args.shorthand}", token=HF_TOKEN)
