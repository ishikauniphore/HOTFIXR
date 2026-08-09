import argparse
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned adapter/model")
parser.add_argument("--base_model", type=str, required=True, help="Path to the fine-tuned adapter/model")
parser.add_argument("--skip_push", action="store_true", help="Skip pushing the merged model to the HF Hub")
args = parser.parse_args()

base_model_name = args.base_model
adapter_path = args.model_path
merged_path = args.model_path

model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {merged_path}")

if not args.skip_push:
    # Push to Hugging Face Hub
    hf_username = os.getenv("HF_USERNAME")
    if not hf_username:
        raise RuntimeError("HF_USERNAME is not set.")
    hf_token = os.getenv("HF_TOKEN")
    model_name = args.model_path.split("/")[-1]
    hf_repo_id = f"{hf_username}/{model_name}"

    model.push_to_hub(hf_repo_id, token=hf_token)
    tokenizer.push_to_hub(hf_repo_id, token=hf_token)
    print(f"Model pushed to https://huggingface.co/{hf_repo_id}")
