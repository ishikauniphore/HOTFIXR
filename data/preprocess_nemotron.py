# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import numpy as np
import os
from prompt_variations import *
import datasets
from verl.utils.hdfs_io import copy, makedirs

LANGUAGE_SET = ['French', 'Spanish', 'Arabic', 'Portuguese', 'Italian']

def extract_last_boxed(s):
    try:
        temp = s.split("boxed")[-1]
        temp = temp[temp.index("{")+1:temp.rfind("}")]
        return temp
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/home/ubuntu/HOTFIXR/data/nemotron_chat/", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--prompt", type=str, default="current")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "nvidia/Nemotron-Post-Training-Dataset-v2"
    full_dataset = datasets.load_dataset(data_source)

    if "stem" in args.local_save_dir:
        stem = full_dataset['stem']
        stem = stem.select(np.arange(0,30000))
        stem = stem.rename_column("reasoning", "is_reasoning_on")
        stem = stem.map(lambda x: {'query': x['messages'][1]['content']})
        stem = stem.map(lambda x: {'reasoning': x['messages'][2]['content'].split("\n\n\\boxed")[0].strip()})
        stem = stem.map(lambda x: {'response': x['messages'][2]['content'][x['messages'][2]['content'].rfind("\\boxed"):].strip()})
        final_prompt_template = STEM_PROMPT
        final_full_dataset = stem

    elif "math" in args.local_save_dir:
        math = full_dataset['math']
        math = math.select(np.arange(0, 30000))
        math = math.rename_column("reasoning", "is_reasoning_on")
        math = math.map(lambda x: {'query': x['messages'][1]['content']})
        math = math.map(lambda x: {'reasoning': x['messages'][2]['content'].split("\n\n\\boxed")[0].strip()})
        math = math.map(lambda x: {'response': x['messages'][2]['content'][x['messages'][2]['content'].rfind("\\boxed"):].strip()})
        final_prompt_template = MATH_PROMPT
        final_full_dataset = math

    elif "chat" in args.local_save_dir:
        chat = full_dataset['chat']
        chat = chat.select(np.arange(210000, 270000))
        chat = chat.filter(lambda x: "on" in x['reasoning'])
        chat = chat.rename_column("reasoning", "is_reasoning_on")
        chat = chat.map(lambda x: {'query': x['messages'][1]['content']})
        chat = chat.map(lambda x: {'reasoning': x['messages'][2]['content'].split("<think>")[-1].split("</think>")[0].strip()})
        chat = chat.map(lambda x: {'response': x['messages'][2]['content'].split("</think>")[-1].strip()})
        final_prompt_template = CHAT_PROMPT
        final_full_dataset = chat

    else:
        print('um what are you doing?')
        0/0


    only_grounding = lambda q, a: f"""{{
    "question": "{q}",
    "answer": "{a}"
}}"""

    def create_dataset(ds, split):
        records = []
        for i in range(len(ds)):
            q = ds[i]['query']
            a = ds[i]['response']
            r = ds[i]['reasoning']
            records.append({
                "data_source": data_source + "_" + args.prompt,
                "prompt": [{
                    "role": "user",
                    "content": final_prompt_template(q, r, a, LANGUAGE_SET[i % len(LANGUAGE_SET)])
                }],
                "ability": "data_synthesis",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                        "split": split,
                        "index": len(records) + 1,
                        "question": only_grounding(q, a),
                        "grounding_question": q,
                        "grounding_answer": a,
                        "grounding_reasoning": r,
                        "grounding_language": LANGUAGE_SET[i % len(LANGUAGE_SET)]
                }
            })

        return datasets.Dataset.from_list(records)
    
    def create_dataset_jsonl(ds, split):
        records = []
        
        for i in range(len(ds)):
            q = ds[i]['query']
            a = ds[i]['response']

            records.append({
                "prompt": q,
                "completion": a,
                "id": len(records)
            })
            if len(records) >= 100:
                break
        return records


    TRAIN_SIZE = 10000
    TEST_SIZE = 1000
    VALID_SIZE = 10000
    MAX_TOKENS = 4096

    # Use ~4 chars per token as a rough estimate to filter prompts
    def is_short_enough(example):
        q = example.get('query', '').strip().replace("\n", " ")
        a = example.get('response', '').strip().replace("\n", " ")
        r = example.get('reasoning', '').strip().replace("\n", " ")
        content = MATH_PROMPT(q, a, r, "English")
        return len(content) // 2 < MAX_TOKENS

    filtered = final_full_dataset.filter(is_short_enough)
    total_needed = (TRAIN_SIZE + TEST_SIZE + VALID_SIZE)
    assert len(filtered) >= total_needed, (
        f"Not enough examples after filtering: {len(filtered)} < {total_needed}"
    )

    train_dataset = create_dataset(filtered.select(range(TRAIN_SIZE)), "train")
    LANGUAGE_SET.append("English")
    valid_dataset = create_dataset(filtered.select(range(TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE)), "valid")
    test_dataset = create_dataset(filtered.select(range(len(filtered)-TEST_SIZE-1, len(filtered))), "test")
    # all_dataset = create_dataset(filtered.select(range(0, len(filtered)-TEST_SIZE)), "all").select(range(0, 10000))

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    train_dataset.select(range(0, args.train_size)).to_parquet(os.path.join(local_save_dir, "all_train.parquet"))
    valid_dataset.to_parquet(os.path.join(local_save_dir, "valid.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    # all_dataset.to_parquet(os.path.join(local_save_dir, "all.parquet"))

    # records = create_dataset_jsonl(filtered.select(range(TRAIN_SIZE)), "train")
    # for record in records:
    #     with open('/home/ec2-user/prismatic-synthesis/prismatic-synthesis/data/datasets/numina_seed.jsonl', 'a+') as f:
    #         f.write(json.dumps(record))
    #         f.write("\n")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)