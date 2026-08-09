import sys
sys.path.append('/home/ubuntu/HOTFIXR')
from argparse import ArgumentParser
import os
from config import *
from rewards.confidence import compute_confidence_batch
from rewards.proximity import compute_proximity
from rewards.gradient import compute_gradient
from rewards.diversity import compute_diversity
from rewards.answer_variance import compute_answer_variance_batch
import requests
from tqdm import tqdm
import pickle

def normalize(nums):
    nums = np.array(nums)
    nums = (nums - nums.min()) / (nums.max() - nums.min() + 1e-4)
    return nums


def filter(train_data: pd.DataFrame, size, dataset_name, dir_name):
    acquisition_funcs = [compute_gradient] #[compute_proximity, compute_diversity]
    acquisition_names = ['gradient'] #['proximity', 'gradient', 'diversity']
    acquisition_rewards = []

    for j in range(len(acquisition_funcs)):
        acquisition_rewards.append([])
        if not os.path.exists(f'filtering_metadata/{dir_name}/{acquisition_names[j]}.pkl'):
            requests.post(
                url="http://127.0.0.1:5145/start_service",
                json={"service": acquisition_names[j], "kwargs": {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "dataset_name": dataset_name}}
            )

            for i in tqdm(range(len(train_data)), desc=acquisition_names[j]):
                data = {
                    "question": train_data.iloc[i]['question'],
                    "answer": train_data.iloc[i]['answer']
                }

                acquisition_rewards[-1].append(acquisition_funcs[j](data))
            with open(f'filtering_metadata/{dir_name}/{acquisition_names[j]}.pkl', 'wb+') as f:
                pickle.dump(acquisition_rewards[-1], f)
        else:
            with open(f'filtering_metadata/{dir_name}/{acquisition_names[j]}.pkl', 'rb') as f:
                acquisition_rewards[-1] =  pickle.load(f)

    
    acquisition_funcs = [compute_confidence_batch, compute_answer_variance_batch]
    acquisition_names = ['confidence', 'answer_variance']
    for j in range(len(acquisition_funcs)):
        if not os.path.exists(f'filtering_metadata/{dir_name}/{acquisition_names[j]}.pkl'):
            data = []
            for i in tqdm(range(len(train_data)), desc=acquisition_names[j]):
                data.append(train_data.iloc[i]['question'])

            acquisition_rewards.append(acquisition_funcs[j](data, {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "dataset_name": dataset_name}))
            with open(f'filtering_metadata/{dir_name}/{acquisition_names[j]}.pkl', 'wb+') as f:
                pickle.dump(acquisition_rewards[-1], f)
    
    rankings = np.zeros((len(train_data)))
    for j in range(len(acquisition_rewards)):
        rankings += normalize(acquisition_rewards[j])
    
    rankings /= len(acquisition_funcs)
    train_data['rank'] = rankings
    train_data = train_data.sort_values('rank', ascending=False)
    return train_data[:size]


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data", type=str, default="numina")
    argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argparser.add_argument("--dir_name", type=str, default="qwen_metamath")
    argparser.add_argument("--output_file", type=str, default="filtered.parquet")
    argparser.add_argument("--size", type=int, default=1000)
    args = argparser.parse_args()

    train_data = load_data(f"/home/ubuntu/HOTFIXR/data/{args.data}/all.parquet")
    filtered = filter(train_data, args.size, f"/home/ubuntu/HOTFIXR/data/{args.data}/train.parquet", args.dir_name)

    def format(col):
        filtered[col] = filtered[col].apply(lambda x: x.replace("\n", ""))
    
    format('question')
    format('answer')
    format('reasoning')
    file_name = args.output_file
    filtered.to_parquet(file_name)