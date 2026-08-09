import sys
sys.path.append('/home/ubuntu/HOTFIXR/')
from argparse import ArgumentParser
import os
from config import *

def filter(train_data: pd.DataFrame, size):
    return train_data.sample(n=size)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data", type=str, default="nemotron_stem")
    argparser.add_argument("--size", type=int, default=1000)
    argparser.add_argument("--file", type=str, default="selection.parquet")
    args = argparser.parse_args()

    train_data = load_data(f"/home/ubuntu/HOTFIXR/data/{args.data}/train.parquet")

    filtered = filter(train_data, args.size)

    file_name = args.file
    
    def format(col):
        filtered[col] = filtered[col].apply(lambda x: x.replace("\n", ""))
    
    format('question')
    format('answer')
    format('reasoning')
    filtered.to_parquet(file_name)