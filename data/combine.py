import pandas as pd

datasets = ['nemotron_chat', 'nemotron_stem', 'nemotron_math']

for split in ['train']:
    ds = []
    for dataset in datasets:
        temp = pd.read_parquet(f"{dataset}/{split}.parquet", engine='pyarrow')
        ds.append(temp[:167])
    
    df = pd.concat(ds)
    df.to_parquet(f'nemotron/{split}.parquet')