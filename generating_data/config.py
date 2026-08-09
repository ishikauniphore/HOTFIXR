import numpy as np
import datasets
import pandas as pd

TRAINING_DATA_DIR = "/home/ubuntu/HOTFIXR/training_data/"

def extract_last_boxed(s):
    try:
        temp = s.split("boxed")[-1]
        temp = temp[temp.index("{")+1:temp.rfind("}")]
        return temp
    except:
        return None

def load_data(data_name):
    df = pd.read_parquet(data_name, engine='pyarrow')

    questions = list(df.apply(lambda row: row['extra_info']['grounding_question'], axis=1))
    answers = list(df.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))
    reasonings = list(df.apply(lambda row: row['extra_info']['grounding_reasoning'], axis=1))
    languages = list(df.apply(lambda row: row['extra_info']['grounding_language'], axis=1))

    return pd.DataFrame.from_dict({
            "index": np.arange(len(df)),
            "question": questions,
            "reasoning": reasonings,
            "answer": answers,
            "language": languages
        })
    
    0/0