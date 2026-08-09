from vllm import LLM
from argparse import ArgumentParser
import torch
import pickle

def evaluate_embed_sim(questions, answers, outputs, embedding_model):
    pred_embs = embedding_model.embed(outputs)
    pred_embs = torch.tensor([o.outputs.embedding for o in pred_embs])

    ref_embs = embedding_model.embed(answers)
    ref_embs = torch.tensor([o.outputs.embedding for o in ref_embs])

    embed_sim = (pred_embs @ ref_embs.T).diag()

    return embed_sim

def determine_accuracy(scores, task):
    if "classification" in task:
        return [1.0 if s > 0.9 else 0.0 for s in scores]
    else:
        return [1.0 if s > 0.7 else 0.0 for s in scores]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="ishikauniphore/student_3bT-7bS-v2_nemotron_stem_mcot")
    args = parser.parse_args()

    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'rb') as f:
        experiments = pickle.load(f)

    embedding_model = LLM('Qwen/Qwen3-Embedding-8B', task="embed", tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.5)
    for exp in experiments:
        exp['embed_sim'] = evaluate_embed_sim(exp['questions'], exp['answers'], exp['outputs'], embedding_model)
        exp['embed_sim_accuracy'] = determine_accuracy(exp['embed_sim'], exp['task'])
    
    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'wb+') as f:
        pickle.dump(experiments, f)
