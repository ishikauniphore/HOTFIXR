from argparse import ArgumentParser
from rouge_score import rouge_scorer
import pickle

def evaluate_rouge(questions, answers, outputs, rouge_scorer):
    return [rouge_scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(outputs, answers)]

def determine_accuracy(scores, task):
    return [1.0 if s > 0.5 else 0.0 for s in scores]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="ishikauniphore/student_3bT-7bS-v2_nemotron_stem_mcot")
    args = parser.parse_args()

    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'rb') as f:
        experiments = pickle.load(f)

    rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for exp in experiments:
        exp['rouge_l'] = evaluate_rouge(exp['questions'], exp['answers'], exp['outputs'], rouge_scorer)
        exp['rouge_l_accuracy'] = determine_accuracy(exp['rouge_l'], exp['task'])
    
    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'wb+') as f:
        pickle.dump(experiments, f)
    