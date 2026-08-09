import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import threading
import queue
import uuid
import torch
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class RewardsResponse(BaseModel):
    acquisition_reward: float


class WorkerQueue:
    """Serializes GPU calls through a single background worker thread, with dynamic batching."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._results: dict = {}
        self._events: dict = {}
        self._lock = threading.Lock()

    def start(self, fn, *args):
        t = threading.Thread(target=self._worker, args=(fn, args), daemon=True)
        t.start()

    def _worker(self, fn, args):
        while True:
            # Block until at least one item is ready
            job_id, req = self._queue.get()
            batch = [(job_id, req)]
            # Drain any additional pending items without blocking
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                results = fn([req for _, req in batch], *args)
            except Exception:
                results = [RewardsResponse(acquisition_reward=-5.0)] * len(batch)

            with self._lock:
                for (jid, _), result in zip(batch, results):
                    self._results[jid] = result
                    self._events[jid].set()

    def submit(self, req) -> RewardsResponse:
        job_id = str(uuid.uuid4())
        event = threading.Event()
        with self._lock:
            self._events[job_id] = event
        self._queue.put((job_id, req))
        event.wait()
        with self._lock:
            result = self._results.pop(job_id)
            del self._events[job_id]
        return result


# --- module-level queues (one per service) ---
_lingualdeficit_queue: WorkerQueue | None = None

def calculate_answer_uncertainty_from_scores(scores, tokenizer):
    """Same top1/top2-logprob-gap confidence as calculate_answer_confidence, but
    walking HF generate()'s per-step `scores` (from output_scores=True) instead of
    vLLM's per-position logprobs dicts."""
    ind = 0
    for i, step_logits in enumerate(scores):
        top1_id = int(step_logits[0].argmax())
        if "answer" in tokenizer.decode([top1_id]):
            ind = i
            break

    top1, top2 = [], []
    for step_logits in scores[ind:]:
        logprobs = F.log_softmax(step_logits[0], dim=-1)
        vals, _ = logprobs.topk(2)
        top1.append(vals[0].item())
        # top2.append(vals[1].item())

    if not top1:
        return -5.0
    diffs = torch.exp(torch.tensor(top1))
    avg_diff = (torch.ones_like(diffs) - diffs).mean() #(1.0/(torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2)))).mean()
    return float(avg_diff)

def _compute_lingualdeficit(reqs, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    results = []
    device = next(model.parameters()).device

    english_only_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in English inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>\n{question}\n</question>\n"
        "<reasoning>"
    )
    mcot_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in the language of the question inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>{question}</question>\n"
        "<reasoning>"
    )

    reasoning_close_ids = tokenizer("</reasoning>", add_special_tokens=False).input_ids

    def find_reasoning_end(ids_list):
        for i in range(len(ids_list) - len(reasoning_close_ids) + 1):
            if ids_list[i:i + len(reasoning_close_ids)] == reasoning_close_ids:
                return i
        return len(ids_list)

    for req in reqs:
        question = req.data['question']

        eng_inputs = tokenizer(english_only_instruction(question), return_tensors='pt').to(device)
        mcot_inputs = tokenizer(mcot_instruction(question), return_tensors='pt').to(device)
        eng_prompt_len = eng_inputs['input_ids'].shape[1]
        mcot_prompt_len = mcot_inputs['input_ids'].shape[1]

        with torch.no_grad():
            eng_gen_out = model.generate(
                **eng_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
            mcot_gen_ids = model.generate(
                **mcot_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        eng_gen_ids = eng_gen_out.sequences
        eng_confidence = calculate_answer_uncertainty_from_scores(eng_gen_out.scores, tokenizer)
        conf = min(eng_confidence, 1.0)

        eng_new_ids = eng_gen_ids[0][eng_prompt_len:].tolist()
        mcot_new_ids = mcot_gen_ids[0][mcot_prompt_len:].tolist()

        eng_reason_end = find_reasoning_end(eng_new_ids)
        mcot_reason_end = find_reasoning_end(mcot_new_ids)

        if eng_reason_end == 0 or mcot_reason_end == 0:
            results.append(RewardsResponse(acquisition_reward=0.0))
            continue

        with torch.no_grad():
            eng_out = model(input_ids=eng_gen_ids, output_hidden_states=True)
            mcot_out = model(input_ids=mcot_gen_ids, output_hidden_states=True)

        # Last layer hidden states: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        eng_hidden = eng_out.hidden_states[-1][0]
        mcot_hidden = mcot_out.hidden_states[-1][0]

        # Mean-pool over the reasoning token positions (generated tokens up to </reasoning>)
        eng_rep = eng_hidden[eng_prompt_len:eng_prompt_len + eng_reason_end].mean(dim=0)
        mcot_rep = mcot_hidden[mcot_prompt_len:mcot_prompt_len + mcot_reason_end].mean(dim=0)

        cos_dist = 1.0 - F.cosine_similarity(eng_rep.unsqueeze(0), mcot_rep.unsqueeze(0)).item()
        rep = cos_dist
        results.append(RewardsResponse(acquisition_reward=conf + rep))

    return results


# --- init functions (called from all.py on service start) ---

def init_lingualdeficit_worker(language_tokenizer, language_auto_model):
    global _lingualdeficit_queue
    _lingualdeficit_queue = WorkerQueue()
    _lingualdeficit_queue.start(_compute_lingualdeficit, language_tokenizer, language_auto_model)

def lingualdeficit(req):
    if _lingualdeficit_queue is None:
        raise RuntimeError("lingualdeficit worker not initialized — call /start_service first")
    return _lingualdeficit_queue.submit(req)
