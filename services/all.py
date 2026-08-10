import sys
sys.path.append('/home/ubuntu/HOTFIXR/')

import os
from typing import List, Dict
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
import datasets
from sklearn.cluster import MiniBatchKMeans
import services.service_utils as service_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '1'

API_KEY = os.getenv("CONF_API_KEY", "")
localhost = 'localhost'

# vllm
language_model, llm_sampling_params = None, None

# vllm embedding
embedding_model = None
cluster_model: MiniBatchKMeans | None = None
cluster_centers_tensor: torch.Tensor | None = None  # shape: (100, embedding_dim)
NUM_CLUSTERS = 100

# gradient automodel
language_auto_model: AutoModelForCausalLM = None
language_tokenizer: AutoTokenizer = None

class ActivateRequest(BaseModel):
    service: str
    kwargs: Dict = Field(default_factory=dict)

class ActivateResponse(BaseModel):
    status: str
    service: str

class RewardsRequest(BaseModel):
    data: Dict

class RewardsResponse(BaseModel):
    acquisition_reward: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

def clean_up():
    global language_auto_model, language_tokenizer
    language_auto_model = None
    language_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    


############################################# START SERVICE #############################################
@app.post("/start_service", response_model=ActivateResponse)
def start_service(req: ActivateRequest):
    clean_up()
    global language_model, llm_sampling_params, embedding_model, language_auto_model, language_tokenizer

    if "lingualdeficit" in req.service:
        model_name = req.kwargs.get("model_name")
        language_auto_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        language_tokenizer = AutoTokenizer.from_pretrained(model_name)
        language_tokenizer.pad_token_id = language_tokenizer.eos_token_id
        language_tokenizer.padding_size = "left"
        service_utils.init_lingualdeficit_worker(language_tokenizer, language_auto_model)
        return {"status": "ok", "service": req.service}
    
    print('Unknown service requested:', req.service)
    0/0
            
############################################# START SERVICE #############################################

############################################# CONFIDENCE #############################################

@app.post("/lingualdeficit", response_model=RewardsResponse)
def lingualdeficit_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
    return service_utils.lingualdeficit(req)

@app.post("/end_service", response_model=RewardsResponse)
def end_service():
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("all:app", host=localhost, port=5145)
