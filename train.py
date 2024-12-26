

import json
import os
from peft import LoraConfig, PeftModel
import hydra
import torch
from huggingface_hub import login, snapshot_download
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline, set_seed)
import random 
import numpy as np
from utils.dataloader import QuestionDataset
from utils.prompt_utils import PromptHandler
from utils.response_utils import ResponseHandler, ResponseGrader
import logging
import copy
from filelock import FileLock
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed):
    """
    Set global seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in PyTorch (optional, might impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):