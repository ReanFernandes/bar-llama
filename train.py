
"""
Steps for fine-tuning
	Instantiate dataset and dataloader class
	Compile the training dataset from the dataset configs
	At this point the entire dataset must be loaded
	Load the train config from hydra
	Load the model and tokenizer
	Make sure the pad token is added to the tokenizer and model vocab
	OPTIONAL: Decide if we are doing lora or qlora, based on that need to load the quantisation configs
	Load lora config and instantiate from cfg.training.lora_config
	Join lora adapter to the model with get_peft_model
	create training args and
    train the model
"""
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
    #--------------------- set global seed ---------------------
    set_global_seed(cfg.seeds.seed)
    logging.info(f"Global seed set to {cfg.seeds.seed}")



    #--------------------- Logging and configuration related info---------------------
     
    logging.info(f"Running evaluation of the {cfg.eval.training_status} model on the {cfg.evaluation_dataset.dataset_label}")
    logging.info(f"Current evaluation config is : \n Prompt type : {cfg.eval.prompt.prompt_type} \n Response type : {cfg.eval.prompt.response_type} \n Explanation type : {cfg.eval.prompt.explanation_type} \n Response Format : {cfg.eval.prompt.response_format} ")
     
    logging.info(f"Running Fine-tuning on the {cfg.training_dataset.dataset_label} dataset")
    logging.info(f"Current training config is : \n Prompt type : {cfg.training.prompt.prompt_type} \n Response type : {cfg.training.prompt.response_type} \n Explanation type : {cfg.training.prompt.explanation_type} \n Response Format : {cfg.training.prompt.response_format} ")
    