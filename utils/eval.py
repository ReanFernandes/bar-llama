"""
Basic skeleton for running model inference :
steps 
1. Load hydra config
2. load dataset and dataloader related stuff
3. load the actual model and tokenizer.
4. logic block for correct model adapter if needed
5. load inference related stuff
6. for loop for running inference per question
7. save raw outputs 
8. parse the results using parsing utils
9. logic block for saving results to appropriate location based on the adaptor config
10.  grading logic block
11. saving the final evaluation metrics
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
                          BitsAndBytesConfig, pipeline)
from utils.dataloader import QuestionDataset
from utils.prompt_utils import PromptHandler
import logging

# set the current directory as the working directory

os.chdir(os.path.dirname(os.path.abspath(__file__)))
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):