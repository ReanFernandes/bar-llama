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
    #-----------------Logging and configuration related information-----------------
    
    logging.info(f"Running evaluation of the {cfg.eval.training_status} model on the {cfg.dataset.dataset_label}")
    logging.info(f"Current evaluation config is : \n Prompt type : {cfg.eval.prompt.prompt_type} \n Response type : {cfg.eval.prompt.response_type} \n Explanation type : {cfg.eval.prompt.explanation_type} \n Response Format : {cfg.eval.prompt.response_format} ")
    
    # ----------------- Loading dataset and dataloader related stuff -----------------

    dataset = QuestionDataset(cfg.dataset)
    dataloader = DataLoader(dataset, collate_fn=dataset.custom_collate_fn,  **cfg.dataloader)

    # ----------------- Loading the actual model and tokenizer -----------------

    access_token = os.environ.get('HUGGINGFACE_TOKEN') # set this in the environment beforehand
    login(token=access_token) 
    logging.info(f"Loading model : {cfg.model.model_id}")
    4#load the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id, **cfg.tokenizer['kwargs'])
    
    #load model based on quantisation status
    if cfg.eval.quantisation_status == 'quantised_model':
        logging.info("Quantised model selected, loading quantisation config")
        bnb_config = BitsAndBytesConfig(**cfg.quantisation)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, quantization_config=bnb_config)
    elif cfg.eval.quantisation_status == 'full_model':
        logging.info("Full model selected, loading model")
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id)
    else:
        raise ValueError("Invalid quantisation status. Please check the config file.")

    # ----------------- Logic block for correct model adapter if needed -----------------
    # TODO add padding token as was done in the training to ensure proper generation
    if cfg.eval.training_status == 'trained':
        logging.info("Fine-tuned model selected, loading adapter")
        cfg.eval.pipeline_available = False
        logging.warning("Inference will be done using model.generate instead of hf pipeline since peft model is not compatible with it")
        model = PeftModel.from_pretrained(model, cfg.eval.lora_adapter_path)
        model = model.merge_and_unload()
    elif cfg.eval.training_status == 'untrained':
        logging.info("Baseline model selected, no adapter needed")
        logging.warning("Inference will be done using hf pipeline since model.generate consistently generates gibberish")
        # initialize the the hf pipeline for the model
        cfg.eval.pipeline_available = True
        text_gen_pipeline = pipeline(
                                    "text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    )

    # ----------------- Loading inference related stuff -----------------

    # create prompt handler instance
    promptor = PromptHandler(cfg.eval.prompt, cfg.eval.prompt.include_system_prompt)

    # create empty data storage list 
    raw_outputs = []

    #----------------- specify correct saving paths for the raw output -----------------
    """ Slight caveat here that i currently have not that clear of an idea about how i want to go about saving outputs. It is super dependent on the type of experiment im aiming for, e.g. trying diff decoding techniques would need to involve saving the files that way"""
    ### sanity check : The output_directory in eval configs is set to ${hydra:runtime.cwd}/model_outputs/raw_responses
    ### Start with level 0 : the global seed : Output will be ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>
    raw_save_path = os.path.join(cfg.eval.output_directory, cfg.seeds.label) # the global seed for the entire pipeline

    ### Level 1 : 


