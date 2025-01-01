
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import hydra
import torch
from huggingface_hub import login, snapshot_download
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, set_seed)
from omegaconf import DictConfig, OmegaConf

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

    #---------------------setting flags for run, need to set these for the code to run properly  ---------------------
    # set this flag as false if we are not quantising the model prior to training
    use_quantisation = False
    if use_quantisation: # this will be the QLORA training case, in case LORA breaks things
        logging.warning("Quantisation will be done on loaded model, THIS IS A QLORA FINE-TUNING RUN")
    elif not use_quantisation: # LORA training case
        logging.warning("No quantisation will be done on loaded model, THIS IS A LORA FINE-TUNING RUN")

    # for the timebeiing i wont work on this becauae it seems too much work, will include it at the end of the code testing
    run_validation_on_epoch_end = False # every epoch, the inference will be run on the latest checkpoint, and results will be logged

    logging.warning(f"Following flags are set for the run : \n Use Quantisation before fine-tuning : {use_quantisation} \n Perform Validation on epoch end : {run_validation_on_epoch_end}")
    #--------------------- Logging and configuration related info---------------------
     
    logging.info(f"Running Fine-tuning on the {cfg.dataset.dataset_label} dataset")
    logging.info(f"Current training config is : \n Prompt type : {cfg.train.prompt.prompt_type} \n Response type : {cfg.train.prompt.response_type} \n Explanation type : {cfg.train.prompt.explanation_type} \n Response Format : {cfg.train.prompt.response_format} ")
    
    #--------------------- Load the training dataset ---------------------# 
    # the training dataset config contains a hardcoded path to a deprecated dataset, and we will load the dataset 
    # based on the explanation type, since there are two main training sets, one which is raw and the other which is distilled
    try: 
        if cfg.train.prompt.explanation_type == "structured": #override dataset path
            cfg.dataset.dataset_path = cfg.distilled_dataset_sft_path
        elif cfg.train.prompt.explanation_type == "unstructured":
            cfg.dataset.dataset_path = cfg.raw_dataset_sft_path
    except Exception as e:
        logging.error(f"Error loading the dataset : {e},  using default Distilled dataset path, stop run if necessary")
        raise e

    # create train dataset instance and prompt handler instance
    unformatted_dataset = QuestionDataset(cfg.dataset) 
    promptor = PromptHandler(cfg.train.prompt)

    for data_item in unformatted_dataset:
        # run over loop to populate the prompt_dict for the training ready samples
        promptor.create_prompt(question_item=data_item,
                                mode="train", #when set to "train", the model format for training text is use to create the prompt
                                store_prompt=True) # this stores the prompt to the prompt_dict of the class

    #copy over the prompt_dict to the train_set,  at this point the dataset should look like a colleciton of {"text": "<training_question_sample_text>"}             
    train_set = promptor.prompt_dict

    # create the dataloader with custom collate function
    #TODO: Complete this when adding validation callback
    if run_validation_on_epoch_end:
        pass
        

    #--------------------- Load the evaluation dataset ---------------------#

    #--------------------- Load the model and tokenizer ---------------------#
    access_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=access_token)

    # Load tokenizer. Here we add the pad token to step away from the usual strategy of setting the pad token to the end token, this lead
    # to the model not learning when to stop generating text
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.model_id, **cfg.tokenizer['kwargs'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # trying to folow llama 2's style, but in theory can be anything
    logging.info(f"Tokenizer {cfg.tokenizer.model_id} loaded successfully")

    # Load model. Note to self: Previously used quantised lora, skipping that now. Hopefully this works
    if use_quantisation:
        bnb_config = BitsAndBytesConfig(**cfg.quantization)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, quantization_config=bnb_config)
        logging.info(f"Quantised model {cfg.model.model_id} loaded successfully")
    elif not use_quantisation :
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id)
        logging.info(f"Model {cfg.model.model_id} loaded successfully")

    # lengthen the models vocabulary to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache=False #KV cache not used during fine_tuning, memory will be use needlessly otherwise
    model.config.pad_token_id = tokenizer.pad_token_id
    # enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) # voodoo stuff, was needed for earlier iterations but might not be needed here, still keeping it for testing before final run

    if use_quantisation:
        model = prepare_model_for_kbit_training(model,bnb_config)
        logging.info("Model prepared for kbit training")
    
    #--------------------- Load the lora config ---------------------#
    target_modules = OmegaConf.to_container(cfg.training.lora_config.target_modules)
    lora_config = LoraConfig(
                r=128,
                lora_alpha=cfg.training.lora_config.lora_alpha,
                lora_dropout=cfg.training.lora_config.lora_dropout,
                bias=cfg.training.lora_config.bias,
                target_modules=target_modules,
                task_type=cfg.training.lora_config.task_type
            )
    logging.info(f"Lora config loaded, following are the details : \n {lora_config}")
    model = get_peft_model(model, lora_config)
    logging.info("Lora adapter added to model")
    model.print_trainable_params()
    #--------------------- Create the pathing logic to store the model ---------------------#
    # The  starting point is ${hydra:runtime.cwd}/sft_adapters
    #Level 1 : Add seed label : ${hydra:runtime.cwd}/sft_adapters/<seed_label>
    raw_adapter_path = os.path.join(cfg.train.lora_adapter_path, cfg.seeds.label)

    #Level 2 : Add training dataset label : ${hydra:runtime.cwd}/sft_adapters/<seed_label>/<training_dataset_label>
    raw_adapter_path = os.path.join(raw_adapter_path, cfg.dataset.dataset_label)

    #Level 3 : Generation strategy : ${hydra:runtime.cwd}/sft_adapters/<seed_label>/<training_dataset_label>/<generation_strategy>
    raw_adapter_path = os.path.join(raw_adapter_path, cfg.generation.label)

    #Level 4 : Experiment label : ${hydra:runtime.cwd}/sft_adapters/<seed_label>/<training_dataset_label>/<generation_strategy>/<model_adapter_name>
    raw_adapter_path = os.path.join(raw_adapter_path, cfg.train.model_adapter_name) # this will be a combo like  "llama2_json_answer_first_few_shot_structured"

    # this should basically be the completed directory path for this models adapter to be saved to. 
    os.makedirs(raw_adapter_path, exist_ok=True)
    #--------------------- Create training arguments ---------------------#

    training_args = TrainingArguments(
        run_name = cfg.train.wandb.run_name,
        output_dir = cfg.train.training_args.output_dir,
        num_train_epochs = cfg.train.training_args.num_train_epochs,
        per_device_train_batch_size = cfg.train.training_args.per_device_train_batch_size,
        gradient_checkpointing = cfg.train.training_args.gradient_checkpointing,
        gradient_accumulation_steps = cfg.train.training_args.gradient_accumulation_steps,

    )
    

if __name__ == "__main__":
    main()