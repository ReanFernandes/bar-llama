
"""
Steps for fine-tuning
	Instantiate dataset and dataloader class
	Compile the training dataset from the dataset configs
	At this point the entire dataset must be loaded
	Load the train config from hydra
	Load the model and tokenizer
	Make sure the pad token is added to the tokenizer and model vocab
	OPTIONAL: Decide if we are doing lora or qlora, based on that need to load the quantisation configs
	Load lora config and instantiate from cfg.train.lora_config
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
import wandb
import random 
import numpy as np
from utils.dataloader import QuestionDataset
from utils.prompt_utils import PromptHandler
from utils.response_utils import ResponseHandler, ResponseGrader
import logging
import copy
from filelock import FileLock
import pandas as pd
from trl import SFTTrainer
from datasets import  Dataset
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
    #--------------------- set global seed (VALIDATED)---------------------
    set_global_seed(cfg.seeds.seed)
    logging.info(f"Global seed set to {cfg.seeds.seed}")
    cfg.train.wandb.run_name += f"_{cfg.seeds.label}_{cfg.model.model_label}" # add seed name to wandb run name to separate diff seed runs
    #---------------------setting flags for run, need to set these for the code to run properly (VALIDATED)  ---------------------
    # set this flag as false if we are not quantising the model prior to training
    use_quantisation = True
    if use_quantisation: # this will be the QLORA training case, in case LORA breaks things
        logging.warning("Quantisation will be done on loaded model, THIS IS A QLORA FINE-TUNING RUN")
    elif not use_quantisation: # LORA training case
        logging.warning("No quantisation will be done on loaded model, THIS IS A LORA FINE-TUNING RUN")

    # for the timebeiing i wont work on this becauae it seems too much work, will include it at the end of the code testing
    run_validation_on_epoch_end = False # every epoch, the inference will be run on the latest checkpoint, and results will be logged

    logging.warning(f"Following flags are set for the run : \n Use Quantisation before fine-tuning : {use_quantisation} \n Perform Validation on epoch end : {run_validation_on_epoch_end}")
    #--------------------- Logging (local and WandB) and configuration related info (VALIDATED)---------------------
     
    logging.info(f"Running Fine-tuning on the {cfg.dataset.dataset_label} dataset")
    logging.info(f"Current training config is : \n Prompt type : {cfg.train.prompt.prompt_type} \n Response type : {cfg.train.prompt.response_type} \n Explanation type : {cfg.train.prompt.explanation_type} \n Response Format : {cfg.train.prompt.response_format} ")
    # wandb.login(key=os.environ.get('WANDB_token'))
    # #create new customised name for run name using the seed and also the train set label
    # run_name = f"{cfg.seeds.label}_{cfg.dataset.dataset_label}_{cfg.generation.label}_{cfg.train.train_config_label}"

    # wandb.init(project=cfg.train.wandb.project_name, name=run_name)
    #--------------------- Load the training dataset (VALIDATED)---------------------# 
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
        try:
            promptor.create_prompt(question_item=data_item,
                                mode="train", #when set to "train", the model format for training text is use to create the prompt
                                store_prompt=True,# this stores the prompt to the prompt_dict of the class
                                model_name=cfg.model.model_label) # model_label 60
        except Exception as e:
            logging.error(f"Error creating prompt for {data_item} : {e}, skipping this item")
            continue
    #copy over the prompt_dict to the train_set,  at this point the dataset should look like a colleciton of {"text": "<training_question_sample_text>"}             
    train_set = promptor.prompt_dict

    # create the dataloader with custom collate function
    #TODO: Complete this when adding validation callback
    if run_validation_on_epoch_end:
        pass
        

    #--------------------- Load the evaluation dataset ---------------------#
    # not implemented yet, might not do so in the interest of time but will try to
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
    target_modules = OmegaConf.to_container(cfg.train.lora_config.target_modules)
    lora_config = LoraConfig(
                r=cfg.train.lora_config.r,
                lora_alpha=cfg.train.lora_config.lora_alpha,
                lora_dropout=cfg.train.lora_config.lora_dropout,
                bias=cfg.train.lora_config.bias,
                target_modules=target_modules,
                task_type=cfg.train.lora_config.task_type
            )
    logging.info(f"Lora config loaded, following are the details : \n {lora_config}")
    model = get_peft_model(model, lora_config)
    logging.info("Lora adapter added to model")
    # model.print_trainable_params()
    #--------------------- Create the pathing logic to store the model (VALIDATED)---------------------#
    # The  starting point is ${hydra:runtime.cwd}/sft_adapters
    # #Level 1 : Start with dataset label : ${hydra:runtime.cwd}/sft_adapters/<training_dataset_label>
    # raw_adapter_path = os.path.join(cfg.train.lora_adapter_path, cfg.dataset.dataset_label)

    # # #Level 2 : Generation strategy : ${hydra:runtime.cwd}/sft_adapters/<training_dataset_label>/<generation_strategy>
    # # raw_adapter_path = os.path.join(raw_adapter_path, cfg.generation.label)

    # #Level 3 : Experiment label : ${hydra:runtime.cwd}/sft_adapters/<training_dataset_label>/<model_adapter_name>
    # raw_adapter_path = os.path.join(raw_adapter_path, cfg.train.model_adapter_name) # this will be a combo like  "llama2_json_answer_first_few_shot_structured"

    # # this should basically be the completed directory path for this models adapter to be saved to. 
    ###UPDATED LOGIC THAT DECOUPLES THE SEED FROM THE SAVED ADAPTER AND ADDS MODEL LEVEL###
    #Level 0 : Start with model label : ${hydra:runtime.cwd}/sft_adapters/<model_label>
    raw_adapter_path = os.path.join(cfg.train.lora_adapter_path, cfg.model.model_label)

    #Level 1 : Dataset label : ${hydra:runtime.cwd}/sft_adapters/<model_label>/<training_dataset_label>
    raw_adapter_path = os.path.join(raw_adapter_path, cfg.dataset.dataset_label)

    #Level 2 : Experiment label : ${hydra:runtime.cwd}/sft_adapters/<model_label>/<training_dataset_label>/<model_adapter_name>
    raw_adapter_path = os.path.join(raw_adapter_path, cfg.train.model_adapter_name) # this will be a combo like "llama2_json_answer_first_few_shot_structured"

    # Create directory path for this model's adapter
    os.makedirs(raw_adapter_path, exist_ok=True)
    try:
        logging.info(f"Model adapter will be saved to {raw_adapter_path}")
    except:
        pass
    #--------------------- Create training arguments ---------------------#

    training_args = TrainingArguments(
        run_name = cfg.train.wandb.run_name,
        output_dir = cfg.train.training_args.output_dir,
        num_train_epochs = 10,
        per_device_train_batch_size = cfg.train.training_args.per_device_train_batch_size,
        gradient_checkpointing = cfg.train.training_args.gradient_checkpointing,
        gradient_accumulation_steps = cfg.train.training_args.gradient_accumulation_steps,
        optim = cfg.train.training_args.optim,
        save_strategy = "no",  #  no-save to make sure the i dont fill up with multiple checkpoints
        logging_steps = cfg.train.training_args.logging_steps,
        learning_rate = cfg.train.training_args.learning_rate,
        weight_decay = cfg.train.training_args.weight_decay,
        fp16 = cfg.train.training_args.fp16,
        bf16 = cfg.train.training_args.bf16,
        max_grad_norm = max(cfg.train.training_args.max_grad_norm, 0.5), 
        max_steps = cfg.train.training_args.max_steps,
        group_by_length = cfg.train.training_args.group_by_length,
        lr_scheduler_type = cfg.train.training_args.lr_scheduler_type,
        warmup_ratio = cfg.train.training_args.warmup_ratio,  # warmup for cosine annealing
        report_to = "wandb",
        seed=cfg.seeds.seed # have to set this here, since other things have to be seeded before this class is instantiated
        # Additional parameters for cosine scheduling
        # num_cycles = cfg.train.training_args.num_cycles  # for cosine_with_restarts
    )
    hf_format_train_set =  Dataset.from_list(train_set)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset = hf_format_train_set,
        # dataset_text_field='text',
        peft_config = lora_config,
        # max_seq_length = None,
        args = training_args,
        # packing = False
    )
    trainer.train()
    trainer.save_model(raw_adapter_path)
    logging.info(f"Finetuning complete, model saved to {raw_adapter_path}")
    wandb.finish()
    
    

if __name__ == "__main__":
    main()