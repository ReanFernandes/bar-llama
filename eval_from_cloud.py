"""
Modified version of eval.py that loads adapters from HuggingFace instead of local storage.
This script allows users to easily evaluate models with the uploaded HuggingFace adapters
while maintaining the same configuration-based path structure as the original code.

Usage remains the same as eval.py, with all the same configuration parameters.
"""

import json
import os
from peft import LoraConfig, PeftModel
import hydra
import torch
from huggingface_hub import login, snapshot_download
from omegaconf import DictConfig, OmegaConf
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

# set the current directory as the working directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The model will basically not run if a CPU is used, please ensure you're running this on a GPU
if device != "cuda":
    logging.warning("CPU set for execution, this might take forever to run!!")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV_PATH = os.path.join(os.getcwd(), "final_run_results.csv")
LOCK_FILE_PATH = os.path.join(os.getcwd(), "master_metrics.lock")

# Only hardcode the HuggingFace repository ID
HF_REPO_ID = "HolySaint/bar-Llama-adapters"

def flatten_and_serialize(metrics_dict):
    """
    Flatten the metrics_dict and serialize lists (like num_training_domains) for CSV compatibility.
    """
    flattened = {}
    for key, value in metrics_dict.items():
        if isinstance(value, list):  # Convert lists to JSON strings for CSV compatibility
            flattened[key] = json.dumps(value)
        elif isinstance(value, dict):  # Recursively flatten nested dictionaries
            for sub_key, sub_value in value.items():
                flattened[sub_key] = json.dumps(sub_value) if isinstance(sub_value, list) else sub_value
        else:
            flattened[key] = value
    return flattened

def append_metrics_to_csv(metrics_dict):
    """
    Safely append metrics to a master CSV file using file locking.
    """
    lock = FileLock(LOCK_FILE_PATH)  # Initialize file lock
    with lock:  # Acquire lock
        try:
            # Flatten the metrics dict to handle nested structures
            flattened_data = flatten_and_serialize(metrics_dict)

            # Append to CSV: create file if it doesn't exist
            if not os.path.exists(MASTER_CSV_PATH):
                pd.DataFrame([flattened_data]).to_csv(MASTER_CSV_PATH, index=False)
            else:
                pd.DataFrame([flattened_data]).to_csv(MASTER_CSV_PATH, mode='a', header=False, index=False)
            
            print("Metrics successfully appended to master CSV.")
        except Exception as e:
            print(f"Error while writing to master CSV: {e}")

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

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Add a note indicating we're using the cloud version
    logging.info("Running eval_from_cloud.py - This script loads adapters from HuggingFace Hub")
    
    #-----------------set global seed-----------------
    seed = cfg.seeds.seed
    logging.info(f"Setting global seed to {seed}")
    set_global_seed(seed)
    #-----------------Logging and configuration related information-----------------
    
    logging.info(f"Running evaluation of the {cfg.eval.training_status} model on the {cfg.evaluation_dataset.dataset_label}")
    logging.info(f"Current evaluation config is : \n Prompt type : {cfg.eval.prompt.prompt_type} \n Response type : {cfg.eval.prompt.response_type} \n Explanation type : {cfg.eval.prompt.explanation_type} \n Response Format : {cfg.eval.prompt.response_format} ")
    
    # ----------------- Loading dataset and dataloader related stuff -----------------

    dataset = QuestionDataset(cfg.evaluation_dataset)
    dataloader = DataLoader(dataset, collate_fn=dataset.custom_collate_fn,  **cfg.dataloader)

    # ----------------- Loading the actual model and tokenizer -----------------

    access_token = os.environ.get('HUGGINGFACE_TOKEN') # set this in the environment beforehand
    login(token=access_token) 
    logging.info(f"Loading model : {cfg.model.model_id}")
    ##load the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id, **cfg.tokenizer['kwargs'])
    
    #load model based on quantisation status
    if cfg.eval.quantisation_status == 'quantised_model':
        logging.info("Quantised model selected, loading quantisation config")
        bnb_config = BitsAndBytesConfig(**cfg.quantization)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, quantization_config=bnb_config)
    elif cfg.eval.quantisation_status == 'full_model':
        logging.info("Full model selected, loading model")
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id)
    else:
        raise ValueError("Invalid quantisation status. Please check the config file.")

    # ----------------- Logic block for correct model adapter if needed -----------------
    if cfg.eval.training_status == 'trained':
        # Add pad token to tokenizer and resize model embeddings 
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info(f"Tokenizer special tokens: {tokenizer.special_tokens_map} have been added")
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Build the subfolder path dynamically using the same logic as the original code
        # Level 0: Model label
        subfolder = cfg.model.model_label
        
        # Level 1: Dataset label
        subfolder = os.path.join(subfolder, cfg.dataset.dataset_label)
        
        # Level 2: Experiment label
        subfolder = os.path.join(subfolder, cfg.eval.eval_model_name)
        
        # Log HuggingFace adapter details
        logging.info(f"Loading adapter from Hugging Face Hub: {HF_REPO_ID}")
        logging.info(f"Using dynamically constructed subfolder path: {subfolder}")
        
        # Load the adapter from HuggingFace Hub
        model = PeftModel.from_pretrained(
            model,
            HF_REPO_ID,
            subfolder=subfolder,
            device_map="auto"
        )
        
        logging.info("Fine-tuned model selected, loading adapter")
        cfg.eval.pipeline_available = False
        logging.warning("Inference will be done using model.generate instead of hf pipeline since peft model is not compatible with it")
        model = model.merge_and_unload()
        logging.info("Peft model loaded successfully from HuggingFace Hub")
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
                                    device=device
                                    )

    # ----------------- Loading inference related stuff -----------------
    # model to cuda
    model.to(device)
    # create prompt handler instance
    promptor = PromptHandler(cfg.eval.prompt)

    # create empty data storage list 
    raw_outputs = []

    #----------------- specify correct saving paths for the raw output -----------------
    
    ### sanity check : The output_directory in eval configs is set to ${hydra:runtime.cwd}/model_outputs/raw_responses/<model_label>
    # Start with level -1 : the model name :  
    raw_save_path = os.path.join(cfg.eval.output_directory, cfg.model.model_label)

    ### Start with level 0 : the global seed : Output will be ${hydra:runtime.cwd}/model_outputs/raw_responses/<model_label>/<seed_label>
    raw_save_path = os.path.join(raw_save_path, cfg.seeds.label) # the global seed for the entire pipeline

    ### Level 1 : Number of training samples(and optionally number of domains, but not testing for that currently) : For the timebeing i do not care about testing epoch wise results, so directly skipping it and movbing on to num samples

    raw_save_path = os.path.join(raw_save_path, cfg.dataset.dataset_label) # model has been trained on a certain dataset and that datasets label is used

    ## Level 2 : Training and quantisation status :  results in   ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>/<dataset_label>/<training_status>_<quantisation_status>
    train_quant = cfg.eval.training_status + '_' + cfg.eval.quantisation_status
    raw_save_path = os.path.join(raw_save_path, train_quant)

    ## Level 3 : Decoding method : generation strategy, currently only temp based, but configs also exist for other methods as needed
    # results in This is basically  ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>/<dataset_label>/<training_status>_<quantisation_status>/<generation_label>
    raw_save_path = os.path.join(raw_save_path, cfg.generation.label)

    ## Level 4 : Experiment configuration : Penultimate path, full experiment config name 
    # results in ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>/<dataset_label>/<training_status>_<quantisation_status>/<generation_label>/<experiment_config_label>
    raw_save_path = os.path.join(raw_save_path, cfg.eval.train_config_label)

    ## Level 5 : The actual raw output file : Final path
    # results in ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>/<dataset_label>/<training_status>_<quantisation_status>/<generation_label>/<experiment_config_label>/<dataset_label>.json
    raw_save_path = os.path.join(raw_save_path, cfg.evaluation_dataset.dataset_label + '.json')

    # create the directory structure
    os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
    logging.info(f"Raw output for this run will be saved to {raw_save_path}")
   
    # ----------------- For loop for running inference per question -----------------
    logging.info("Starting inference")
    # for the sake of simplicity im an not batching the inference here rather just evaluating one question at a time. 
    # could possibly be improved by batching the inference and then saving the results in a batched manner.

    for count, data in enumerate(dataloader):
        if count%10 == 0: 
            logging.info(f"Processing question number {count}")
            #dump data to raw outputs
            with open(raw_save_path, 'w', encoding='utf-8') as f:
                json.dump(raw_outputs,f, indent=4)
        
        prompt, ground_truth = promptor.create_prompt(data, 
                                                      mode="eval",
                                                      pipeline_available=cfg.eval.pipeline_available,
                                                      model_name=cfg.model.model_label)
        
        logging.info(f"Domain : {data['domain']}, Question : {data['question_number']}")

        if cfg.eval.pipeline_available is True: 
            sequences = text_gen_pipeline(prompt,
                                          return_full_text=False,
                                          **cfg.generation.kwargs)
            processed_data = {"prompt": prompt,
                              "response":sequences[0]['generated_text'],
                              "ground_truth":ground_truth}
        elif cfg.eval.pipeline_available is False: 
            
            # using manual generation for peft model
            inputs = tokenizer(prompt, return_tensors="pt",add_special_tokens=False)
            inputs.to(model.device)

            #generate model response 
            output = model.generate(**inputs, **cfg.generation.kwargs)

            #Filter out the questions and only extract the model generated response
            sequences = tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            processed_data = {"prompt": prompt,
                              "response":sequences[0],
                              "ground_truth":ground_truth}
        
        raw_outputs.append(processed_data)
    logging.info("Inference complete")
    # ----------------- Save raw outputs -----------------
    # save model outputs 
    logging.info(f"Saving raw outputs to {raw_save_path}")
    with open(raw_save_path, 'w', encoding='utf-8') as f:
        json.dump(raw_outputs,f, indent=4)

    # ----------------- Parse the results using parsing utils -----------------
    # this way might probably make the parsing configs redundant, as they only contain address related to storing the responses.abs
    # thus if i code this functionality in, ican reduce an added complexity by directly ensuring that while the code runs the parsed
    # responses are saved to the correct location
    # this is necessary because there are now new parameters like num_samples and decoding_method that cannot be added directly to the 
    # eval configs and subsequently the parsing configs, thus making it easier.
    # to the raw save path, replace model_outputs with parsed_model_outputs


    parsed_save_path = raw_save_path.replace('raw_responses', 'parsed_responses')
    os.makedirs(os.path.dirname(parsed_save_path), exist_ok=True)
    parsing_metadata_path = raw_save_path.replace('model_outputs', 'parsing_metadata')
    os.makedirs(os.path.dirname(parsing_metadata_path), exist_ok=True)
    #-------------- parsing block to extract the relevant information from the raw outputs -----------------#
    logging.info(f"Starting parsing of raw outputs to {parsed_save_path}")

    # Instantiate parser, since we are usin it in situ, we can directly pass cfg.eval.prompt while instantiating

    parser = ResponseHandler(cfg.eval.prompt)

    for data_item in raw_outputs:
        parser.assess(data_item)

    # create list for parsed outputs 
    parsed_output_list = copy.deepcopy(parser.response_dump)
    # save the parsed outputs
    logging.info(f"Saving parsed outputs to {parsed_save_path}")
    parser.dump_to(parsed_save_path)
    logging.info("Saved parsed outputs")
    
    # save the metadata
    logging.info(f"Saving parsing metadata to {parsing_metadata_path}")
    parser.dump_stats(parsing_metadata_path)
    logging.info("Saved parsing metadata")


    #----------------------- Grading block -----------------#
    # this class takes the parsed responses, which contain the ground truth and the model responses, and then calculates all the metrics
    # that are needed for evaluation
    
    # create a new path for the metrics to be saved, in line with the previous paths
    metrics_save_path = raw_save_path.replace('raw_responses', 'metrics') 
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    logging.info(f"Metrics will be saved to {metrics_save_path}")

    # Create a comparison dict with infor about the current run for easier dataframe construction
    comparison_dict = {
        'config': { 
            'model_name': cfg.model.model_label, # Name used for the model
            'seed': cfg.seeds.label, # Seed used for the run
            'training_status': cfg.eval.training_status, # can be 'trained' or 'untrained'
            'quantisation_status': cfg.eval.quantisation_status, # can be 'quantised_model' or 'full_model'
            'training_dataset': cfg.dataset.dataset_label if cfg.eval.training_status == 'trained' else None, # dataset used for training
            'num_training_samples': cfg.dataset.num_sample_label if cfg.eval.training_status == 'trained' else 0, # number of samples used for training, if applicable otherwise 0 for untrained
            'num_training_domains': list(cfg.dataset.domains) if cfg.eval.training_status == 'trained' else None, # number, or rather the list of names of domains used for training, if applicable
            'randomised_training_samples' : cfg.dataset.randomise_questions if cfg.eval.training_status == 'trained' else None, # whether the samples were selected randomly from the train set
            'generation_strategy': cfg.generation.label, # decoding strategy used for inference
            'prompt_type': cfg.eval.prompt.prompt_type, # type of prompt used, can be 'few_shot' or 'zero_shot'
            'explanation_type': cfg.eval.prompt.explanation_type, # can be 'structured' or 'unstructured'
            'response_type': cfg.eval.prompt.response_type, # can be 'fact_first' or 'answer_first'
            'response_format': cfg.eval.prompt.response_format, # can be 'json' or 'markdown' or 'number_list'
            'evaluation_dataset' : cfg.evaluation_dataset.dataset_label, # label of bar exam test to be evaluated
            'adapter_source': 'huggingface',  # Add this to indicate the adapter was loaded from HuggingFace
            'hf_repo_id': HF_REPO_ID  # Add the repository ID
        },
        'metrics': None
    }
    #instantiate response grader
    grader = ResponseGrader(comparison_dict=comparison_dict) 

    for data_item in parsed_output_list:
        grader.grade_response(data_item)
    
    # compile the metrics, and dump them to save path
    logging.info(f"Metrics calculated, saving backup to {metrics_save_path}") 
    metrics = grader.finalise_metrics() # receives the entire comparison dict with data and exp metadata
    grader.dump_metrics(metrics_save_path)

    # append the metrics to the master csv
    append_metrics_to_csv(metrics)

    logging.info(f"Evaluation completed for {cfg.eval.train_config_label}, on {cfg.evaluation_dataset.dataset_label}")
    logging.info(f"Summary of metrics : {metrics['metrics']}")

if __name__ == "__main__":
    main()
