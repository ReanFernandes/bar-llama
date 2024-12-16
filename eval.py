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
from utils.response_utils import ResponseHandler
import logging

# set the current directory as the working directory

os.chdir(os.path.dirname(os.path.abspath(__file__)))
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
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
        # add pad token to tokenizer and resize model embeddings 
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info(f"Tokenizer special tokens: {tokenizer.special_tokens_map} have been added")
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        logging.info("Fine-tuned model selected, loading adapter")
        cfg.eval.pipeline_available = False
        logging.warning("Inference will be done using model.generate instead of hf pipeline since peft model is not compatible with it")
        model = PeftModel.from_pretrained(model, cfg.eval.lora_adapter_path)
        model = model.merge_and_unload()
        logging.info("Peft model loaded successfully")
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

    ### Level 1 : Number of training samples(and optionally number of domains, but not testing for that currently) : For the timebeing i do not care about testing epoch wise results, so directly skipping it and movbing on to num samples

    raw_save_path = os.path.join(raw_save_path, cfg.dataset.dataset_label) # This is basically  ${hydra:runtime.cwd}/model_outputs/raw_responses/<seed_label>/<dataset_label>

    ## Level 2 : Training and quantisation status :  
    train_quant = cfg.eval.training_status + '_' + cfg.eval.quantisation_status
    raw_save_path = os.path.join(raw_save_path, train_quant)

    ## Level 3 : Decoding method : generation strategy, currently only temp based, but configs also exist for other methods as needed
    raw_save_path = os.path.join(raw_save_path, cfg.generation.label)

    ## Level 4 : Experiment configuration : Penultimate path, full experiment config name 
    raw_save_path = os.path.join(raw_save_path, cfg.eval.train_config_label)

    ## Level 5 : The actual raw output file : Final path
    raw_save_path = os.path.join(raw_save_path, cfg.evaluation_dataset.dataset_label + '.json')

    logging.info(f"Raw output for this run will be saved to {raw_save_path}")
   
    # ----------------- For loop for running inference per question -----------------
    logging.info("Starting inference")
    # for the sake of simplicity im an not batching the inference here rather just evaluating one question at a time. 
    # could possibly be improved by batching the inference and then saving the results in a batched manner

    for count, data in enumerate(dataloader):
        if count%10 == 0: 
            logging.info(f"Processing question number {count}")
            #dump data to raw outputs
            with open(raw_save_path, 'w', encoding='utf-8') as f:
                json.dump(raw_outputs,f, indent=4)
        prompt, ground_truth = promptor.create_prompt(data, 
                                                      mode="eval",
                                                      pipeline_available=cfg.eval.pipeline_available)
        logging.info(f"Domain : {data['domain']}, Question : {data['question_number']}")

        if cfg.eval.pipeline_available is True: 
            sequences = text_gen_pipeline(prompt, **cfg.generation.kwargs)
            processed_data = {"prompt": prompt,
                              "response":sequences[0]['generated_text'],
                              "ground_truth":ground_truth}
        elif cfg.eval.pipeline_available is False: 
            # using manual generation for peft model
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs.to(model.device)

            #generate model response 
            output = model.generate(**inputs, **cfg.generation.kwargs)
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
    parsing_metadata_path = raw_save_path.replace('model_outputs', 'parsing_metadata')
    #-------------- parsing block to extract the relevant information from the raw outputs -----------------#
    logging.info(f"Starting parsing of raw outputs to {parsed_save_path}")

    # Instantiate parser, since we are usin it in situ, we can directly pass cfg.eval.prompt while instantiating

    parser = ResponseHandler(cfg.eval.prompt)

    for data_item in raw_outputs:
        parser.assess(data_item)

    # save the parsed outputs
    logging.info(f"Saving parsed outputs to {parsed_save_path}")
    parser.dump_to(parsed_save_path)
    logging.info("Saved parsed outputs")
    
    # save the metadata
    logging.info(f"Saving parsing metadata to {parsing_metadata_path}")
    parser.dump_stats(parsing_metadata_path)
    logging.info("Saved parsing metadata")


    #----------------------- Grading block -----------------#

    











