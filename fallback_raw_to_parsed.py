"""
This file functions as a fallback in case anything breaks after the raw responses are collected. It 
1. Serves as a starting point to debug response parsing for whatever edge cases in the raw responses break parsing
2. Serves as a backup to just run and parse all the saved raw responses
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import itertools
from pathlib import Path
import json
import logging
from utils.response_utils import ResponseHandler, ResponseGrader
import copy
from filelock import FileLock
import pandas as pd
import os


COMPONENTS = {
  'response_formats': [
      'json', 
      'number_list', 
      'markdown'
  ],
  'response_types': [
      'answer_first', 
      'fact_first'
  ],
  'prompt_types': [
      'few_shot', 
      'zero_shot'
  ],
  'explanation_types': [
      'structured', 
      'unstructured'
  ],
  'seeds': [
      'seed_21',
    #   'seed_1337', 
    #   'seed_42'
  ],
  'datasets': [
      'all_domains_1_samples',
      'all_domains_10_samples',
      'all_domains_20_samples',
      'all_domains_75_samples', 
      'all_domains_125_samples',
      'all_domains_all_samples'
  ],
  'generation': [
    #   'greedy',
    #   'temp_025',
      'temp_06',
    #   'temp_09'
  ],
  'evaluation_datasets': [
      'test_set_1',
      'test_set_2'
  ],
  'quantisation': [
      'full_model',
    #   'quantised_model'
  ]
}
os.chdir(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV_PATH = os.path.join(os.getcwd(), "master_metrics_after_the_fact.csv")
LOCK_FILE_PATH = os.path.join(os.getcwd(), "master_metrics.lock")

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
def load_full_config(combo):
    prompt_name = f"{combo['response_format']}_{combo['response_type']}_{combo['prompt_type']}_{combo['explanation_type']}"
    
    overrides = [
        "dataset=" + combo['dataset'],  # Removed + prefix
        "evaluation_dataset=" + combo['eval_dataset'],
        "prompt=" + prompt_name,
        "eval=" + prompt_name,
        "train=" + prompt_name,
        "generation=" + combo['gen'],
        "parsing=" + prompt_name,
        "seeds=" + combo['seed']
    ]

    try:
        cfg = hydra.compose(config_name="config", overrides=overrides)
        return cfg
    except Exception as e:
        logging.error(f"Config load failed for {prompt_name}: {e}")
        return None
# def load_full_config(combo):
#    prompt_name = f"{combo['response_format']}_{combo['response_type']}_{combo['prompt_type']}_{combo['explanation_type']}"
   
#    overrides = [
#        f"+dataset={combo['dataset']}",
#        f"+evaluation_dataset={combo['eval_dataset']}",
#        f"+prompt={prompt_name}",
#        f"+eval={prompt_name}",
#        f"+train={prompt_name}",
#        f"+generation={combo['gen']}",
#        f"+parsing={prompt_name}",
#        f"+seeds={combo['seed']}"
#    ]

#    try:
#        cfg = hydra.compose(config_name="config", overrides=overrides)
#        return cfg
#    except Exception as e:
#        logging.error(f"Config load failed: {e}")
#        return None
def get_raw_path(cfg, combo):
   raw_path = Path(cfg.eval.output_directory) / combo['seed'] / combo['dataset'] / \
          f"trained_{combo['quant']}" / combo['gen'] / \
          cfg.eval.train_config_label / f"{combo['eval_dataset']}.json"
   
   parsed_path = Path(str(raw_path).replace("raw_responses", "parsed_responses"))
   metrics_path = Path(str(raw_path).replace("raw_responses", "metrics")) 
   
   return raw_path, parsed_path, metrics_path

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
   logging.basicConfig(level=logging.INFO)
   
   combos = [{
       'response_format': rf,
       'response_type': rt,
       'prompt_type': pt,
       'explanation_type': et,
       'seed': seed,
       'dataset': dataset,
       'gen': gen,
       'eval_dataset': eval_set,
       'quant': quant
   } for rf, rt, pt, et, seed, dataset, gen, eval_set, quant in 
   itertools.product(
       COMPONENTS['response_formats'],
       COMPONENTS['response_types'],
       COMPONENTS['prompt_types'],
       COMPONENTS['explanation_types'],
       COMPONENTS['seeds'],
       COMPONENTS['datasets'],
       COMPONENTS['generation'],
       COMPONENTS['evaluation_datasets'],
       COMPONENTS['quantisation']
   )]

   total = len(combos)
   processed = skipped = failed = 0
   
   for combo in combos:
       try:
           cfg = load_full_config(combo)
           if not cfg:
               failed += 1
               continue

           raw_path, parsed_path, metrics_path = get_raw_path(cfg, combo)
           if not raw_path.exists():
               logging.info(f"Skipping missing: {raw_path}")
               skipped += 1
               continue

           logging.info(f"Processing [{processed}/{total}]: {raw_path.parts[-2]} for {raw_path.parts[-5]} ")

           try:
               with open(raw_path) as f:
                   raw_outputs = json.load(f)
           except json.JSONDecodeError as e:
               logging.error(f"JSON parse error {raw_path}: {e}")
               failed += 1
               continue

        #    parsed_path = raw_path.parent / raw_path.name.replace('raw_responses', 'parsed_responses')
        #    metrics_path = raw_path.parent / raw_path.name.replace('raw_responses', 'metrics')
           parsed_path.parent.mkdir(parents=True, exist_ok=True)

           parser = ResponseHandler(cfg.eval.prompt)
           parse_failed = 0
           for data in raw_outputs:
               try:
                   parser.assess(data)
               except Exception as e:
                   logging.error(f"Parse error in {raw_path}: {e}")
                   parse_failed += 1

           if parse_failed == len(raw_outputs):
               logging.error(f"All responses failed parsing in {raw_path}")
               failed += 1
               continue

           parsed_outputs = copy.deepcopy(parser.response_dump)
           parser.dump_to(str(parsed_path))

           comparison_dict = {
               'config': {
                   'model_name': cfg.model.model_label,
                   'seed': cfg.seeds.label,
                   'training_status': 'trained',
                   'quantisation_status': cfg.eval.quantisation_status,
                   'training_dataset': cfg.dataset.dataset_label,
                   'num_training_samples': cfg.dataset.num_sample_label,
                   'num_training_domains': list(cfg.dataset.domains),
                   'randomised_training_samples': cfg.dataset.randomise_questions,
                   'generation_strategy': cfg.generation.label,
                   'prompt_type': cfg.eval.prompt.prompt_type,
                   'explanation_type': cfg.eval.prompt.explanation_type,
                   'response_type': cfg.eval.prompt.response_type,
                   'response_format': cfg.eval.prompt.response_format,
                   'evaluation_dataset': cfg.evaluation_dataset.dataset_label
               },
               'metrics': None
           }

           grader = ResponseGrader(comparison_dict)
           for data in parsed_outputs:
               grader.grade_response(data)
           
           metrics = grader.finalise_metrics()
           grader.dump_metrics(str(metrics_path))
           append_metrics_to_csv(metrics)
           
           processed += 1
           logging.info(f"Processed {raw_path.name}")
           
       except Exception as e:
           logging.error(f"Failed {combo}: {e}")
           failed += 1
           continue

   logging.info(f"Complete. Total: {total}, Processed: {processed}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
   main()