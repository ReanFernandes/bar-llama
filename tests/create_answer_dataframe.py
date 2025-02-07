import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Extended components dictionary with new additions
COMPONENTS = {
    # Components for constructing eval config names
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
    
    # Model-related components
    'model_names': [
        'llama2',
        # 'llama3'  # For future use
    ],
    'training_status': [
        'trained',
        # 'untrained'
    ],
    
    # Other components
    'seeds': [
        'seed_21',
        # 'seed_1337',
        # 'seed_42'
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
        # 'greedy',
        'temp_025',
        'temp_06',
        'temp_09'
    ],
    'evaluation_datasets': [
        'test_set_1',
        'test_set_2',
        # 'val_set_1',
        # 'val_set_2',
        # 'val_set_3',
        # 'val_set_4'
    ],
    'quantisation': [
        'full_model',
        # 'quantised_model'
    ]
}

@dataclass
class ExperimentConfig:
    """Dataclass to hold configuration parameters for an experiment"""
    model_name: str
    training_status: str
    seed: str
    dataset: str
    quantisation: str
    generation: str
    response_format: str
    response_type: str
    prompt_type: str
    explanation_type: str
    eval_dataset: str

    def to_dict(self) -> Dict[str, str]:
        """Convert config to dictionary"""
        return {
            'model_name': self.model_name,
            'training_status': self.training_status,
            'seed': self.seed,
            'dataset': self.dataset,
            'quantisation': self.quantisation,
            'generation': self.generation,
            'response_format': self.response_format,
            'response_type': self.response_type,
            'prompt_type': self.prompt_type,
            'explanation_type': self.explanation_type,
            'eval_dataset': self.eval_dataset
        }

class PathConstructor:
    """Handles construction of file paths based on experiment configurations"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def construct_config_label(self, config: ExperimentConfig) -> str:
        """Construct the configuration label used in the file path"""
        return f"{config.response_format}_{config.response_type}_{config.prompt_type}_{config.explanation_type}"
    
    def get_parsed_response_path(self, config: ExperimentConfig) -> Path:
        """Construct the full path to a parsed response file"""
        train_quant = f"{config.training_status}_{config.quantisation}"
        config_label = self.construct_config_label(config)
        
        path_components = [
            self.base_path,
            "parsed_responses",
            config.seed,
            config.dataset,
            train_quant,
            config.generation,
            config_label,
            f"{config.eval_dataset}.json"
        ]
        
        return Path(*path_components)

class ResponseProcessor:
    """Handles processing of parsed response files"""
    
    @staticmethod
    def process_file(filepath: Path) -> Optional[List[Dict[str, Any]]]:
        """Process a single parsed response file"""
        try:
            with open(filepath, 'r') as f:
                parsed_responses = json.load(f)
            
            processed_results = []
            for response in parsed_responses:
                result = {
                    'question_id': response.get('ground_truth', {}).get('question_number'),
                    'ground_truth_label': response.get('ground_truth', {}).get('correct_answer'),
                    'ground_truth_domain': response.get('ground_truth', {}).get('domain'),
                    'predicted_label': response.get('response', {}).get('chosen_option_label') if response.get('response', {}).get('chosen_option_label') in {'A', 'B', 'C', 'D'} else None,
                    'predicted_domain': response.get('response', {}).get('domain'),
                    # 'confidence': response.get('confidence', None)
                }
                processed_results.append(result)
            return processed_results
        
        except FileNotFoundError:
            logger.debug(f"File not found: {filepath}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing file {filepath}: {str(e)}")
            return None

class QuestionLevelAnalyzer:
    """Main class for conducting question-level analysis"""
    
    def __init__(self, base_path: str, components: Dict[str, List[str]]):
        self.components = components
        self.path_constructor = PathConstructor(base_path)
        self.results_df = None
        
    def _generate_configs(self) -> List[ExperimentConfig]:
        """Generate all possible experiment configurations"""
        configs = []
        
        for model in self.components['model_names']:
            for status in self.components['training_status']:
                for seed in self.components['seeds']:
                    for dataset in self.components['datasets']:
                        for quant in self.components['quantisation']:
                            for gen in self.components['generation']:
                                for resp_format in self.components['response_formats']:
                                    for resp_type in self.components['response_types']:
                                        for prompt_type in self.components['prompt_types']:
                                            for exp_type in self.components['explanation_types']:
                                                for eval_dataset in self.components['evaluation_datasets']:
                                                    config = ExperimentConfig(
                                                        model_name=model,
                                                        training_status=status,
                                                        seed=seed,
                                                        dataset=dataset,
                                                        quantisation=quant,
                                                        generation=gen,
                                                        response_format=resp_format,
                                                        response_type=resp_type,
                                                        prompt_type=prompt_type,
                                                        explanation_type=exp_type,
                                                        eval_dataset=eval_dataset
                                                    )
                                                    configs.append(config)
        return configs

    def process_config(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Process a single configuration"""
        filepath = self.path_constructor.get_parsed_response_path(config)
        results = ResponseProcessor.process_file(filepath)
        
        if results:
            # Add configuration information to each question result
            for result in results:
                result.update(config.to_dict())
                result['file_exists'] = True
        else:
            # Create a placeholder result for missing files
            results = [{
                **config.to_dict(),
                'file_exists': False,
                'question_id': None,
                'ground_truth_label': None,
                'ground_truth_domain': None,
                'predicted_label': None,
                'predicted_domain': None,
                'confidence': None
            }]
            
        return results

    def build_analysis_dataset(self, num_workers: int = 4) -> pd.DataFrame:
        """Build the complete analysis dataset using parallel processing"""
        configs = self._generate_configs()
        all_results = []
        
        logger.info(f"Processing {len(configs)} configurations...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm for progress tracking
            results = list(tqdm(
                executor.map(self.process_config, configs),
                total=len(configs),
                desc="Processing configurations"
            ))
            
            for result_list in results:
                all_results.extend(result_list)
        
        self.results_df = pd.DataFrame(all_results)
        return self.results_df

    def save_results(self, output_path: str):
        """Save the analysis results to a CSV file"""
        if self.results_df is not None:
            self.results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.error("No results to save. Run build_analysis_dataset first.")

def main():
    """Main execution function"""
    # Set up base path and output path
    base_path = "/home/fr/fr_fr/fr_rf1031/bar-llama/model_outputs"
    output_path = "question_level_analysis_results.csv"
    
    # Initialize analyzer
    analyzer = QuestionLevelAnalyzer(base_path, COMPONENTS)
    
    # Build and save dataset
    results_df = analyzer.build_analysis_dataset(num_workers=4)
    analyzer.save_results(output_path)
    
    # Print summary statistics
    logger.info("\nAnalysis Summary:")
    logger.info(f"Total configurations processed: {len(results_df)}")
    logger.info(f"Files found: {results_df['file_exists'].sum()}")
    logger.info(f"Files missing: {(~results_df['file_exists']).sum()}")

if __name__ == "__main__":
    main()