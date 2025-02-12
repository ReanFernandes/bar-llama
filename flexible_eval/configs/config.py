from typing import Dict, List, Any

class ExperimentConfig:
    def __init__(self):
        self.COMPONENTS: Dict[str, List[str]] = {
            'response_formats': [
                'number_list',
                'markdown',
                # Add more as needed
            ],
            'response_types': [
                'answer_first',
                'fact_first',
            ],
            'prompt_types': [
                'few_shot',
                'zero_shot',
            ],
            'explanation_types': [
                'structured',
                'unstructured',
            ],
            'seeds': [
                'seed_42',
                'seed_3991',
            ],
            'datasets': [
                'all_domains_75_samples',
                'all_domains_all_samples',
            ],
            'generation': [
                'greedy',
                'temp_025',
                'temp_06',
                'temp_09',
            ],
            'evaluation_datasets': [
                'test_set_1',
                'test_set_2',
            ],
            'quantisation': [
                'full_model',
            ]
        }
        
        # SLURM configuration
        self.SLURM_CONFIG = {
            'partition': 'gpu_8',
            'time': '1:15:00',
            'mem': '36G',
            'cpus_per_task': 1,
            'gpus': 1,
            'max_concurrent_jobs': 16  # Max jobs running at once within an array
        }
        
        # Base paths
        self.BASE_DIR = "/home/fr/fr_fr/fr_rf1031/bar-llama"
        self.FLEX_DIR = f"{self.BASE_DIR}/flexible_eval"
        self.LOGS_DIR = f"{self.FLEX_DIR}/logs"
        
        # Environment setup
        self.ENV_SETUP = {
            'cuda_version': '12.4',
            'env_path': '/home/fr/fr_fr/fr_rf1031/llama-env/bin/activate',
            'hf_token': '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'
        }
    
    def disable_component_values(self, component: str, values: List[str]) -> None:
        """Disable specific values for a component by commenting them out"""
        if component in self.COMPONENTS:
            self.COMPONENTS[component] = [
                v for v in self.COMPONENTS[component] 
                if v not in values
            ]
    
    def enable_only_component_values(self, component: str, values: List[str]) -> None:
        """Enable only specific values for a component"""
        if component in self.COMPONENTS:
            self.COMPONENTS[component] = [
                v for v in values 
                if v in self.COMPONENTS[component]
            ]
    
    def update_slurm_config(self, updates: Dict[str, Any]) -> None:
        """Update SLURM configuration parameters"""
        self.SLURM_CONFIG.update(updates)

