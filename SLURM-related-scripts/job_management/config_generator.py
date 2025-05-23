from dataclasses import dataclass
from itertools import product
from typing import Dict, List



@dataclass(frozen=True)  # Make it immutable
class ComponentConfig:
    response_formats: List[str] = (                         # Using tuples instead of lists
        'json',
        # 'number_list', 
        # 'markdown',
    )
    model: List[str] = (                         # models
        'llama2',
        'llama3',
    )
    response_types: List[str] = (
        'answer_first', 
        'fact_first',
    )
    prompt_types: List[str] = (
        'few_shot', 
        'zero_shot',
    )
    explanation_types: List[str] = (
        'structured', 
        'unstructured',
    )
    seeds: List[str] = (
        'seed_206',
        'seed_989',
        # 'seed_786',
        # 'seed_21', 
        # 'seed_1337', 
        # 'seed_42', 
        # 'seed_3991'  # Deprecated
    )
    datasets: List[str] = (
        'all_domains_1_samples', 
        'all_domains_10_samples',
        'all_domains_20_samples',
        'all_domains_75_samples',
        'all_domains_125_samples',
        'all_domains_all_samples',
    )
    generation: List[str] = (
        'greedy', 
        'temp_025', 
        'temp_06', 
        # 'temp_09', #not doing right now to save time on inference jobs, will incorporate later on
    )
    evaluation_datasets: List[str] = (
        'test_set_1',
        'test_set_2',
        # 'val_set_1',
        # 'val_set_2',
        # 'val_set_3',
        # 'val_set_4',
    )
    quantisation: List[str] = (
        'full_model', 
        # 'quantised_model'
    )
    training_status: List[str] = (
        'trained', 
        'untrained',
    )
def generate_train_eval_pairs(components: ComponentConfig = ComponentConfig()) -> List[Dict]:
    """Generate paired configurations for training and evaluation"""
    base_configs = [
        f"{rf}_{rt}_{pt}_{et}"
        for rf, rt, pt, et in product(
            components.response_formats,
            components.response_types,
            components.prompt_types,
            components.explanation_types
        )
    ]
    
    paired_configs = []
    # Remove seed from this product since it's only for eval
    for dataset, base_config,model in product(
        components.datasets,
        base_configs,
        components.model
    ):
        # Training config without seed specification
        train_config = (
            f"model={model} "
            f"tokenizer={model} "
            f"dataset={dataset} "
            f"prompt={base_config} "
            f"train={base_config} "
            f"++train.training_args.per_device_train_batch_size=7 "
            f"++train.training_args.gradient_accumulation_steps=1"
        )

        eval_configs = []
        for eval_set, seed, gen, quant, train_status in product(
            components.evaluation_datasets,
            components.seeds,
            components.generation,
            components.quantisation,
            components.training_status,
            
        ):
            if train_status == 'untrained' and dataset != 'all_domains_1_samples': # We dont want to run untrained inference for different training datasets as that wastes compute
                continue
            eval_config_string = (
                f"seeds={seed} "
                f"model={model} "
                f"tokenizer={model} "
                f"dataset={dataset} "
                f"generation={gen} "
                f"evaluation_dataset={eval_set} "
                f"eval={base_config} "
                f"++eval.quantisation_status={quant} "
                f"++eval.training_status={train_status}"
            )
            eval_configs.append({
                'identifiers': {
                    'eval_dataset': eval_set,
                    'seed': seed,
                    'generation': gen,
                    'quantisation': quant,
                    'training_status': train_status
                },
                'config_string': eval_config_string
            })

        paired_configs.append({
            'identifiers': {
                'base_config': base_config,
                'dataset': dataset,
                'model': model
            },
            'train_config_string': train_config,  # Original string preserved
            'eval_configs': eval_configs
        })    
    return paired_configs