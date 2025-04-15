import subprocess
from itertools import product

# Configuration components
COMPONENTS = {
    'model_labels': ['llama2', 'llama3'],
    'response_formats': ['json', 'markdown', 'number_list'],
    'response_types': ['answer_first', 'fact_first'],
    'prompt_types': ['few_shot', 'zero_shot'],
    'explanation_types': ['structured', 'unstructured'],
    'seeds': ['seed_42', 'seed_123', 'seed_206', 'seed_456', 'seed_789', 'seed_989'],
    'datasets': [
        'all_domains_1_samples', 
        'all_domains_10_samples',
        'all_domains_20_samples',
        'all_domains_75_samples',
        'all_domains_125_samples',
        'all_domains_all_samples'
    ],
    'evaluation_datasets': [
        'test_set_1',
        'test_set_2'
    ],
    'generation': ['greedy', 'temp_025', 'temp_06'],
    'quantisation': ['full_model', 'quantised_model'],
    'training_status': ['trained', 'untrained']
}

def get_user_selection(options, prompt):
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"  [{i}] {option}")
    
    while True:
        try:
            choice = input(f"Select [0-{len(options)-1}]: ")
            index = int(choice)
            if 0 <= index < len(options):
                return options[index]
            print(f"Please enter a number between 0 and {len(options)-1}")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("HuggingFace Adapter Evaluation")
    print("==============================")
    
    # Get user selections for each component
    model_label = get_user_selection(COMPONENTS['model_labels'], "Select model:")
    dataset_label = get_user_selection(COMPONENTS['datasets'], "Select training dataset:")
    response_format = get_user_selection(COMPONENTS['response_formats'], "Select response format:")
    response_type = get_user_selection(COMPONENTS['response_types'], "Select response type:")
    prompt_type = get_user_selection(COMPONENTS['prompt_types'], "Select prompt type:")
    explanation_type = get_user_selection(COMPONENTS['explanation_types'], "Select explanation type:")
    evaluation_dataset = get_user_selection(COMPONENTS['evaluation_datasets'], "Select evaluation dataset:")
    generation = get_user_selection(COMPONENTS['generation'], "Select generation strategy:")
    seed = get_user_selection(COMPONENTS['seeds'], "Select seed:")
    quantisation = get_user_selection(COMPONENTS['quantisation'], "Select quantisation status:")
    training_status = get_user_selection(COMPONENTS['training_status'], "Select training status:")
    
    # Generate the adapter name based on your pattern
    adapter_name = f"{model_label}_{response_format}_{response_type}_{prompt_type}_{explanation_type}"
    
    # Build the command
    command = [
        "python3", "eval_from_cloud.py",
        f"model.model_label={model_label}",
        f"dataset.dataset_label={dataset_label}",
        f"evaluation_dataset.dataset_label={evaluation_dataset}",
        f"eval.eval_model_name={adapter_name}",
        f"eval.prompt.response_format={response_format}",
        f"eval.prompt.response_type={response_type}",
        f"eval.prompt.prompt_type={prompt_type}",
        f"eval.prompt.explanation_type={explanation_type}",
        f"generation.label={generation}",
        f"seeds.label={seed}",
        f"eval.quantisation_status={quantisation}",
        f"eval.training_status={training_status}",
        f"eval.train_config_label={adapter_name}"
    ]
    
    # Print a summary of what will be run
    print("\nConfiguration summary:")
    print(f"  Model: {model_label}")
    print(f"  Training dataset: {dataset_label}")
    print(f"  Evaluation dataset: {evaluation_dataset}")
    print(f"  Adapter: {adapter_name}")
    print(f"  Generation: {generation}")
    print(f"  Seed: {seed}")
    print(f"  Quantisation: {quantisation}")
    print(f"  Training status: {training_status}")
    
    # Ask for confirmation
    confirm = input("\nRun this configuration? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return
    
    # Run the command
    print("\nRunning evaluation...")
    subprocess.run(command)

if __name__ == "__main__":
    main()