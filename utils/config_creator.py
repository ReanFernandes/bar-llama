import os
import itertools
import re

import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define the paths
TEMPLATE_DIR = "../config_creator/templates/"
ACTUAL_CONFIG_DIR = "../conf/"

# Define the combinable parameters and their possible values
combinable_params = {
    "training_status": ["trained", "untrained"],
    "quantisation_status": ["quantised_model", "full_model"],
    # "config_label": ["config1", "config2"],  # Example values; replace with your real ones
    "prompt_type": ["few_shot", "zero_shot"],
    "explanation_type": ["structured", "unstructured"],
    "response_type": ["fact_first", "answer_first"],
    "response_format": ["json", "markdown", "number_list"],
    "model_name": ["llama2"]
}

# Non-combinable parameters for dataset and dataloader templates
non_combinable_params = {
    "seed_dataset_label": "high_temp_structured_expl_dataset",
    "num_questions": "10",
    "randomise_questions": "True"
}

# Function to find missing placeholders
def find_unsubstituted_placeholders(template_str):
    # Regex to match placeholders like <param_name>
    return re.findall(r"<(.*?)>", template_str)

# Function to replace placeholders in the template and enforce placeholder checks
def substitute_template(template_str, param_values, template_name):
    for key, value in param_values.items():
        template_str = template_str.replace(f"<{key}>", str(value))

    # Find any remaining placeholders
    missing_placeholders = find_unsubstituted_placeholders(template_str)

    # If any placeholders are found, raise an error with specific details
    if missing_placeholders:
        raise ValueError(f"Unsubstituted placeholders detected in {template_name}: {', '.join(missing_placeholders)}")
    
    return template_str

# Function to create filenames based on combinable parameters
def generate_filename(params):
    return f"{params['response_format']}_{params['response_type']}_{params['prompt_type']}_{params['explanation_type']}.yaml"

# Process combinable templates and enforce checks before saving configs
def process_combinable_templates(template_dir, actual_config_dir, param_combinations):
    for param_combination in param_combinations:
        param_values = dict(zip(combinable_params.keys(), param_combination))
        for template_filename in os.listdir(template_dir):
            template_path = os.path.join(template_dir, template_filename)
            
            # Only process relevant YAML templates (train, eval, parsing, prompt)
            if template_filename in ["train.yaml", "eval.yaml", "parsing.yaml", "prompt.yaml"]:
                with open(template_path, 'r') as file:
                    template_content = file.read()

                # Substitute placeholders with enforced checking
                substituted_content = substitute_template(template_content, param_values, template_filename)

                # Generate the output filename and directory
                output_filename = generate_filename(param_values)
                subfolder = template_filename.split(".")[0]  # Use template name as subfolder (train, eval, etc.)
                output_dir = os.path.join(actual_config_dir, subfolder)

                # Ensure the directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Save the substituted content
                output_file = os.path.join(output_dir, output_filename)
                with open(output_file, 'w') as output:
                    output.write(substituted_content)

                print(f"Config saved to: {output_file}")

# Process non-combinable templates and enforce checks before saving configs
def process_non_combinable_templates(template_dir, actual_config_dir, non_combinable_params):
    for template_filename in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_filename)

        # Only process relevant YAML templates (dataset, dataloader)
        if template_filename in ["dataset.yaml", "dataloader.yaml"]:
            with open(template_path, 'r') as file:
                template_content = file.read()

            # Substitute placeholders with enforced checking
            substituted_content = substitute_template(template_content, non_combinable_params, template_filename)

            # Save in the appropriate subfolder (dataset or dataloader)
            subfolder = template_filename.split(".")[0]
            output_dir = os.path.join(actual_config_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)

            # Save the config file
            output_file = os.path.join(output_dir, f"{non_combinable_params['seed_dataset_label']}.yaml")
            with open(output_file, 'w') as output:
                output.write(substituted_content)

            print(f"Non-combinable config saved to: {output_file}")

# Generate all combinations for the combinable parameters
param_combinations = list(itertools.product(*combinable_params.values()))

# Process the combinable templates (train, eval, parsing, prompt)
process_combinable_templates(TEMPLATE_DIR, ACTUAL_CONFIG_DIR, param_combinations)

# Process the non-combinable templates (dataset, dataloader)
process_non_combinable_templates(TEMPLATE_DIR, ACTUAL_CONFIG_DIR, non_combinable_params)
