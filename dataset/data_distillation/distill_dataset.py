# import asyncio
# import openai
# import json
# import yaml
# import os
# from tqdm import tqdm

# # Set up the OpenAI client
# client = openai.AsyncOpenAI(
#     base_url="http://localhost:8080/v1",  # Server address
#     api_key="no-key-needed"  # Placeholder for server requiring no key
# )

# # Define the JSON schema for the task
# function_schema = {
#     "name": "generate_structured_response",
#     "description": "Generate a structured response for a legal question.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "Legal_Concept": {"type": "string", "description": "The main legal concept."},
#             "Fact_Analysis": {"type": "string", "description": "Analysis of the facts in relation to the question."},
#             "Rule_Application": {"type": "string", "description": "Application of legal rules to the facts."},
#             "Legal_Conclusion": {"type": "string", "description": "The final legal conclusion based on analysis and rules."},
#         },
#         "required": ["Legal_Concept", "Fact_Analysis", "Rule_Application", "Legal_Conclusion"]
#     },
# }

# # def create_prompt(entry, examples, few_shot=True):
# #     """
# #     Creates a few-shot or zero-shot prompt with the provided examples and entry.
# #     """
# #     if few_shot and examples:
# #         examples_text = "\n\n".join([
# #             f"""Example {i+1}:
# #             Question:
# #             {example["question"]}

# #             Answer Choices:
# #             {json.dumps(example["options"], indent=2)}

# #             Correct Answer:
# #             {example["correct_answer"]}

# #             Original Explanation:
# #             {example["explanation"]}

# #             Restructured Explanation:
# #             {json.dumps(example["restructured_explanation"], indent=2)}
# #             """
# #             for i, example in enumerate(examples)
# #         ])
# #     else:
# #         examples_text = ""

# #     return f"""
# #     You are an advanced legal text assistant tasked with restructuring explanations for legal questions into a predefined JSON format. The purpose of this restructuring is to ensure clarity, consistency, and machine-readability for downstream applications, such as automated legal reasoning and analysis.

# #     The JSON structure you must produce strictly adheres to this format:
# #     {{
# #         "Legal_Concept": "A concise description of the main legal concept addressed in the explanation.",
# #         "Fact_Analysis": "A clear analysis of the factual elements relevant to the question and answer choices.",
# #         "Rule_Application": "A detailed application of the relevant legal rules to the provided facts.",
# #         "Legal_Conclusion": "A definitive conclusion, based on the analysis and application of legal rules."
# #     }}

# #     ### Examples for Guidance:
# #     {examples_text}

# #     ### Task Instructions:
# #     1. Carefully analyze the provided question, answer choices, correct answer, and original explanation.
# #     2. Extract and reorganize the information into the above JSON structure.
# #     3. Use the examples as templates for formatting and tone, but ensure your output strictly reflects the input provided.
# #     4. Avoid introducing any new information or deviating from the original explanation's intent.

# #     ### Input Data:
# #     Question:
# #     {entry['question']}

# #     Answer Choices:
# #     {json.dumps(entry['options'], indent=2)}

# #     Correct Answer:
# #     {entry['correct_answer']}

# #     Original Explanation:
# #     {entry['explanation']}

# #     ### Expected Output:
# #     Output ONLY the JSON object conforming to the structure above. Do NOT include any additional text, commentary, or formatting outside the JSON.
# #     """

# def create_prompt(entry, examples, few_shot=True):
#     """
#     Creates a few-shot or zero-shot prompt with the provided examples and entry,
#     adjusted for nested options.
#     """
#     if few_shot and examples:
#         examples_text = "\n\n".join([
#             f"""Example {i+1}:
#             Question:
#             {example["question"]}

#             Answer Choices:
#             {json.dumps(example["options"], indent=2)}

#             Correct Answer:
#             {example["correct_answer"]}

#             Original Explanation:
#             {example["explanation"]}

#             Restructured Explanation:
#             {json.dumps(example["restructured_explanation"], indent=2)}
#             """
#             for i, example in enumerate(examples)
#         ])
#     else:
#         examples_text = ""

#     # Flatten options into a readable string for the prompt
#     options_text = "\n".join([f"{key}: {value}" for key, value in entry['options'].items()])

#     return f"""
#     You are an advanced legal text assistant tasked with restructuring explanations for legal questions into a predefined JSON format. The purpose of this restructuring is to ensure clarity, consistency, and machine-readability for downstream applications, such as automated legal reasoning and analysis.

#     The JSON structure you must produce strictly adheres to this format:
#     {{
#         "Legal_Concept": "A concise description of the main legal concept addressed in the explanation.",
#         "Fact_Analysis": "A clear analysis of the factual elements relevant to the question and answer choices.",
#         "Rule_Application": "A detailed application of the relevant legal rules to the provided facts.",
#         "Legal_Conclusion": "A definitive conclusion, based on the analysis and application of legal rules."
#     }}

#     ### Examples for Guidance:
#     {examples_text}

#     ### Task Instructions:
#     1. Carefully analyze the provided question, answer choices, correct answer, and original explanation.
#     2. Extract and reorganize the information into the above JSON structure.
#     3. Use the examples as templates for formatting and tone, but ensure your output strictly reflects the input provided.
#     4. Avoid introducing any new information or deviating from the original explanation's intent.

#     ### Input Data:
#     Question:
#     {entry['question']}

#     Answer Choices:
#     {options_text}

#     Correct Answer:
#     {entry['correct_answer']}

#     Original Explanation:
#     {entry['explanation']}

#     ### Expected Output:
#     Output ONLY the JSON object conforming to the structure above. Do NOT include any additional text, commentary, or formatting outside the JSON.
#     """

# # async def generate_response(entry, examples, params, few_shot=True):
# #     """
# #     Asynchronously generates a structured response for a given entry using the function_call API.
# #     """
# #     try:
# #         # Create the prompt
# #         prompt = create_prompt(entry, examples, few_shot=few_shot)

# #         # Generate the structured response
# #         response = await client.chat.completions.create(
# #             messages=[
# #                 {"role": "user", "content": prompt}
# #             ],
# #             model=params["model"],
# #             temperature=params.get("temperature", 0.2),
# #             max_tokens=params.get("max_tokens", 1024),
# #             top_p=params.get("top_p", 1.0),
# #             function_call={"name": function_schema["name"]},  # Enforce the schema
# #             functions=[function_schema],  # Provide the schema
# #         )

# #         # Extract the function-generated result
# #         result = response.choices[0].message.function_call.arguments
# #         return json.loads(result)

# #     except Exception as e:
# #         print(f"Error processing entry {entry.get('question', 'unknown')}: {e}")
# #         return None
# async def generate_response(entry, examples, params, few_shot=True):
#     """
#     Asynchronously generates a structured response for a given entry.
#     Handles cases where the model outputs JSON in `message.content` instead of `function_call.arguments`.
#     """
#     try:
#         # Create the prompt
#         prompt = create_prompt(entry, examples, few_shot=few_shot)

#         # Generate the structured response
#         response = await client.chat.completions.create(
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             model=params["model"],
#             temperature=params.get("temperature", 0.2),
#             max_tokens=params.get("max_tokens", 1024),
#             top_p=params.get("top_p", 1.0),
#             function_call={"name": function_schema["name"]},  # Enforce schema (optional for fallback)
#             functions=[function_schema],  # Provide schema
#         )

#         # Attempt to extract the JSON from function_call.arguments
#         if response.choices[0].message.function_call and response.choices[0].message.function_call.arguments:
#             result = response.choices[0].message.function_call.arguments
#             return json.loads(result)

#         # Fallback: Attempt to extract JSON from message.content
#         elif response.choices[0].message.content:
#             result = response.choices[0].message.content
#             return json.loads(result)

#         # If no valid JSON is found, raise an error
#         else:
#             raise ValueError("Model response did not include valid JSON.")

#     except json.JSONDecodeError as e:
#         print(f"JSON decoding error for entry '{entry['question']}': {e}")
#         return None
#     except Exception as e:
#         print(f"Error processing entry '{entry['question']}': {e}")
#         return None

# # async def process_dataset(dataset, examples, params, output_file, few_shot=True):
# #     """
# #     Processes the entire dataset asynchronously, saving results incrementally.
# #     """
# #     # Load existing results if the file exists
# #     if os.path.exists(output_file):
# #         with open(output_file, "r") as f:
# #             processed_data = json.load(f)
# #     else:
# #         processed_data = []

# #     processed_questions = {entry["question"] for entry in processed_data}
# #     unprocessed_entries = [entry for entry in dataset if entry["question"] not in processed_questions]

# #     batch_size = params.get("batch_size", 50)
# #     for i in tqdm(range(0, len(unprocessed_entries), batch_size), desc="Processing batches"):
# #         batch = unprocessed_entries[i : i + batch_size]
# #         tasks = [generate_response(entry, examples, params, few_shot) for entry in batch]
# #         results = await asyncio.gather(*tasks)

# #         # Filter out failed entries
# #         results = [res for res in results if res]

# #         # Save results incrementally
# #         processed_data.extend(results)
# #         with open(output_file, "w") as f:
# #             json.dump(processed_data, f, indent=4)

# async def process_dataset(dataset, examples, params, output_file, few_shot=True):
#     """
#     Processes the entire dataset asynchronously, replacing the original explanation
#     with the restructured explanation, and saving results incrementally.
#     """
#     # Load existing results if the file exists
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             processed_data = json.load(f)
#     else:
#         processed_data = []

#     # Get a set of already processed questions to avoid re-processing
#     processed_questions = {entry["question"] for entry in processed_data}
#     unprocessed_entries = [entry for entry in dataset if entry["question"] not in processed_questions]

#     batch_size = params.get("batch_size", 50)
#     for i in tqdm(range(0, len(unprocessed_entries), batch_size), desc="Processing batches"):
#         batch = unprocessed_entries[i : i + batch_size]
#         tasks = [generate_response(entry, examples, params, few_shot) for entry in batch]
#         results = await asyncio.gather(*tasks)

#         # Replace the explanation for each entry in the batch
#         for entry, result in zip(batch, results):
#             if result:
#                 entry["explanation"] = result  # Replace the explanation field with the restructured version
#                 processed_data.append(entry)

#         # Save the updated dataset incrementally
#         with open(output_file, "w") as f:
#             json.dump(processed_data, f, indent=4)


# def load_examples(examples_file):
#     """
#     Loads examples for few-shot prompting.
#     """
#     try:
#         with open(examples_file, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading examples: {e}")
#         return []

# def load_dataset(dataset_file):
#     """
#     Loads the dataset to process.
#     """
#     try:
#         with open(dataset_file, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return []

# def main(config_file):
#     """
#     Main function to run the dataset processing.
#     """
#     # Load configuration
#     with open(config_file, "r") as f:
#         config = yaml.safe_load(f)

#     # Load examples and dataset
#     examples = load_examples(config["experiments"]["few_shot_high_temp"]["examples_file"])
#     dataset = load_dataset(config["dataset_location"])

#     # Experiment parameters
#     params = {
#         "model": config["server_config"]["model"],
#         "temperature": config["experiments"]["few_shot_high_temp"].get("temperature", 0.2),
#         "max_tokens": config["experiments"]["few_shot_high_temp"].get("max_tokens", 1024),
#         "top_p": config["experiments"]["few_shot_high_temp"].get("top_p", 1.0),
#         "batch_size": config["experiments"]["few_shot_high_temp"].get("batch_size", 50),
#     }

#     # Process the dataset asynchronously
#     output_file = os.path.join(config["result_directory"], "distilled_dataset.json")
#     asyncio.run(process_dataset(dataset, examples, params, output_file))

# if __name__ == "__main__":
#     main("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/data_distillation/data_distill_config.yaml")
import asyncio
import openai
import json
import yaml
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename="data_distillation.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Set up the OpenAI client
client = openai.AsyncOpenAI(
    base_url="http://localhost:8080/v1",  # Server address
    api_key="no-key-needed"  # Placeholder for server requiring no key
)

# Define the JSON schema for the task
function_schema = {
    "name": "generate_structured_response",
    "description": "Generate a structured response for a legal question.",
    "parameters": {
        "type": "object",
        "properties": {
            "Legal_Concept": {"type": "string", "description": "The main legal concept."},
            "Fact_Analysis": {"type": "string", "description": "Analysis of the facts in relation to the question."},
            "Rule_Application": {"type": "string", "description": "Application of legal rules to the facts."},
            "Legal_Conclusion": {"type": "string", "description": "The final legal conclusion based on analysis and rules."},
        },
        "required": ["Legal_Concept", "Fact_Analysis", "Rule_Application", "Legal_Conclusion"]
    },
}

def create_prompt(entry, examples, few_shot=True):
    """
    Creates a few-shot or zero-shot prompt with the provided examples and entry.
    """
    if few_shot and examples:
        examples_text = "\n\n".join([
            f"""Example {i+1}:
            Question:
            {example["question"]}

            Answer Choices:
            {json.dumps(example["options"], indent=2)}

            Correct Answer:
            {example["correct_answer"]}

            Original Explanation:
            {example["explanation"]}

            Restructured Explanation:
            {json.dumps(example["restructured_explanation"], indent=2)}
            """
            for i, example in enumerate(examples)
        ])
    else:
        examples_text = ""

    # Flatten options into a readable string for the prompt
    options_text = "\n".join([f"{key}: {value}" for key, value in entry['options'].items()])

    return f"""
    You are an advanced legal text assistant tasked with restructuring explanations for legal questions into a predefined JSON format. The purpose of this restructuring is to ensure clarity, consistency, and machine-readability for downstream applications, such as automated legal reasoning and analysis.

    The JSON structure you must produce strictly adheres to this format:
    {{
        "Legal_Concept": "A concise description of the main legal concept addressed in the explanation.",
        "Fact_Analysis": "A clear analysis of the factual elements relevant to the question and answer choices.",
        "Rule_Application": "A detailed application of the relevant legal rules to the provided facts.",
        "Legal_Conclusion": "A definitive conclusion, based on the analysis and application of legal rules."
    }}

    ### Examples for Guidance:
    {examples_text}

    ### Task Instructions:
    1. Carefully analyze the provided question, answer choices, correct answer, and original explanation.
    2. Extract and reorganize the information into the above JSON structure.
    3. Use the examples as templates for formatting and tone, but ensure your output strictly reflects the input provided.
    4. Avoid introducing any new information or deviating from the original explanation's intent.

    ### Input Data:
    Question:
    {entry['question']}

    Answer Choices:
    {options_text}

    Correct Answer:
    {entry['correct_answer']}

    Original Explanation:
    {entry['explanation']}

    ### Expected Output:
    Output ONLY the JSON object conforming to the structure above. Do NOT include any additional text, commentary, or formatting outside the JSON.
    """

async def generate_response(entry, examples, params, few_shot=True):
    """
    Asynchronously generates a structured response for a given entry.
    """
    try:
        # Create the prompt
        prompt = create_prompt(entry, examples, few_shot=few_shot)

        # Generate the structured response
        response = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=params["model"],
            temperature=params.get("temperature", 0.2),
            max_tokens=params.get("max_tokens", 1024),
            top_p=params.get("top_p", 1.0),
            function_call={"name": function_schema["name"]},  # Enforce schema
            functions=[function_schema],  # Provide schema
        )

        # Attempt to extract the JSON from function_call.arguments
        if response.choices[0].message.function_call and response.choices[0].message.function_call.arguments:
            result = response.choices[0].message.function_call.arguments
            return json.loads(result)

        # Fallback: Attempt to extract JSON from message.content
        elif response.choices[0].message.content:
            result = response.choices[0].message.content
            return json.loads(result)

        # If no valid JSON is found, raise an error
        else:
            raise ValueError("Model response did not include valid JSON.")

    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error for entry '{entry['question']}': {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing entry '{entry['question']}': {e}")
        return None

async def process_dataset(dataset, examples, params, output_file, few_shot=True):
    """
    Processes the entire dataset asynchronously, replacing the original explanation
    with the restructured explanation, and saving results incrementally.
    """
    # Load existing results if the file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            processed_data = json.load(f)
            logging.info(f"Loaded {len(processed_data)} processed entries from {output_file}.")
    else:
        processed_data = []

    # Get a set of already processed questions to avoid re-processing
    processed_questions = {entry["question"] for entry in processed_data}
    unprocessed_entries = [entry for entry in dataset if entry["question"] not in processed_questions]
    logging.info(f"Starting processing of {len(unprocessed_entries)} unprocessed entries.")

    batch_size = params.get("batch_size", 50)
    for i in tqdm(range(0, len(unprocessed_entries), batch_size), desc="Processing batches"):
        batch = unprocessed_entries[i : i + batch_size]
        tasks = [generate_response(entry, examples, params, few_shot) for entry in batch]
        results = await asyncio.gather(*tasks)

        # Replace the explanation for each entry in the batch
        for entry, result in zip(batch, results):
            if result:
                entry["explanation"] = result  # Replace the explanation field with the restructured version
                processed_data.append(entry)
            else:
                logging.warning(f"Failed to process entry: {entry['question']}")

        # Save the updated dataset incrementally
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=4)
        logging.info(f"Processed {len(batch)} entries. Saved progress to {output_file}.")

    logging.info(f"Processing complete. Total entries processed: {len(processed_data)}.")

def load_examples(examples_file):
    """
    Loads examples for few-shot prompting.
    """
    try:
        with open(examples_file, "r") as f:
            examples = json.load(f)
            logging.info(f"Loaded {len(examples)} examples from {examples_file}.")
            return examples
    except Exception as e:
        logging.error(f"Error loading examples: {e}")
        return []

def load_dataset(dataset_file):
    """
    Loads the dataset to process.
    """
    try:
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
            logging.info(f"Loaded dataset with {len(dataset)} entries from {dataset_file}.")
            return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return []

def main(config_file):
    """
    Main function to run the dataset processing.
    """
    logging.info("Starting dataset processing.")
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load examples and dataset
    examples = load_examples(config["experiments"]["few_shot_high_temp"]["examples_file"])
    dataset = load_dataset(config["dataset_location"])

    # Experiment parameters
    params = {
        "model": config["server_config"]["model"],
        "temperature": config["experiments"]["few_shot_high_temp"].get("temperature", 0.2),
        "max_tokens": config["experiments"]["few_shot_high_temp"].get("max_tokens", 1024),
        "top_p": config["experiments"]["few_shot_high_temp"].get("top_p", 1.0),
        "batch_size": config["experiments"]["few_shot_high_temp"].get("batch_size", 50),
    }

    # Process the dataset asynchronously
    output_file = os.path.join(config["result_directory"], "distilled_dataset_extra.json")
    asyncio.run(process_dataset(dataset, examples, params, output_file))

if __name__ == "__main__":
    main("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/data_distillation/data_distill_config.yaml")
