import json
import textwrap
import logging

class PromptHandler:
    '''
    copilot generated lolol
    PromptHandler is a class designed to handle the creation and management of prompts for a language model. It supports both zero-shot and few-shot learning paradigms and can format responses in various ways, including JSON and Markdown.
    Attributes:
        cfg (dict): Configuration dictionary containing various settings for prompt handling.
        system (str): The system prompt text.
        example (str): Example text for few-shot learning, if applicable.
        prompt_dict (list): List to store the generated prompts.
    Methods:
        __init__(config: dict):
            Initializes the PromptHandler with  the given configuration.
        _set_system_prompt():
            Retrieves and formats the system prompt based on the configuration settings.
        input_prompt(question_item):
            Processes a question item into an input prompt. Returns the input prompt and associated metadata.
        _set_response(data_item):
            Generates a response based on the configuration settings and the provided data item.
        _get_example():
            Retrieves and formats example questions and responses for few-shot learning.
        create_prompt(question_item):
            Creates a bundled system prompt and input prompt to be passed to the model. Returns the prompt and ground truth metadata.
    '''
    def __init__(self, config: dict):
        '''
        There are two ways this class could work, the first being to bundle both the system prompt and input prompt together, and pass the final
        prompt to the LLM, the second way is to pass the system prompt and the input prompt to the model class, which will bundle the two together in a message format
        that something like ollama can procces.
        
        '''
        self.cfg = config
        self.system = self._set_system_prompt()
        if self.cfg["prompt_type"] == "few_shot": # this means that we provide the model with an example to guide the model
            self.example = self._get_example()
            # append the example to the system prompt
            self.system += self.example
        logging.info(f"Prompt Handler initialized with configuration: {self.cfg}")
        self.prompt_dict = [] # list to store the prompts if we plan to do that (for training data putting together thing)
    def _set_system_prompt(self):
        '''
        Retrieve the system prompt, here in this case this prompt simply prepends the task to be done at hand. 
        whether a zero shot or a few shot task is specified, this part is the same.
        
        '''
        with open(self.cfg["system_prompt"], "r", encoding="utf-8") as file:
            return file.read()

    def input_prompt(self, question_item):
        '''
        Process the question item into the input prompt. Explicitly call this function if only the input prompt
        is required without the system prompt attached, which is in case the system prompt is being passed explicitly in the main function"
        '''

        input_prompt = textwrap.dedent(f"""
Question:
{question_item["question"]}
Answer Choices:
{question_item["options"]["optionA"]}
{question_item["options"]["optionB"]}
{question_item["options"]["optionC"]}
{question_item["options"]["optionD"]}
Response:
""")
        return input_prompt, {"question_number": question_item["question_number"],
                              "correct_answer": question_item["correct_answer"],
                              "explanation": question_item["explanation"],
                              "domain": question_item["domain"]} 
   
    def _set_response(self, data_item):
            " This function is so ugly. I cant with the indentation. but i guess since it works ill just let it be as is. Based on the response type requested, this function will provide the response to add to the example question"
            resp_format = self.cfg["response_format"]
            resp_type = self.cfg["response_type"]

            if resp_format == "json":
                if resp_type == "fact_first":
                    if self.cfg["explanation_type"] == "structured": # json_fact_first_structured
                        return f"""\n{{
        "domain": "{data_item["domain"]}",
        "explanation": {{
            "Legal_concept": "{data_item["explanation"]["Legal_Concept"]}",
            "Fact_analysis": "{data_item["explanation"]["Fact_Analysis"]}",
            "Rule_application": "{data_item["explanation"]["Rule_Application"]}",
            "Legal_conclusion": "{data_item["explanation"]["Legal_Conclusion"]}"
        }},
        "chosen_option_label": "{data_item["correct_answer"]}"
}}"""
                    elif self.cfg["explanation_type"] == "unstructured": # json_fact_first_unstructured
                        return f"""\n{{
        "domain": "{data_item["domain"]}",
        "explanation": "{data_item["explanation"]}",
        "chosen_option_label": "{data_item["correct_answer"]}"
}}"""
                if self.cfg["explanation_type"] == "structured": # json_answer_first_structured
                        return f"""\n{{
        "domain": "{data_item["domain"]}",
        "chosen_option_label": "{data_item["correct_answer"]}",
        "explanation": {{
            "Legal_concept": "{data_item["explanation"]["Legal_Concept"]}",
            "Fact_analysis": "{data_item["explanation"]["Fact_Analysis"]}",
            "Rule_application": "{data_item["explanation"]["Rule_Application"]}",
            "Legal_conclusion": "{data_item["explanation"]["Legal_Conclusion"]}"
        }}
}}"""           
                elif self.cfg["explanation_type"] == "unstructured": # json_answer_first_unstructured
                    return f"""\n{{
        "domain": "{data_item["domain"]}",
        "chosen_option_label": "{data_item["correct_answer"]}",
        "explanation": "{data_item["explanation"]}"
}}"""
            elif resp_format == "markdown":
                if resp_type == "fact_first":
                    if self.cfg["explanation_type"] == "structured": # markdown_fact_first_structured
                        return textwrap.dedent(
f"""\n## Chosen Domain
{data_item["domain"]}

## Explanation

### Legal concept
{data_item["explanation"]["Legal_Concept"]}

### Fact analysis
{data_item["explanation"]["Fact_Analysis"]}

### Rule application
{data_item["explanation"]["Rule_Application"]}

### Legal Conclusion
{data_item["explanation"]["Legal_Conclusion"]}

## Chosen Option
{data_item["correct_answer"]}
""")
                    elif self.cfg["explanation_type"] == "unstructured": # markdown_fact_first_unstructured
                        return textwrap.dedent(
f"""\n## Chosen Domain
{data_item["domain"]}

## Explanation
{data_item["explanation"]}

## Chosen Option
{data_item["correct_answer"]}
""")
                if self.cfg["explanation_type"] == "structured": # markdown_answer_first_structured
                    return textwrap.dedent(
f"""\n## Chosen Domain
{data_item["domain"]}

## Chosen Option
{data_item["correct_answer"]}

## Explanation

### Legal concept
{data_item["explanation"]["Legal_Concept"]}

### Fact analysis
{data_item["explanation"]["Fact_Analysis"]}

### Rule application
{data_item["explanation"]["Rule_Application"]}

### Legal Conclusion
{data_item["explanation"]["Legal_Conclusion"]}
""")
                elif self.cfg["explanation_type"] == "unstructured": # markdown_answer_first_unstructured
                    return textwrap.dedent(
f"""\n## Chosen Domain
{data_item["domain"]}

## Chosen Option
{data_item["correct_answer"]}

## Explanation
{data_item["explanation"]}
""")
            else: # response_format is number_list
                if  resp_type == "fact_first" :
                    if self.cfg["explanation_type"] == "structured": # number_list_fact_first_structured
                        return f"""\n
1. Chosen Domain: {data_item["domain"]}
2. Legal Concept: {data_item["explanation"]["Legal_Concept"]}
3. Fact Analysis: {data_item["explanation"]["Fact_Analysis"]}
4. Rule Application: {data_item["explanation"]["Rule_Application"]}
5. Legal Conclusion: {data_item["explanation"]["Legal_Conclusion"]}
6. Chosen Option Label: {data_item["correct_answer"]}
"""
                    elif self.cfg["explanation_type"] == "unstructured": # number_list_fact_first_unstructured
                        return f"""\n    
1. Chosen Domain: {data_item["domain"]}
2. Explanation: {data_item["explanation"]}
3. Chosen Option Label: {data_item["correct_answer"]}
"""
                if self.cfg["explanation_type"] == "structured": # number_list_answer_first_structured
                    return f"""\n
1. Chosen Domain: {data_item["domain"]}
2. Chosen Option Label: {data_item["correct_answer"]}
3. Legal Concept: {data_item["explanation"]["Legal_Concept"]}
4. Fact Analysis: {data_item["explanation"]["Fact_Analysis"]}
5. Rule Application: {data_item["explanation"]["Rule_Application"]}
6. Legal Conclusion: {data_item["explanation"]["Legal_Conclusion"]}
"""
                elif self.cfg["explanation_type"] == "unstructured": # number_list_answer_first_unstructured
                    return f"""\n
1. Chosen Domain: {data_item["domain"]}
2. Chosen Option Label: {data_item["correct_answer"]}
3. Explanation: {data_item["explanation"]}
"""

    def _get_example(self):
        ''' 
        Obtain the example file and format it for the few shot task, this is only called if the few shot task is specified in the config file.
        '''

        with open(self.cfg["example_path"], "r", encoding="utf-8") as file:
            examples = json.load(file)

        examples_text = "\n\n".join([
            textwrap.dedent(f"""Example {i+1}: 
Question:
{example["question"]}

Answer Choices:
{example["options"]["optionA"]}
{example["options"]["optionB"]}
{example["options"]["optionC"]}
{example["options"]["optionD"]}

Response:
{self._set_response(example)}
""")
        for i, example in enumerate(examples)
    ])

        return examples_text

    # def create_prompt(self, question_item):
    #     ''' 
    #         Create a bundled system prompt and input prompt, to be passed to the model as one message.
    #     '''
    #     input_prompt, ground_truth = self.input_prompt(question_item)
        
    #     if self.cfg["mode"] == "train": # use in the format of the model, with the answers appended after the input prompt
    #         # add an exception where if there is no correct answer, or explanation provided,  the question item will be skipped
            
    #         expected_response = self._set_response(question_item)
    #         if self.cfg["model_name"]=="llama2":
    #             prompt_template = f"""[INST]\n<<SYS>>\n{self.system}\n<</SYS>>{input_prompt}[/INST]{expected_response}</s>"""
    #         elif self.cfg["model_name"]=="mistral":
    #             print("model not implemented yet")
                
    #     elif self.cfg["mode"]=="eval":
    #         if self.cfg["pipeline_available"] is True:
    #             prompt_template = f"""{self.system}\n{input_prompt}"""
    #         elif self.cfg["pipeline_available"] is False:
    #             prompt_template = f"""[INST]\n<<SYS>>\n{self.system}\n<</SYS>>{input_prompt}[/INST]"""
    #         else:
    #             Exception("pipeline available not set correctly")
        
    #     if self.cfg["store_prompt"]:
    #         # append prompt_template as a value for the key "text", to self.prompt_list
    #         self.prompt_dict.append({"text":prompt_template})
        
    #     return prompt_template, ground_truth
    def create_prompt(self, question_item, mode="eval", model_name="llama2", pipeline_available=False, store_prompt=False):
        ''' 
        Create a bundled system prompt and input prompt, to be passed to the model as one message.
        '''
        input_prompt, ground_truth = self.input_prompt(question_item)
        
        if mode == "train":  # Use in the format of the model, with the answers appended after the input prompt
            expected_response = self._set_response(question_item)
            if model_name == "llama2":
                if self.cfg["include_system_prompt"]:
                    prompt_template = f"""[INST]\n<<SYS>>\n{self.system}\n<</SYS>>{input_prompt}[/INST]{expected_response}</s>"""
                else:
                    prompt_template = f"""[INST]\n{input_prompt}[/INST]{expected_response}</s>"""
            elif model_name == "mistral":
                if self.cfg["include_system_prompt"]:
                    raise NotImplementedError("Mistral model formatting is not implemented yet.")
                else:
                    raise NotImplementedError("Mistral model formatting is not implemented yet.")
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")
        
        elif mode == "eval":
            if pipeline_available:
                if self.cfg["include_system_prompt"]:
                    prompt_template = f"""{self.system}\n{input_prompt}"""
                else:
                    prompt_template = f"""{input_prompt}"""
            else:
                if self.cfg["include_system_prompt"]:
                    prompt_template = f"""[INST]\n<<SYS>>\n{self.system}\n<</SYS>>{input_prompt}[/INST]"""
                else:
                    prompt_template = f"""[INST]\n{input_prompt}[/INST]"""
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if store_prompt:
            # Append prompt_template as a value for the key "text", to self.prompt_dict
            self.prompt_dict.append({"text": prompt_template})

        return prompt_template, ground_truth