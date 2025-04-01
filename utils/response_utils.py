import re
import logging
import json
import os
import pandas as pd
import itertools

class ResponseHandler():
    def __init__(self,config):
        """
        This class is responsible for all things related to handling the response of 
        the llms response. It will recieve both response of the llm and the objective
        ground truth and will process them for comparison. 
        """
    
        self.cfg = config
        # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.count = 0 # counter which is incremented for every response that is assessed, used to log the current question number if any processing errors for that question
        self.response_dump = []
        self.stats = {
            "total_processed":0,
            "incompletely_proccessed":{
                "Criminal_law":{"domain":0,"chosen_option_label":0,"Legal_concept":0,"Fact_analysis":0,"Rule_application":0,"Legal_conclusion":0},
                "Evidence":{"domain":0,"chosen_option_label":0,"Legal_concept":0,"Fact_analysis":0,"Rule_application":0,"Legal_conclusion":0},
                "Contracts":{"domain":0,"chosen_option_label":0,"Legal_concept":0,"Fact_analysis":0,"Rule_application":0,"Legal_conclusion":0},
                "Constitutional_law":{"domain":0,"chosen_option_label":0,"Legal_concept":0,"Fact_analysis":0,"Rule_application":0,"Legal_conclusion":0},
                "Property":{"domain":0,"chosen_option_label":0,"Legal_concept":0,"Fact_analysis":0,"Rule_application":0,"Legal_conclusion":0}
            },
            "broken_json_output":0
        }


    def parse(self, response):
        """
        Takes the model output, and based on the prompt type and format,
        will extract the relevant fields, and return a dict of the fields.
        self.cfg["response_format"] =["json","markdown","number_list"]
        """
        if self.cfg["explanation_type"] == "structured":
            explanation_fields = {
                "Legal_concept": "n/a",
                "Fact_analysis": "n/a",
                "Rule_application": "n/a",
                "Legal_conclusion": "n/a"
            }
        elif self.cfg["explanation_type"] == "unstructured":
            explanation_fields = "n/a"

        fields = {
            "domain": "n/a",
            "chosen_option_label": "n/a",
            "explanation": explanation_fields
        }        
        # most of the times if the answer is completed then the llm creates its own question starting with "Question:" i want to remove this in the starting itself
        # as the text after this term is irrelevant. THIS IS THE COMMON PATTERN FOR NON-FINETUNED MODELS, FINETUNING TAKES CARE OF THIS ISSUE
        # cleaned_response = re.split(r"\n\nQuestion:\n", response, maxsplit=1)[0]
        system_token_pattern = r'\n*(?:<</(?:SYS|INST)>>|\[/(?:SYS|INST)\]|</s>)\n*Question:'
        cleaned_response = re.split(system_token_pattern, response, maxsplit=1)[0]
        if self.cfg["response_format"] == "markdown":
            fields = self._parse_markdown(cleaned_response, fields)
        elif self.cfg["response_format"] == "json":
            fields = self._parse_json(cleaned_response, fields)  
        elif self.cfg["response_format"] == "number_list":
            fields = self._parse_number_list(cleaned_response, fields)
        return fields
    
    def _parse_markdown(self, response, fields):


        #extract domain first
        try:
                domain_match = re.search(r"## Chosen Domain\n(.+?)\n", response, re.DOTALL)
                if domain_match:
                    fields['domain'] = domain_match.group(1).strip()
                else: 
                    raise Exception
        except Exception as e:
            logging.error("Failed to extract domain: %s for response: %s", str(e), self.count)

        #extract explanation        
        if self.cfg["explanation_type"] == "structured":
            explanation_keys = ["Legal_concept", "Fact_analysis", "Rule_application", "Legal_conclusion"]
            patterns = [
                r"### Legal concept\n(.+?)\n\n",
                r"### Fact analysis\n(.+?)\n\n",
                r"### Rule application\n(.+?)\n\n",
                r"### Legal Conclusion\n(.+?)\n\n"
            ]
            # 
            for key, pattern in zip(explanation_keys, patterns):
                try:
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        fields['explanation'][key] = match.group(1).strip() 
                    else:
                        raise Exception
                except Exception as e:
                    logging.error(f"|Q. {self.current_q_num}||D: {self.domain}| Failed to extract {key}: {str(e)}")
        elif self.cfg["explanation_type"] == "unstructured":
            try:
                explanation_match = re.search(r"## Explanation\n(.+?)\n", response, re.DOTALL)
                if explanation_match:
                    fields['explanation'] = explanation_match.group(1).strip()
                else: 
                    raise Exception
            except Exception as e:
                logging.error(f"|Q. {self.current_q_num}||D: {self.domain}| Failed to extract explanation: {str(e)}")
        # Extract chosen option
        try:
            chosen_option_match = re.search(r"## Chosen Option\n([^\n]+)", response)
            if chosen_option_match:
                fields['chosen_option_label'] = chosen_option_match.group(1).strip()
            else: 
                    raise Exception
        except Exception as e:
            logging.error(f"|Q. {self.current_q_num}||D: {self.domain}| Failed to extract chosen option: {str(e)}")

        return  fields
    def _parse_json(self, response, fields):
        """And in my hubris,
            while my tower fell,
            I thought adding more blocks to the top,
            would save me from this hell

            and now i sit on this pile of rubble
            sifting through the ash
            trying to piece things together
            every run begins only to crash
            """
        def clean_response(response: str) -> str:
                # First get the JSON object up until any instruction tokens
            instruction_patterns = [
                r'<</INST>.*$',
                r'</s>.*$',
                r'Question:.*$'
            ]
            
            for pattern in instruction_patterns:
                response = re.sub(pattern, '', response, flags=re.DOTALL)

            # Clean up any remaining newlines/whitespace
            response = response.strip()
            
            # Fix JSON structure if needed
            if not response.startswith('{'):
                response = '{' + response
            if not response.endswith('}'):
                response = response + '}'
            
            # Fix escaped quotes
            response = response.replace('\"', '"')
            
            return response
        def fix_invalid_json(invalid_json, explanation_type):
                  
            invalid_json = invalid_json.strip()
            # Fix incomplete string values within lines. This happens when the model is truncated while still in the midst of filling out some field
            lines = invalid_json.split('\n')
            # First escape inner quotes
            fixed_lines2 = []
            for line in lines:
                if ':' in line:
                    quotes_indices = [i for i, char in enumerate(line) if char == '"']
                    if len(quotes_indices) > 2:
                        value_start = quotes_indices[2]
                        value_end = quotes_indices[-1]
                        inner_quotes = [i for i in quotes_indices if value_start < i < value_end]
                        for idx in reversed(inner_quotes):
                            line = line[:idx] + "\\" + line[idx:]
                fixed_lines2.append(line)

            # Then handle incomplete lines
            fixed_lines = []
            for line in fixed_lines2:
                quote_count = line.count('"')
                if quote_count % 2 != 0:
                    if re.search(r':\s*".*$', line):
                        line += '[INCOMPLETE]",'
                fixed_lines.append(line)

            invalid_json = '\n'.join(fixed_lines)
        
            nested_objects = re.findall(r'\{[^{}]*\}', invalid_json)
            for obj in nested_objects:
                obj_fixed = re.sub(r',\s*(?=[}\]])', '', obj)
                invalid_json = invalid_json.replace(obj, obj_fixed)

            # Remove trailing commas at the top level 
            invalid_json = re.sub(r',\s*(?=[}\]])', '', invalid_json)
            # Balance inner braces for nested objects,  agin truncation issue
            nested_objects = re.findall(r'\{[^{}]*\}', invalid_json)
            for obj in nested_objects:
                open_braces = obj.count('{')
                close_braces = obj.count('}')
                if open_braces > close_braces:
                    fixed_obj = obj + '}' * (open_braces - close_braces)
                    invalid_json = invalid_json.replace(obj, fixed_obj)
                elif close_braces > open_braces:
                    fixed_obj = '{' * (close_braces - open_braces) + obj
                    invalid_json = invalid_json.replace(obj, fixed_obj)

            # Ensure the JSON starts and ends with correct braces, since the inner level nested objects are fixed till this point, i know this wont always work but atleast its robust
            if not invalid_json.startswith('{'):
                invalid_json = '{' + invalid_json
            if not invalid_json.endswith('}'):
                invalid_json = invalid_json + '}'
            # Find first complete JSON structure
            # Look for a complete JSON object, for instances where  the model generates extra garbage and puts it inside braces
            json_match = re.search(r'({[^}]*})[^{]*$', invalid_json, re.DOTALL)
            if json_match:
                invalid_json = json_match.group(1)
            # Balance the overall number of braces for the entire JSON structure
            open_braces = invalid_json.count('{')
            close_braces = invalid_json.count('}')
            if open_braces > close_braces:
                invalid_json += '}' * (open_braces - close_braces)
            elif close_braces > open_braces:
                invalid_json = '{' * (close_braces - open_braces) + invalid_json

            # again remove any missed trailing commas for redundancy
            invalid_json = re.sub(r',\s*(?=[}\]])', '', invalid_json)

            # Define required fields based on explanation_type
            if explanation_type == "structured":
                required_fields = {
                    "domain": "n/a",
                    "chosen_option_label": "n/a",
                    "explanation": {
                        "Legal_concept": "n/a",
                        "Fact_analysis": "n/a",
                        "Rule_application": "n/a",
                        "Legal_conclusion": "n/a"
                    }
                }
            else:
                required_fields = {
                    "domain": "n/a",
                    "chosen_option_label": "n/a",
                    "explanation": "n/a"
                }

            try:
                json_data = json.loads(invalid_json)
                logging.warn(f"|Q. {self.current_q_num}||D: {self.domain}| Successfully fixed invalid JSON. Not all fields may be present.")
            except json.JSONDecodeError:
                logging.warn(f"|Q. {self.current_q_num}||D: {self.domain}| Failed to fix invalid JSON, extracting as many fields as possible.")
                try:
                    json_data = {}
                    

                    for key, value in required_fields.items():
                        if key == 'explanation':
                            json_data['explanation'] = {}
                            exp_match = re.search(r'"explanation"\s*:\s*({[^}]+})', invalid_json, re.DOTALL)
                            if exp_match:
                                exp_text = exp_match.group(1)
                                for subkey, subval in value.items():
                                    submatch = re.search(rf'"{subkey}"\s*:\s*"(.*?)(?:",|\n|}})', exp_text, re.DOTALL)
                                    json_data['explanation'][subkey] = submatch.group(1) if submatch else subval
                            continue
                        
                        
                        match = re.search(rf'"{key}"\s*:\s*"(.*?)(?:",|\n)', invalid_json, re.DOTALL)
                        json_data[key] = match.group(1) if match else value
                    
                except Exception as e:
                    logging.error(f"|Q. {self.current_q_num}||D: {self.domain}| failed to extract valid fields from fixed JSON: {str(e)}")
                    # assign how much ever was able to be extracted from the invalid data to the required fields, and load it back to the json_data
                    for key, value in required_fields.items():
                        if key not in json_data:
                            json_data[key] = value
            return json.dumps(json_data, indent=4)

        response = clean_response(response=response)
        try:
            # this block is a last moment implementation that deals with a new issue of the models response escaping double quotes everywhere,
            # something that shouldnt happen on account of the fine-tuning prompt not containing such behaviour. 
            # i am implementing this in interest of still processing cases where there are responses to the question, and giving more
            # priority to the answering instead of the formatting. 
            system_token_pattern = r'\n*(?:<</(?:SYS|INST)>>|\[/(?:SYS|INST)\]|</s>)\n*Question:'
            response = re.split(system_token_pattern, response, maxsplit=1)[0]
            response = response.replace('\"', '"')

            json_data = json.loads(response)
        except json.JSONDecodeError:
            logging.warn(f"|Q. {self.current_q_num}||D: {self.domain}| has broken JSON output. Attempting to fix it.")
            response = fix_invalid_json(response, self.cfg["explanation_type"])
            try:
                json_data = json.loads(response)
                logging.info(f"|Q. {self.current_q_num}||D: {self.domain}| was made valid into Valid json. Extracting responses.")
            except json.JSONDecodeError:
                logging.info(f"|Q. {self.current_q_num}||D: {self.domain}| Unable to convert response to JSON ")
        if not isinstance(json_data, dict):
            logging.error(f"|Q. {self.current_q_num}||D: {self.domain}| JSON data is not a dictionary.")
            return fields
        fields['domain'] = json_data.get('domain', 'n/a')
        fields['chosen_option_label'] = json_data.get('chosen_option_label', 'n/a')

        if self.cfg["explanation_type"] == "structured":
            explanation_data = json_data.get('explanation', {})
            if isinstance(explanation_data, dict):
                fields['explanation'] = {
                    "Legal_concept": explanation_data.get('Legal_concept', 'n/a'),
                    "Fact_analysis": explanation_data.get('Fact_analysis', 'n/a'),
                    "Rule_application": explanation_data.get('Rule_application', 'n/a'),
                    "Legal_conclusion": explanation_data.get('Legal_conclusion', 'n/a')
                }
            else:
                logging.warning("Expected a dictionary for structured explanation but got different format.")
        elif self.cfg["explanation_type"] == "unstructured":
            fields['explanation'] = json_data.get('explanation', 'n/a')
            if not isinstance(fields['explanation'], str):
                logging.warning("Expected a string for unstructured explanation but got different format.")

        return fields


    def _parse_number_list(self, response, fields):
        """
        Parses unstructured or semi-structured output that loosely follows a given schema.
        Looks for entries numbered and titled as specified in the schema, and ignores anything after a specified marker.
        """
        # Schema pattern , hardcoding this  to stop headaches 
        if self.cfg["response_type"] == "answer_first":
            if self.cfg["explanation_type"] == "structured":
                schema_keys = {
                    "1. Chosen Domain": "domain",
                    "2. Chosen Option Label": "chosen_option_label",
                    "3. Legal Concept": "explanation.Legal_concept",
                    "4. Fact Analysis": "explanation.Fact_analysis",
                    "5. Rule Application": "explanation.Rule_application",
                    "6. Legal Conclusion": "explanation.Legal_conclusion"
                }
            elif self.cfg["explanation_type"] == "unstructured":
                schema_keys = {
                    "1. Chosen Domain": "domain",
                    "2. Chosen Option Label": "chosen_option_label",
                    "3. Explanation": "explanation"
                }    
        elif self.cfg["response_type"]=="fact_first":
            if self.cfg["explanation_type"] == "structured":
                schema_keys = {
                    "1. Chosen Domain": "domain",
                    "2. Legal Concept": "explanation.Legal_concept",
                    "3. Fact Analysis": "explanation.Fact_analysis",
                    "4. Rule Application": "explanation.Rule_application",
                    "5. Legal Conclusion": "explanation.Legal_conclusion",
                    "6. Chosen Option Label": "chosen_option_label"
                }
            elif self.cfg["explanation_type"] == "unstructured":
                schema_keys = {
                    "1. Chosen Domain": "domain",
                    "2. Explanation": "explanation",
                    "3. Chosen Option Label": "chosen_option_label"
                }

        # Extract data for each schema key
        for key, field_path in schema_keys.items():
            # The pattern matches the key followed by any text until the next digit-period or end of section
            pattern = rf"{re.escape(key)}: (.*?)(?=\n\d\.|\n\n|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                # Determine if we need to place the value in a nested dictionary
                if '.' in field_path:
                    main_key, sub_key = field_path.split('.')
                    fields[main_key][sub_key] = match.group(1).strip()
                else:
                    fields[field_path] = match.group(1).strip()
            else:
                logging.warning(f"No match found for {key} in question number {self.current_q_num} in domain {self.domain}")
                # self.stats["incompletely_proccessed"][self.domain][sub_key]+=1
        return fields

    def assess(self, response):
        """
        response is of a dictionary of the following type :
        {
            "prompt": "",
            "response": "",
            "ground_truth": {
                "question_number": ,
                "correct_answer": "",
                "explanation": { "can be structured or unstructured
                }
        }
        this function calls the parse function to parse the response of the model. 
        """
        
        self.count+=1 #increment the count
        self.stats["total_processed"]+=1 #for loggign
        self.domain = response["ground_truth"]["domain"]
        self.current_q_num = response["ground_truth"]["question_number"]
        response_dict = self.parse(response["response"])
        response["response"] = response_dict
        self.response_dump.append(response)
        
        
    def dump_to(self, file_path):
        """
        Dumps the response_dump to a file
        """
        with open(file_path, "w") as f:
            json.dump(self.response_dump, f, indent=4)
        logging.info(f"Dumped the response to {file_path}")
    
    def dump_stats(self, file_path):
        """
        Dumps the stats to a file
        """
        with open(file_path, "w") as f:
            json.dump(self.stats, f, indent=4)
        logging.info(f"Dumped the stats to {file_path}")




class ResponseGrader:
    def __init__(self, comparison_dict):
        # Malformed counts for logging
        self.malformed_count = {'malformed_label': 0, 'malformed_domain': 0}

        # General prediction counts
        self.correct_predictions = 0
        self.total_predictions = 0
        self.correct_label_and_domain = 0

        # Domain list and counters
        self.domains = [
            "Constitutional Law", "Contracts", "Criminal Law",
            "Evidence", "Real Property", "Torts", "Civil Procedure"
        ]
        
        # Initialize counters for domain-specific accuracies
        self.domain_label_correct_counts = {domain: 0 for domain in self.domains}
        self.domain_label_total_counts = {domain: 0 for domain in self.domains}
        self.domain_combined_correct_counts = {domain: 0 for domain in self.domains}
        self.domain_combined_total_counts = {domain: 0 for domain in self.domains}

        # Track error patterns
        self.error_types = {
            'correct_domain_wrong_label': 0,
            'correct_label_wrong_domain': 0,
            'both_wrong': 0
        }

        # Confidence tracking for domains
        self.domain_prediction_counts = {domain: 0 for domain in self.domains}

        # Get the directory containing this script (response_utils.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to domain_mapping.json
        domain_mapping_path = os.path.join(script_dir, "bar_domain_mapping.json")

        # Check if the file exists and load the domain mapping
        if not os.path.exists(domain_mapping_path):
            raise FileNotFoundError(f"Domain mapping file not found at: {domain_mapping_path}")

        with open(domain_mapping_path, "r") as f:
            self.domain_mapping = json.load(f)
        
        self.comparison_dict = comparison_dict

    def map_domain(self, domain):
        """Map domain names to standardized versions using a preloaded mapping."""
        normalized_domain = re.sub(r'\W+', '_', domain).lower()
        return self.domain_mapping.get(normalized_domain, domain)

    def grade_chosen_option(self, chosen_option_label, ground_truth_label, domain):
        """Grade the chosen option label for correctness and track domain-specific metrics."""
        if chosen_option_label not in ['A', 'B', 'C', 'D']:
            self.malformed_count['malformed_label'] += 1
            return

        # Increment total count for the domain
        if domain in self.domain_label_total_counts:
            self.domain_label_total_counts[domain] += 1

        # Increment correct count if the chosen option matches the ground truth
        if ground_truth_label == chosen_option_label:
            self.correct_predictions += 1
            if domain in self.domain_label_correct_counts:
                self.domain_label_correct_counts[domain] += 1

    def grade_domain(self, chosen_domain, ground_truth_domain):
        """Grade the correctness of the domain."""
        mapped_chosen_domain = self.map_domain(chosen_domain)
        mapped_ground_truth_domain = self.map_domain(ground_truth_domain)

        # Track predictions for confidence analysis
        if mapped_chosen_domain in self.domain_prediction_counts:
            self.domain_prediction_counts[mapped_chosen_domain] += 1

        if mapped_chosen_domain == mapped_ground_truth_domain:
            return True

        self.malformed_count['malformed_domain'] += 1
        return False

    def grade_response(self, data_item):
        """Grade a single response by evaluating both the chosen option and domain."""
        self.total_predictions += 1

        # Ground truth domain
        ground_truth_domain = self.map_domain(data_item['ground_truth']['domain'])

        # Increment combined total count for this domain
        if ground_truth_domain in self.domain_combined_total_counts:
            self.domain_combined_total_counts[ground_truth_domain] += 1

        # Grade chosen option
        self.grade_chosen_option(
            data_item['response']['chosen_option_label'],
            data_item['ground_truth']['correct_answer'],
            ground_truth_domain
        )

        # Grade domain
        is_domain_correct = self.grade_domain(
            data_item['response']['domain'],
            data_item['ground_truth']['domain']
        )

        # Track error patterns
        chosen_option = data_item['response']['chosen_option_label']
        correct_option = data_item['ground_truth']['correct_answer']

        if is_domain_correct and chosen_option != correct_option:
            self.error_types['correct_domain_wrong_label'] += 1
        elif not is_domain_correct and chosen_option == correct_option:
            self.error_types['correct_label_wrong_domain'] += 1
        elif not is_domain_correct and chosen_option != correct_option:
            self.error_types['both_wrong'] += 1

        # Increment combined correct count if both label and domain are correct
        if is_domain_correct and chosen_option == correct_option:
            self.correct_label_and_domain += 1
            if ground_truth_domain in self.domain_combined_correct_counts:
                self.domain_combined_correct_counts[ground_truth_domain] += 1

    def calculate_accuracy(self, correct_counts, total_counts):
        """Calculate accuracy for a given set of counts."""
        return {
            f"{domain.replace(' ', '_')}_accuracy": correct / total if total > 0 else 0
            for domain, (correct, total) in zip(correct_counts.keys(), zip(correct_counts.values(), total_counts.values()))
        }

    def calculate_domain_confidence(self):
        """Calculate confidence as prediction frequency for each domain."""
        return {
            f"{domain.replace(' ', '_')}_confidence": count / self.total_predictions if self.total_predictions > 0 else 0
            for domain, count in self.domain_prediction_counts.items()
        }

    def finalise_metrics(self):
        """Calculate and finalize metrics, adding them to the comparison dictionary."""
        # Overall accuracies
        label_accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        combined_accuracy = self.correct_label_and_domain / self.total_predictions if self.total_predictions > 0 else 0
        misclassification_rate = 1 - label_accuracy

        # Per-domain accuracies
        domain_label_accuracy = self.calculate_accuracy(
            self.domain_label_correct_counts, self.domain_label_total_counts
        )
        domain_combined_accuracy = self.calculate_accuracy(
            self.domain_combined_correct_counts, self.domain_combined_total_counts
        )

        # Domain confidence
        domain_confidence = self.calculate_domain_confidence()

        # Add metrics to comparison dictionary
        self.comparison_dict['metrics'] = {
            'label_accuracy': label_accuracy,
            'misclassification_rate': misclassification_rate,
            'combined_accuracy': combined_accuracy,
            'malformed_label': self.malformed_count['malformed_label'],
            'malformed_domain': self.malformed_count['malformed_domain'],
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'correct_label_and_domain': self.correct_label_and_domain,
            **domain_label_accuracy,
            **domain_combined_accuracy,
            **domain_confidence,
            **self.error_types  # Add error pattern analysis
        }

        return self.comparison_dict 

    def dump_metrics(self, file_path):
        """Dump the metrics to a file."""
        with open(file_path, "w") as f:
            json.dump(self.comparison_dict, f, indent=4)
        logging.info(f"Dumped the metrics to {file_path}")
