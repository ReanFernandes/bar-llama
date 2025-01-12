import json
import logging
from typing import Dict, List, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self, correction_mode: bool = False, structured_explanation: bool = False):
        self.correction_mode = correction_mode
        self.structured_explanation = structured_explanation
        self.valid_data = []
        self.error_count = 0

    def validate_structured_explanation(self, explanation: Dict) -> bool:
        required_fields = {
            "Legal_Concept": str,
            "Fact_Analysis": str,
            "Rule_Application": str,
            "Legal_Conclusion": str
        }

        # Check if all required fields exist and are of correct type
        for field, field_type in required_fields.items():
            if field not in explanation:
                self._log_error(f"Missing field in structured explanation: {field}", {})
                return False
            if not isinstance(explanation[field], field_type):
                self._log_error(f"Invalid type for {field} in structured explanation", {})
                return False
            if not explanation[field].strip():
                self._log_error(f"Empty value in structured explanation: {field}", {})
                return False

        return True

    def validate_structure(self, data_point: Dict) -> bool:
        required_fields = {
            "domain": str,
            "question_number": int,
            "question": str,
            "options": dict,
            "correct_answer": str,
            "explanation": dict if self.structured_explanation else str
        }

        required_options = {
            "optionA": str,
            "optionB": str,
            "optionC": str,
            "optionD": str
        }

        # Check if all required fields exist and are of correct type
        for field, field_type in required_fields.items():
            if field not in data_point:
                self._log_error(f"Missing field: {field}", data_point)
                return False
            if not isinstance(data_point[field], field_type):
                self._log_error(f"Invalid type for {field}", data_point)
                return False

        # Check if all required options exist
        for option, option_type in required_options.items():
            if option not in data_point["options"]:
                self._log_error(f"Missing option: {option}", data_point)
                return False
            if not isinstance(data_point["options"][option], option_type):
                self._log_error(f"Invalid type for {option}", data_point)
                return False
            if not data_point["options"][option].strip():
                self._log_error(f"Empty value in {option}", data_point)
                return False

        # Validate explanation based on structure type
        if self.structured_explanation:
            if not self.validate_structured_explanation(data_point["explanation"]):
                return False
        else:
            if not data_point["explanation"].strip():
                self._log_error("Empty value in explanation", data_point)
                return False

        # Check for empty values in other fields
        for field, value in data_point.items():
            if field not in ["options", "explanation"]:
                if isinstance(value, str) and not value.strip():
                    self._log_error(f"Empty value in {field}", data_point)
                    return False

        # Validate correct_answer is one of A, B, C, or D
        if data_point["correct_answer"] not in ["A", "B", "C", "D"]:
            self._log_error("Invalid correct_answer", data_point)
            return False

        return True

    def _log_error(self, error_message: str, data_point: Dict):
        self.error_count += 1
        logger.error(f"""
        Error: {error_message}
        Domain: {data_point.get('domain', 'N/A')}
        Question Number: {data_point.get('question_number', 'N/A')}
        """)

    def validate_dataset(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                dataset = json.load(file)

            if not isinstance(dataset, list):
                logger.error("Dataset must be a list of questions")
                return []

            self.valid_data = []
            self.error_count = 0

            for data_point in dataset:
                if self.validate_structure(data_point):
                    self.valid_data.append(data_point)

            logger.info(f"""
            Validation Complete:
            Total questions processed: {len(dataset)}
            Valid questions: {len(self.valid_data)}
            Errors found: {self.error_count}
            Explanation format: {'Structured' if self.structured_explanation else 'Unstructured'}
            """)

            if self.correction_mode:
                output_filepath = filepath.replace('.json', '_corrected.json')
                with open(output_filepath, 'w', encoding='utf-8') as file:
                    json.dump(self.valid_data, file, indent=4)
                logger.info(f"Corrected dataset saved to: {output_filepath}")

            return self.valid_data

        except json.JSONDecodeError:
            logger.error("Invalid JSON file")
            return []
        except FileNotFoundError:
            logger.error("File not found")
            return []

# Example usage
if __name__ == "__main__":
    # Example with structured explanation
    validator_structured = DatasetValidator(
        correction_mode=True,
        structured_explanation=True
    )
    validator_structured.validate_dataset("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/seed_dataset/distilled_dataset.json")

    # Example with unstructured explanation
    validator_unstructured = DatasetValidator(
        correction_mode=False,
        structured_explanation=False
    )
    validator_unstructured.validate_dataset("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/seed_dataset/raw_dataset.json")

     # Example with unstructured explanation
    validator_unstructured = DatasetValidator(
        correction_mode=False,
        structured_explanation=False
    )
    validator_unstructured.validate_dataset("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/test_sets/bar_set_1.json")

    validator_unstructured = DatasetValidator(
        correction_mode=False,
        structured_explanation=False
    )
    validator_unstructured.validate_dataset("/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/test_sets/bar_set_2.json")