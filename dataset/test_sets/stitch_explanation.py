import re
import json

def extract_explanations(explanation_text):
    # Regex pattern to match "Answer to Question X" and extract the question number and explanation
    pattern = r"Answer to Question (\d+)\.\nExplanation:(.*?)(?=Answer to Question \d+|$)"
    matches = re.findall(pattern, explanation_text, re.DOTALL)
    
    explanations = {}
    for match in matches:
        question_number = match[0]
        explanation = match[1].strip().replace("\n", " ")
        explanations[question_number] = explanation
    return explanations

def append_explanations_to_json(parsed_questions, explanations):
    for question in parsed_questions:
        question_number = question['question_number']
        if question_number in explanations:
            question['explanation'] = explanations[question_number]
        else:
            question['explanation'] = "No explanation available."
    return parsed_questions

def main(parsed_questions_file, explanations_file, output_file):
    # Load the parsed questions from JSON
    try:
        with open(parsed_questions_file, 'r') as infile:
            parsed_questions = json.load(infile)
    except Exception as e:
        print(f"Error reading the parsed questions file: {e}")
        return
    
    # Load the explanations from the explanations text file
    try:
        with open(explanations_file, 'r') as exp_file:
            explanation_text = exp_file.read()
    except Exception as e:
        print(f"Error reading the explanations file: {e}")
        return

    # Extract explanations
    explanations = extract_explanations(explanation_text)

    # Append explanations to the parsed questions
    updated_questions = append_explanations_to_json(parsed_questions, explanations)
    
    # Save the updated JSON with explanations
    try:
        with open(output_file, 'w') as outfile:
            json.dump(updated_questions, outfile, indent=4)
        print(f"Updated questions with explanations saved to {output_file}.")
    except Exception as e:
        print(f"Error writing to the output file: {e}")

if __name__ == "__main__":
    parsed_questions_file = 'bar_test_set_2_complete.json'  # The file containing parsed questions
    explanations_file = 'bar_test_2_explanations'  # The file containing explanations in the format shown
    output_file = 'bar_test_set_2_explanations.json'  # Output file with updated data

    # Run the main function
    main(parsed_questions_file, explanations_file, output_file)

