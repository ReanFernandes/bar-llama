import json

def load_correct_answers(file_path):
    correct_answers = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    question_number = parts[0].strip('.')
                    correct_option = parts[1]
                    domain = " ".join(parts[2:])
                    correct_answers[question_number] = {
                        "correct_option": correct_option,
                        "domain": domain
                    }
    except Exception as e:
        print(f"Error reading the correct answers file: {e}")
    return correct_answers

def append_correct_answers(parsed_questions, correct_answers):
    for question in parsed_questions:
        question_number = question['question_number']
        if question_number in correct_answers:
            question['correct_answer'] = correct_answers[question_number]["correct_option"]
            question['domain'] = correct_answers[question_number]["domain"]

def main(parsed_questions_file, correct_answers_file, output_file):
    # Load the parsed questions from JSON
    try:
        with open(parsed_questions_file, 'r') as infile:
            parsed_questions = json.load(infile)
    except Exception as e:
        print(f"Error reading the parsed questions file: {e}")
        return
    
    # Load the correct answers and domains
    correct_answers = load_correct_answers(correct_answers_file)
    
    # Append correct answers and domains to parsed questions
    append_correct_answers(parsed_questions, correct_answers)
    
    # Save the updated JSON with correct answers and domains
    try:
        with open(output_file, 'w') as outfile:
            json.dump(parsed_questions, outfile, indent=4)
        print(f"Updated questions with correct answers and domains saved to {output_file}.")
    except Exception as e:
        print(f"Error writing to the output file: {e}")

if __name__ == "__main__":
    parsed_questions_file = 'bar_test_set_1.json'  # The file containing parsed questions
    correct_answers_file = 'raw_bar_set_1_ans_domain'  # The file containing correct answers and domains
    output_file = 'bar_test_set_1_complete.json'  # Output file with updated data

    # Run the main function
    main(parsed_questions_file, correct_answers_file, output_file)

