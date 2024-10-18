import json

def parse_questions_from_text(text):
    try:
        # Split the text by questions
        questions = text.split("Question ")[1:]
        parsed_data = []

        for question in questions:
            # Split question number, question body, and options
            lines = question.strip().split("\n")
            if len(lines) < 2:
                continue  # Skip if the question structure is invalid

            question_number = lines[0].strip('.')
            question_body = []
            options = {}

            # Find where options start (A) and extract question body
            option_start = -1
            for i, line in enumerate(lines[1:]):
                if line.startswith("(A)"):
                    option_start = i + 1
                    break
                question_body.append(line)
            
            if option_start == -1:
                continue  # Skip if no options were found

            question_body = " ".join(question_body).strip()

            # Extract options
            option_letters = ['(A)', '(B)', '(C)', '(D)']
            current_option = ""
            for line in lines[option_start:]:
                if any(line.startswith(opt) for opt in option_letters):
                    current_option = line[:3]
                    options[current_option] = line[1:].strip()
                elif current_option:  # Append to the current option if needed
                    options[current_option] += " " + line.strip()

            # Ensure that all four options are present
            formatted_options = {
                "optionA": f"A. {options.get('(A)', '').strip()}",
                "optionB": f"B. {options.get('(B)', '').strip()}",
                "optionC": f"C. {options.get('(C)', '').strip()}",
                "optionD": f"D. {options.get('(D)', '').strip()}"
            }
            
            # Store parsed question
            parsed_data.append({
                "question_number": question_number,
                "question": question_body,
                "options": formatted_options
            })

        return parsed_data

    except Exception as e:
        print(f"Error parsing the text: {e}")
        return []

def jsonify_questions(parsed_data, output_file):
    try:
        with open(output_file, 'w') as outfile:
            json.dump(parsed_data, outfile, indent=4)
        print(f"Questions have been successfully parsed and saved to {output_file}.")
    except Exception as e:
        print(f"Error saving the file: {e}")

def read_text_file(input_file):
    try:
        with open(input_file, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return ""

if __name__ == "__main__":
    input_file = '/mnt/b1aa7336-e92e-43c1-9d8b-d627549a7f7d/bar-llama/dataset/test_sets/raw_bar_set_2'  # Path to your input text file
    output_file = 'bar_test_set_2.json'  # Path for the output JSON file

    # Read the text from file
    text = read_text_file(input_file)
    
    if text:
        # Parse and jsonify the text
        parsed_data = parse_questions_from_text(text)
        jsonify_questions(parsed_data, output_file)
    else:
        print("No text to process.")

