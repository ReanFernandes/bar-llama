"""
Test how much overlap exists between the train and test set, and remove question from the train set that are found in test.
Note: As of 15.12.2024, all the samples have been validated and there is no overlap between train and test.
"""

from rapidfuzz import fuzz
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def find_and_remove_fuzzy_overlaps(train_file, test_file, output_train_file, output_overlap_file, similarity_threshold=70):
    """
    Identifies overlapping samples between train and test sets based on 'domain' and fuzzy matching of 'question' fields,
    removes them from the training set, and saves overlaps to a separate file. Logs comparison statistics.

    Parameters:
        train_file (str): Path to the training set JSON file.
        test_file (str): Path to the test set JSON file.
        output_train_file (str): Path to save the cleaned training set JSON file.
        output_overlap_file (str): Path to save the overlapping samples JSON file.
        similarity_threshold (int): Minimum similarity percentage to consider two questions as overlapping.
    """
    # Load the JSON files
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Prepare test data grouped by domain for efficient lookup
    test_questions_by_domain = {}
    for entry in test_data:
        test_questions_by_domain.setdefault(entry["domain"], []).append((entry["question"], entry["question_number"]))

    total_domains = len(test_questions_by_domain)
    logging.info(f"Total domains in test set: {total_domains}")

    overlaps = []
    filtered_train_data = []
    domain_comparison_stats = {}

    # Compare questions using fuzzy matching
    for train_entry in train_data:
        train_domain = train_entry["domain"]
        train_question = train_entry["question"]

        # Check only questions within the same domain
        if train_domain in test_questions_by_domain:
            test_questions = test_questions_by_domain[train_domain]
            matched = False

            for test_question, test_question_number in test_questions:
                similarity = fuzz.ratio(train_question, test_question)
                if similarity >= similarity_threshold:
                    overlaps.append({
                        "train_entry": train_entry,
                        "metadata": {
                            "train_question_number": train_entry["question_number"],
                            "test_question_number": test_question_number,
                            "similarity": similarity
                        }
                    })
                    matched = True

                    # Log overlapping question per domain
                    domain_comparison_stats[train_domain] = domain_comparison_stats.get(train_domain, 0) + 1
                    break

            if not matched:
                filtered_train_data.append(train_entry)
        else:
            # No matching domain, keep the entry
            filtered_train_data.append(train_entry)

    # Save the cleaned training data
    with open(output_train_file, 'w') as f:
        json.dump(filtered_train_data, f, indent=4)

    # Save the overlaps
    with open(output_overlap_file, 'w') as f:
        json.dump(overlaps, f, indent=4)

    # Log domain-level statistics
    logging.info("Comparison completed.")
    logging.info(f"Total questions compared: {len(train_data)}")
    logging.info(f"Total overlapping questions found: {len(overlaps)}")
    for domain, count in domain_comparison_stats.items():
        logging.info(f"Domain: {domain}, Overlapping Questions: {count}")

    print(f"Cleaned training set saved to {output_train_file}")
    print(f"Overlapping samples saved to {output_overlap_file}")

# Paths to the input and output files
train_file = "/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/seed_dataset/distilled_dataset.json"
test_file = "/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/test_sets/bar_set_2.json"
output_train_file = "cleaned_train.json"
output_overlap_file = "fuzzy_overlaps.json"

# Run the script
find_and_remove_fuzzy_overlaps(train_file, test_file, output_train_file, output_overlap_file, similarity_threshold=90)
