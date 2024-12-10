import json
from collections import defaultdict

# Load the JSON files
with open('/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/data_distillation/distillation_result/distilled_dataset.json', 'r') as f:
    mother_data = json.load(f)

with open('/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/data_distillation/distillation_result/distilled_dataset_extra.json', 'r') as f:
    daughter_data = json.load(f)

# Group mother entries by domain and track the highest question_number in each domain
domain_max_question_number = defaultdict(int)

for entry in mother_data:
    domain = entry['domain']
    question_number = entry['question_number']
    domain_max_question_number[domain] = max(domain_max_question_number[domain], question_number)

# Adjust question_numbers in the daughter entries
for entry in daughter_data:
    domain = entry['domain']
    # Start numbering after the last question_number in the same domain from the mother
    entry['question_number'] = domain_max_question_number[domain] + 1
    # Update the maximum for the domain
    domain_max_question_number[domain] += 1

# Merge the daughter data into the mother data
mother_data.extend(daughter_data)

# Save the updated mother JSON back to file
with open('updated_mother.json', 'w') as f:
    json.dump(mother_data, f, indent=4)
