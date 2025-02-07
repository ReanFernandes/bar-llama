import json
from collections import Counter
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_answer_distribution(json_file_path):
    # Read JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract correct answers
    correct_answers = [entry['correct_answer'] for entry in data]
    
    # Calculate distribution
    distribution = Counter(correct_answers)
    
    # Calculate percentages
    total = len(correct_answers)
    percentages = {k: (v/total)*100 for k, v in distribution.items()}
    
    # Print results
    print("\nDistribution of correct answers:")
    for answer, count in distribution.items():
        print(f"Option {answer}: {count} ({percentages[answer]:.1f}%)")
    
    # Create and save plot
    plt.figure(figsize=(8, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.title('Distribution of Correct Answers')
    plt.xlabel('Answer Option')
    plt.ylabel('Count')
    
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    # Get dataset name from file path
    dataset_name = Path(json_file_path).stem
    
    # Save plot
    plt.savefig(f'plots/answer_distribution_{dataset_name}.png')
    plt.close()

# Usage
# analyze_answer_distribution('path_to_your_json_file.json')

if __name__=="__main__":
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/bar_set_1.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/bar_set_2.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/test_1_out.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/test_1_out.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/test_2_out.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/test_3_out.json')
    analyze_answer_distribution(json_file_path='/home/fr/fr_fr/fr_rf1031/bar-llama/dataset/test_sets/test_4_out.json')
