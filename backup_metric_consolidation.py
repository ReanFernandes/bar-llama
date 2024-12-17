"""
The inference code in eval.py ensures that all the concurrent runs of the experiments unload the metrics into a single 
consolidated dataframe. In the event that this doesnt happen, the files will already have been saved to the folder called 'metrics'
this code will walk through that entire folder and gather all the metrics into a single dataframe. Have not tested it yet, and probably wont need 
it but just in case lol
"""

import os
import json
import pandas as pd

def consolidate_metrics(base_directory, output_file):
    """
    Consolidate individual metrics JSON files into a single DataFrame.
    Properly handles 'domain' keys that are lists.
    
    Args:
        base_directory (str): Root directory where metrics files are saved.
        output_file (str): Path to save the consolidated DataFrame as CSV/JSON.
    """
    all_metrics = []  # List to store all parsed metrics
    failed_files = []  # List to store any files that failed to load

    # Traverse the directory structure
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".json"):  # Process only JSON files
                file_path = os.path.join(root, file)
                try:
                    # Load the JSON file
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Flatten the data structure
                    flattened_data = {
                        **data["config"],  # Unpack all configuration fields
                        **data["metrics"]  # Unpack all metrics fields
                    }

                    # Special handling for 'domain' key (serialize it to JSON)
                    if "domains" in flattened_data and isinstance(flattened_data["domains"], list):
                        flattened_data["domains"] = json.dumps(flattened_data["domains"])

                    # Add file path for traceability (optional)
                    flattened_data["file_path"] = file_path

                    # Append to the list
                    all_metrics.append(flattened_data)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    failed_files.append(file_path)

    # Create DataFrame from all collected metrics
    if all_metrics:
        df = pd.DataFrame(all_metrics)

        # Save to CSV or JSON based on output_file extension
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        elif output_file.endswith(".json"):
            df.to_json(output_file, orient="records", indent=4)
        else:
            print("Unsupported file format. Use .csv or .json.")
            return

        print(f"Consolidated metrics saved to {output_file}")
        if failed_files:
            print(f"Warning: Failed to process {len(failed_files)} files. Check for issues.")
    else:
        print("No metrics files were found. Nothing to consolidate.")

# Example usage
if __name__ == "__main__":

    # Note, probably going to use absolute paths here since relative path is too confusing for my tiny brain
    base_directory = "./model_outputs/metrics"  # Root directory where metrics files are stored
    output_file = "./consolidated_metrics.csv"  # Path to save the consolidated file

    consolidate_metrics(base_directory, output_file)
