Contains the original train sets. This folder is used as a starting point from which we put together the dataset in a configuration specific format for the finetuning
## Datasets

This folder contains the original training datasets used as a starting point for creating the configuration-specific datasets for fine-tuning.

### `high_temp_structured_expl_dataset.json`

The distilled dataset which was generated with `temp=0.6` with LLaMa 3 70-B quantised. This contained the explanations in a structured format
### `low_temp_structured_expl_dataset.json`

Similar to the `high_temp_structured_expl_dataset.json`, this dataset also contains structured explanations for multiple-choice questions in the criminal law domain. 

### `unstructured_expl_dataset.json`
Original dataset which contains human generated explanations. Not distilled. 