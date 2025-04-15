# Research Dataset Documentation

## Overview
This repository contains the consolidated results dataset from our research paper. The dataset includes all inference runs recorded and was used to derive the plots and findings presented in our publication.

## Dataset Structure
The CSV file contains the following columns:

### Model Information
- `model_name`: Name of the model used
- `seed`: Random seed used for reproducibility
- `training_status`: Status of model training
- `quantisation_status`: Status of model quantization
- `training_dataset`: Dataset used for training
- `num_training_samples`: Number of samples used in training
- `num_training_domains`: Number of domains covered in training
- `randomised_training_samples`: Whether training samples were randomized

### Generation Parameters
- `generation_strategy`: Strategy used for generation
- `prompt_type`: Type of prompt used
- `explanation_type`: Type of explanation provided
- `response_type`: Type of response generated
- `response_format`: Format of the response

### Evaluation Metrics
- `evaluation_dataset`: Dataset used for evaluation
- `label_accuracy`: Accuracy of label predictions
- `misclassification_rate`: Rate of misclassifications
- `combined_accuracy`: Overall accuracy combining multiple metrics
- `malformed_label`: Count or rate of malformed labels
- `malformed_domain`: Count or rate of malformed domains
- `total_predictions`: Total number of predictions made
- `correct_predictions`: Number of correct predictions
- `correct_label_and_domain`: Number of predictions with both correct label and domain

### Domain-Specific Accuracy
- `Constitutional_Law_accuracy`: Accuracy for Constitutional Law domain
- `Contracts_accuracy`: Accuracy for Contracts domain
- `Criminal_Law_accuracy`: Accuracy for Criminal Law domain
- `Evidence_accuracy`: Accuracy for Evidence domain
- `Real_Property_accuracy`: Accuracy for Real Property domain
- `Torts_accuracy`: Accuracy for Torts domain
- `Civil_Procedure_accuracy`: Accuracy for Civil Procedure domain

### Domain-Specific Confidence
- `Constitutional_Law_confidence`: Confidence for Constitutional Law domain
- `Contracts_confidence`: Confidence for Contracts domain
- `Criminal_Law_confidence`: Confidence for Criminal Law domain
- `Evidence_confidence`: Confidence for Evidence domain
- `Real_Property_confidence`: Confidence for Real Property domain
- `Torts_confidence`: Confidence for Torts domain
- `Civil_Procedure_confidence`: Confidence for Civil Procedure domain

### Error Analysis
- `correct_domain_wrong_label`: Count of predictions with correct domain but wrong label
- `correct_label_wrong_domain`: Count of predictions with correct label but wrong domain
- `both_wrong`: Count of predictions with both wrong label and domain

