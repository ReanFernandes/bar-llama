import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Read the CSV
df = pd.read_csv('master_metrics_after_the_fact.csv')

# Filter for validation sets only
df = df[df['evaluation_dataset'].str.contains('val_set_[1-4]$', regex=True)]

# Convert num_training_samples to numeric and handle any conversion issues
df['num_training_samples'] = pd.to_numeric(df['num_training_samples'], errors='coerce')

# Clean data
df_clean = df.dropna(subset=['num_training_samples', 'label_accuracy'])

# Add dataset labels to plots
def create_scatter_with_datasets(x, y, data, ax=None):
    if ax is None:
        ax = plt.gca()
    
    datasets = ['val_set_1', 'val_set_2', 'val_set_3', 'val_set_4']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for dataset, color in zip(datasets, colors):
        mask = data['evaluation_dataset'] == dataset
        ax.scatter(data[mask][x], data[mask][y], 
                  alpha=0.3, label=dataset, color=color)

# 1. Enhanced Training Samples vs Label Accuracy
plt.figure(figsize=(12, 7))

create_scatter_with_datasets('num_training_samples', 'label_accuracy', df_clean)

# Group by training samples for mean and std
df_grouped = df_clean.groupby(['num_training_samples', 'evaluation_dataset'])['label_accuracy'].agg(['mean', 'std']).reset_index()

# Add mean lines for each dataset
for dataset in df_grouped['evaluation_dataset'].unique():
    dataset_data = df_grouped[df_grouped['evaluation_dataset'] == dataset]
    plt.errorbar(dataset_data['num_training_samples'], dataset_data['mean'],
                yerr=dataset_data['std'], fmt='o-', 
                label=f'{dataset} (mean ± std)', capsize=5, capthick=1, 
                elinewidth=1, markersize=4)

plt.xlabel('Number of Training Samples')
plt.ylabel('Label Accuracy')
plt.title('Impact of Training Samples on Label Accuracy\nby Validation Dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('plots/val1_enhanced_training_impact.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Domain-specific accuracies with subplots
domains = [col for col in df_clean.columns if col.endswith('_accuracy') 
          and col != 'label_accuracy' and 'combined' not in col.lower()]

fig, axs = plt.subplots(3, 3, figsize=(20, 20))
axs = axs.ravel()

for idx, domain in enumerate(domains):
    create_scatter_with_datasets('num_training_samples', domain, df_clean, ax=axs[idx])
    
    # Add mean lines for each dataset
    for dataset in df_clean['evaluation_dataset'].unique():
        dataset_data = df_clean[df_clean['evaluation_dataset'] == dataset]
        df_grouped = dataset_data.groupby('num_training_samples')[domain].mean()
        axs[idx].plot(df_grouped.index, df_grouped.values, '-', 
                     linewidth=2, label=f'{dataset} (mean)')
    
    axs[idx].set_title(domain.replace('_accuracy', '').replace('_', ' '))
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlabel('Number of Training Samples')
    axs[idx].set_ylabel('Accuracy')
    axs[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove extra subplot if any
if len(domains) < 9:
    for idx in range(len(domains), 9):
        fig.delaxes(axs[idx])

plt.tight_layout()
plt.savefig('plots/val2_domain_specific_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Enhanced Explanation Type Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Create separate violin plots for each dataset
for ax, metric in zip([ax1, ax2], ['label_accuracy', 'malformed_label']):
    sns.violinplot(x='explanation_type', y=metric, hue='evaluation_dataset', 
                  data=df_clean, split=True, ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()} Distribution\nby Explanation Type and Dataset')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add mean values as text
    means = df_clean.groupby(['evaluation_dataset', 'explanation_type'])[metric].mean()
    for i, (dataset, exp_type) in enumerate(means.index):
        ax.text(i % len(df_clean['explanation_type'].unique()), 
                df_clean[metric].max() * (0.9 + i//len(df_clean['explanation_type'].unique())*0.05),
                f'{dataset}: {means[(dataset, exp_type)]:.3f}',
                horizontalalignment='center')

plt.tight_layout()
plt.savefig('plots/val3_explanation_type_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Correlation Matrix by Dataset
numeric_cols = ['label_accuracy', 'num_training_samples', 'malformed_label',
                'correct_label_and_domain', 'correct_domain_wrong_label',
                'correct_label_wrong_domain', 'both_wrong']

fig, axs = plt.subplots(2, 2, figsize=(20, 20))
axs = axs.ravel()

for idx, dataset in enumerate(['val_set_1', 'val_set_2', 'val_set_3', 'val_set_4']):
    dataset_data = df_clean[df_clean['evaluation_dataset'] == dataset]
    corr_matrix = dataset_data[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0, fmt='.2f',
                square=True, cbar_kws={"shrink": .5}, ax=axs[idx])
    axs[idx].set_title(f'Correlation Matrix - {dataset}')

plt.tight_layout()
plt.savefig('plots/val4_correlation_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Prediction Type Analysis by Dataset
prediction_cols = ['correct_label_and_domain', 'correct_domain_wrong_label',
                  'correct_label_wrong_domain', 'both_wrong']

plt.figure(figsize=(15, 8))

# Calculate percentages for each dataset
datasets = df_clean['evaluation_dataset'].unique()
x = np.arange(len(prediction_cols))
width = 0.2
multiplier = 0

for dataset in datasets:
    dataset_data = df_clean[df_clean['evaluation_dataset'] == dataset]
    df_pct = dataset_data[prediction_cols].div(dataset_data['total_predictions'], axis=0) * 100
    
    offset = width * multiplier
    rects = plt.bar(x + offset, df_pct.mean(), width, label=dataset)
    plt.bar_label(rects, fmt='%.1f%%')
    multiplier += 1

plt.title('Average Distribution of Prediction Types by Dataset')
plt.xlabel('Prediction Type')
plt.ylabel('Percentage of Total Predictions')
plt.xticks(x + width * 1.5, [col.replace('_', ' ').title() for col in prediction_cols], rotation=45)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/val5_prediction_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

print("Enhanced plots with validation set analysis have been saved to the 'plots' folder!")