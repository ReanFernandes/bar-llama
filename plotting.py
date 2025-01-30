import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Read the CSV
df = pd.read_csv('new_fixed_metrics.csv')

# Convert num_training_samples to numeric and handle any conversion issues
df['num_training_samples'] = pd.to_numeric(df['num_training_samples'], errors='coerce')

# Clean data
df_clean = df.dropna(subset=['num_training_samples', 'label_accuracy'])

# 1. Enhanced Training Samples vs Label Accuracy
plt.figure(figsize=(12, 7))

# Group by training samples for mean and std
df_grouped = df_clean.groupby('num_training_samples')['label_accuracy'].agg(['mean', 'std']).reset_index()

# Main scatter with reduced alpha
plt.scatter(df_clean['num_training_samples'], df_clean['label_accuracy'], 
           alpha=0.3, label='Individual runs', color='lightblue')

# Add mean line with error bands
plt.errorbar(df_grouped['num_training_samples'], df_grouped['mean'],
            yerr=df_grouped['std'], color='red', fmt='o-', 
            label='Mean ± std', capsize=5, capthick=2, 
            elinewidth=2, markersize=8)

plt.xlabel('Number of Training Samples')
plt.ylabel('Label Accuracy')
plt.title('Impact of Training Samples on Label Accuracy\nwith Mean Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/1_enhanced_training_impact.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Domain-specific accuracies with subplots
domains = [col for col in df_clean.columns if col.endswith('_accuracy') 
          and col != 'label_accuracy' and 'combined' not in col.lower()]

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.ravel()

for idx, domain in enumerate(domains):
    # Plot individual points
    axs[idx].scatter(df_clean['num_training_samples'], df_clean[domain], 
                    alpha=0.3, color='lightblue')
    
    # Add mean line
    df_grouped = df_clean.groupby('num_training_samples')[domain].mean()
    axs[idx].plot(df_grouped.index, df_grouped.values, 'r-', 
                 linewidth=2, label='Mean')
    
    axs[idx].set_title(domain.replace('_accuracy', '').replace('_', ' '))
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlabel('Number of Training Samples')
    axs[idx].set_ylabel('Accuracy')
    axs[idx].legend()

# Remove extra subplot if any
if len(domains) < 9:
    for idx in range(len(domains), 9):
        fig.delaxes(axs[idx])

plt.tight_layout()
plt.savefig('plots/2_domain_specific_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Enhanced Explanation Type Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Violin plots with individual points
sns.violinplot(x='explanation_type', y='label_accuracy', data=df_clean, ax=ax1)
sns.stripplot(x='explanation_type', y='label_accuracy', data=df_clean, 
             color='red', alpha=0.3, size=4, ax=ax1)
ax1.set_title('Label Accuracy Distribution by Explanation Type')

sns.violinplot(x='explanation_type', y='malformed_label', data=df_clean, ax=ax2)
sns.stripplot(x='explanation_type', y='malformed_label', data=df_clean, 
             color='red', alpha=0.3, size=4, ax=ax2)
ax2.set_title('Malformed Labels Distribution by Explanation Type')

# Add mean values as text
for ax, metric in zip([ax1, ax2], ['label_accuracy', 'malformed_label']):
    means = df_clean.groupby('explanation_type')[metric].mean()
    for i, mean in enumerate(means):
        ax.text(i, df_clean[metric].max() * 0.9, f'Mean: {mean:.3f}', 
                horizontalalignment='center')

plt.tight_layout()
plt.savefig('plots/3_explanation_type_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Correlation Matrix
numeric_cols = ['label_accuracy', 'num_training_samples', 'malformed_label',
                'correct_label_and_domain', 'correct_domain_wrong_label',
                'correct_label_wrong_domain', 'both_wrong']

# Create correlation matrix
corr_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0, fmt='.2f',
            square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Key Metrics')
plt.tight_layout()
plt.savefig('plots/4_correlation_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Prediction Type Analysis
prediction_cols = ['correct_label_and_domain', 'correct_domain_wrong_label',
                  'correct_label_wrong_domain', 'both_wrong']

# Calculate percentages
df_pct = df_clean[prediction_cols].div(df_clean['total_predictions'], axis=0) * 100

plt.figure(figsize=(10, 6))
ax = df_pct.mean().plot(kind='bar')
plt.title('Average Distribution of Prediction Types')
plt.xlabel('Prediction Type')
plt.ylabel('Percentage of Total Predictions')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add percentage labels on top of bars
for i, v in enumerate(df_pct.mean()):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('plots/5_prediction_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

print("Enhanced plots have been saved to the 'plots' folder!")