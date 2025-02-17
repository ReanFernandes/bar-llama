import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read and prepare the data
df = pd.read_csv('master_metrics.csv')

# Filter for validation sets only
df = df[df['evaluation_dataset'].str.contains('test_set_[1-2]$', regex=True)]

# Replace 'all' with 250 in num_training_samples
df['num_training_samples'] = df['num_training_samples'].replace('all', '250')
df['num_training_samples'] = pd.to_numeric(df['num_training_samples'], errors='coerce')
df_clean = df.dropna(subset=['num_training_samples', 'label_accuracy'])

# Create the plot
plt.figure(figsize=(12, 8))

# Set style
# plt.style.use('seaborn')
colors = ['#2E86C1', '#E67E22']
datasets = ['test_set_1', 'test_set_2']

# Plot for each validation set
for idx, (dataset, color) in enumerate(zip(datasets, colors)):
    dataset_data = df_clean[df_clean['evaluation_dataset'] == dataset]
    
    # Calculate mean and standard deviation for each number of training samples
    stats = dataset_data.groupby('num_training_samples')['label_accuracy'].agg(['mean', 'std']).reset_index()
    
    # Sort by number of training samples to ensure proper line connection
    stats = stats.sort_values('num_training_samples')
    
    # Plot the trend line
    plt.plot(stats['num_training_samples'], stats['mean'], 
            color=color, linewidth=3, label=f'Val Set {idx+1}')
    
    # Add confidence interval
    plt.fill_between(stats['num_training_samples'],
                    stats['mean'] - stats['std'],
                    stats['mean'] + stats['std'],
                    color=color, alpha=0.15)

# Customize the plot
plt.xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
plt.ylabel('Label Accuracy', fontsize=12, fontweight='bold')
plt.title('Impact of Training Samples on Label Accuracy\nAcross Different Validation Sets', 
          fontsize=14, fontweight='bold', pad=20)

# Add grid with lower opacity
plt.grid(True, alpha=0.3, linestyle='--')

# Customize legend
plt.legend(title='Validation Sets', title_fontsize=12, fontsize=10, 
          bbox_to_anchor=(1.02, 1), loc='upper left')

# Add note about 'all' samples
plt.figtext(0.99, 0.01, "'all' samples mapped to 250", 
           fontsize=8, style='italic', ha='right')

# Show exact values at start and end points
for idx, (dataset, color) in enumerate(zip(datasets, colors)):
    dataset_data = df_clean[df_clean['evaluation_dataset'] == dataset]
    stats = dataset_data.groupby('num_training_samples')['label_accuracy'].mean()
    stats = stats.sort_index()  # Ensure proper ordering
    
    # Add value labels at start and end
    start_val = stats.iloc[0]
    end_val = stats.iloc[-1]
    plt.annotate(f'{start_val:.3f}', 
                xy=(stats.index[0], start_val),
                xytext=(-10, 10),
                textcoords='offset points',
                color=color,
                fontsize=9)
    plt.annotate(f'{end_val:.3f}', 
                xy=(stats.index[-1], end_val),
                xytext=(10, 10),
                textcoords='offset points',
                color=color,
                fontsize=9)

# Adjust x-axis to show all training sample sizes
plt.xticks(sorted(df_clean['num_training_samples'].unique()))

# Adjust layout to prevent legend cutoff
plt.tight_layout()

plt.savefig('accuracy_trend.pdf', bbox_inches='tight')
plt.close()