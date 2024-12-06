import pandas as pd
import matplotlib.pyplot as plt
import os

# Create 'pics' directory if it doesn't exist
os.makedirs('pics', exist_ok=True)

# Load the data
data = pd.read_csv('clean_data/filtered_qx_file.csv')

# Ensure numeric conversion
columns = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'Percent_Growth']
for column in columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Define features and label
features = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']
label = 'Percent_Growth'

# Remove top 10 entries farthest from zero for each feature
for feature in features:
    # Calculate absolute values and sort by the farthest from zero
    data = data.reindex(data[feature].abs().sort_values(ascending=True).index)

    # Drop the top 10 farthest entries from zero
    data = data.iloc[100:]

# Calculate absolute values and sort by the farthest from zero
data = data.reindex(data[label].abs().sort_values(ascending=True).index)

# Drop the top 10 farthest entries from zero
data = data.iloc[100:]

# Create scatter plots for each feature against the label
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    plt.scatter(data[feature], data[label], alpha=0.5)
    plt.title(f'{feature} vs {label}')
    plt.xlabel(feature)
    plt.ylabel(label)

plt.tight_layout()
plt.savefig('pics/scatter_plots_less_top100.png')  # Save scatter plots
plt.close()
