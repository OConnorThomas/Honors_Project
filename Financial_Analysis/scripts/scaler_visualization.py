import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Read the data from the CSV file
data = pd.read_csv('clean_data/filtered_qx_file.csv')

# Define features
features = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']

# Extract feature data
X = data[features].values

# Apply different scalers
scalers = {
    'Original': X,
    'StandardScaler': StandardScaler().fit_transform(X),
    'MinMaxScaler': MinMaxScaler().fit_transform(X),
    'RobustScaler': RobustScaler().fit_transform(X)
}

# Create subplots with 2 rows and 4 columns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()

# Plot histograms and box plots
for i, (scaler_name, scaled_data) in enumerate(scalers.items()):
    # Plot histograms
    for j, feature in enumerate(features):
        axes[i * 2].hist(scaled_data[:, j], bins=30, alpha=0.5, label=feature)
    axes[i * 2].set_title(f'{scaler_name} - Histograms')
    axes[i * 2].legend()

    # Plot box plots
    sns.boxplot(data=pd.DataFrame(scaled_data, columns=features), ax=axes[i * 2 + 1])
    axes[i * 2 + 1].set_title(f'{scaler_name} - Box Plots')

plt.tight_layout()
plt.savefig('pics/scaler_visual.png')

# Compute and display summary statistics
for scaler_name, scaled_data in scalers.items():
    df_scaled = pd.DataFrame(scaled_data, columns=features)
    print(f"Summary statistics for {scaler_name}:")
    print(df_scaled.describe(), "\n")
