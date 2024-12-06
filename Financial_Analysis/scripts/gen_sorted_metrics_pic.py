import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = 'clean_data/filtered_qx_file.csv'
df = pd.read_csv(file_path)

# Check if the required columns exist
required_columns = ['Percent_Growth', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"The '{col}' column is not present in the CSV file.")

# Extract the relevant columns
data = df[required_columns].dropna()

# Sort the data by 'Percent_Growth' in ascending order
data = data.sort_values(by='Percent_Growth')

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.tight_layout(pad=5.0)

# Plot 'Percent_Growth' in ascending order
axs[0, 0].plot(data['Percent_Growth'].values, marker='o', linestyle='-', color='b')
axs[0, 0].set_title('Percent Growth (Sorted)')
axs[0, 0].set_xlabel('Index')
axs[0, 0].set_ylabel('Percent Growth (Sorted)')

# Plot each of the other metrics based on the sorted 'Percent_Growth'
metrics = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']
for i, metric in enumerate(metrics):
    ax = axs[(i + 1) // 3, (i + 1) % 3]  # Determine subplot position
    ax.scatter(data.index, data[metric].values, color='r', marker='o')
    ax.set_title(metric)
    ax.set_xlabel('Index')
    ax.set_ylabel(f'{metric} (sorted by Percent Growth)')

# Hide any empty subplots (if any)
for j in range(len(metrics) + 1, 6):
    fig.delaxes(axs[j // 3, j % 3])

# Save the plot as an image file
plt.savefig('pics/metrics_pct_growth_sorted.png')

