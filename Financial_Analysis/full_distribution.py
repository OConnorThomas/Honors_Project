import pandas as pd
import matplotlib.pyplot as plt

data_qx = pd.read_csv('clean_data/filtered_qx_file.csv')
data_fy = pd.read_csv('clean_data/filtered_fy_file.csv')

# Concatenate row-wise (default for pd.concat)
data = pd.concat([data_qx, data_fy], ignore_index=True)

# Extract the 'result' column and reverse the order
growth_data = data['Percent_Growth'].sort_values(ascending=True).reset_index(drop=True)


# Read the CSV file
data = pd.read_csv('out/results.csv')
# Extract the 'result' column and reverse the order
results = data['result'].sort_values(ascending=True).reset_index(drop=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(growth_data, marker='.', linestyle='-', color='red')
plt.title('Data Distribution of All Stocks')
plt.xlabel('Sorted Index')
plt.ylabel('Predicted Growth %')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig('plots/data_distribution.png')
plt.close()

# Normalize the x-axis for each dataset
growth_x = growth_data.index / len(growth_data)  # Normalize indices for growth_data
results_x = results.index / len(results)        # Normalize indices for results

# Plot both datasets on the same x-axis scale
plt.figure(figsize=(10, 6))
plt.plot(growth_x, growth_data, label='Growth Data', color='r')
plt.plot(results_x, results, label='Results', color='b')
plt.xlabel('Relative Sorted Position in Dataset')
plt.xticks(None)
plt.ylabel('Value')
plt.title('Normalized Comparison of Growth Data and Results')
plt.legend()
plt.grid()
plt.savefig('plots/distribution_comparison.png')
plt.close()