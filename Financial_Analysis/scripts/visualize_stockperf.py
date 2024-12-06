import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a pandas DataFrame
file_path = 'clean_data/filtered_qx_file.csv'
df = pd.read_csv(file_path)

# Check if 'Percent_Growth' column exists
if 'Percent_Growth' not in df.columns:
    raise ValueError("The 'Percent_Growth' column is not present in the CSV file.")

# Convert 'Percent_Growth' to float, dropping any NaN values
percent_growth = df['Percent_Growth'].dropna().astype(float)

# Remove the two highest values
percent_growth_updated = percent_growth[percent_growth < percent_growth.nlargest(2).min()]

# Sort the data in ascending order
percent_growth_sorted = percent_growth_updated.sort_values()

# Calculate key statistics
mean_value = percent_growth_sorted.mean()
median_value = percent_growth_sorted.median()
std_dev_value = percent_growth_sorted.std()

# Create an index array for plotting
x_index = range(len(percent_growth_sorted))

# Plot the data using the index array and sorted values
plt.figure(figsize=(10, 6))
plt.plot(x_index, percent_growth_sorted, marker='o', color='b', label='Percent Growth')

# Highlight key points with different colors
plt.axhline(y=mean_value, color='g', linestyle='--', label=f'Mean ({mean_value:.2f})')
plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median ({median_value:.2f})')
plt.axhline(y=std_dev_value, color='m', linestyle='--', label=f'Std Dev ({std_dev_value:.2f})')

# Add labels and title
plt.xlabel('Index (Sorted)')
plt.ylabel('Percent Growth')
plt.title('Percent Growth in Ascending Order with Mean, Median, and Std Dev (Excluding Top 2 Values)')
plt.legend()

# Show the plot
plt.grid()
plt.savefig('pics/global_stock_performance.png')
