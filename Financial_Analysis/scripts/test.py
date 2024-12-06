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

# Generate images for i in the range [0, 100]
for i in range(101):
    # Exclude the i highest values
    if i > 0:
        percent_growth_updated = percent_growth[percent_growth < percent_growth.nlargest(i).min()]
    else:
        percent_growth_updated = percent_growth

    # Calculate key statistics
    mean_value = percent_growth_updated.mean()
    median_value = percent_growth_updated.median()
    std_dev_value = percent_growth_updated.std()

    # Create an index array for plotting
    x_index = range(len(percent_growth_updated))

    # Plot the data using the index array and sorted values
    plt.figure(figsize=(10, 6))
    plt.scatter(x_index, percent_growth_updated, marker='o', color='b', label='Percent Growth')

    # Highlight key points with different colors
    plt.axhline(y=mean_value, color='g', linestyle='--', label=f'Mean ({mean_value:.2f})')
    plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median ({median_value:.2f})')
    plt.axhline(y=std_dev_value, color='m', linestyle='--', label=f'Std Dev ({std_dev_value:.2f})')

    # Add labels and title
    plt.xlabel('Index (Sorted)')
    plt.ylabel('Percent Growth')
    plt.title(f'Percent Growth with Mean, Median, and Std Dev (Excluding Top {i} Values)')
    plt.legend(loc='upper left')

    # Show the plot
    plt.grid()

    # Save the plot as an image file
    plt.savefig(f'pics/temp/global_stock_performance_{i:03}.png')
    plt.close()  # Close the figure to avoid memory issues
