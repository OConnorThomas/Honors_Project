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

y_min = -100
y_max = percent_growth.max() * 1.1

# Generate images for i in the range [0, 100]
for i in range(101):
    plt.figure(figsize=(15, 12))  # Create a 2x2 figure with more space

    # Exclude the i highest values
    if i > 0:
        percent_growth_updated = percent_growth[percent_growth < percent_growth.nlargest(i).min()]
    else:
        percent_growth_updated = percent_growth

    # Sort the data for plotting purposes
    sorted_data = percent_growth_updated.sort_values()
    unsorted_data = percent_growth_updated

    # Define fixed y-axis limits, reset every 5 iterations
    if i % 20 == 0:
        y_max = percent_growth_updated.max() * 1.1

    # Calculate the maximum value for constant dimensions
    max_value = sorted_data.max() if not sorted_data.empty else 0

    # Create an index array for plotting
    x_index_sorted = range(len(sorted_data))
    x_index_unsorted = range(len(unsorted_data))

    # Plot the data sorted by value (Upper Left) as a scatter plot
    plt.subplot(2, 2, 1)
    plt.plot(x_index_sorted, sorted_data, color='g', marker='o', linestyle='-', label='Percent Growth')
    plt.axhline(y=sorted_data.mean(), color='g', linestyle='--', label=f'Mean ({sorted_data.mean():.2f})')
    plt.axhline(y=sorted_data.median(), color='r', linestyle='--', label=f'Median ({sorted_data.median():.2f})')
    plt.axhline(y=sorted_data.std(), color='m', linestyle='--', label=f'Std Dev ({sorted_data.std():.2f})')
    plt.xlabel('Index (Sorted)')
    plt.ylabel('Percent Growth')
    plt.title(f'Ascending Order with Mean, Median, and Std Dev (Excluding Top {i} Values)')
    plt.legend(loc='upper left')
    plt.grid()

    # Plot the data unsorted by value (Upper Right) as a scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(x_index_unsorted, unsorted_data, color='darkgreen', marker='o', label='Percent Growth')
    plt.axhline(y=unsorted_data.mean(), color='g', linestyle='--', label=f'Mean ({unsorted_data.mean():.2f})')
    plt.axhline(y=unsorted_data.median(), color='r', linestyle='--', label=f'Median ({unsorted_data.median():.2f})')
    plt.axhline(y=unsorted_data.std(), color='m', linestyle='--', label=f'Std Dev ({unsorted_data.std():.2f})')
    plt.xlabel('Index (Unsorted)')
    plt.ylabel('Percent Growth')
    plt.title(f'Unsorted Data with Mean, Median, and Std Dev (Excluding Top {i} Values)')
    plt.legend(loc='upper left')
    plt.grid()

    # Plot the data sorted by value with fixed dimensions and max line (Bottom Left) as a scatter plot
    plt.subplot(2, 2, 3)
    plt.plot(x_index_sorted, sorted_data, color='g', marker='o', linestyle='-', label='Percent Growth')
    plt.axhline(y=max_value, color='r', linestyle='--', label=f'Max Value ({max_value:.2f})')
    plt.xlabel('Index (Sorted)')
    plt.ylabel('Percent Growth')
    plt.title(f'Sorted Data with Max Line (Excluding Top {i} Values)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.ylim(y_min, y_max)  # Set static y-axis limits

    # Plot the data unsorted by value with fixed dimensions and max line (Bottom Right) as a scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(x_index_unsorted, unsorted_data, color='darkgreen', marker='o', label='Percent Growth')
    plt.axhline(y=max_value, color='r', linestyle='--', label=f'Max Value ({max_value:.2f})')
    plt.xlabel('Index (Unsorted)')
    plt.ylabel('Percent Growth')
    plt.title(f'Unsorted Data with Max Line (Excluding Top {i} Values)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.ylim(y_min, y_max)  # Set static y-axis limits

    # Save the plot as an image file
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(f'pics/temp/global_stock_performance_{i:03}.png')
    plt.close()  # Close the figure to avoid memory issues
