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

# Loop through i in range(101) to create plots
for i in range(101):
    # Exclude the i highest values from 'Percent_Growth'
    if i > 0:
        percent_growth_updated = data['Percent_Growth'][data['Percent_Growth'] < data['Percent_Growth'].nlargest(i).min()]
    else:
        percent_growth_updated = data['Percent_Growth']

    # Update data with the filtered Percent_Growth
    filtered_data = data[data['Percent_Growth'].isin(percent_growth_updated)]

    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.tight_layout(pad=5.0)

    # Plot 'Percent_Growth' in the default order (ascending index) as scatter plot
    axs[0, 0].scatter(range(len(filtered_data)), filtered_data['Percent_Growth'].values, color='b', marker='o')
    axs[0, 0].set_title('Percent Growth (Default Order)')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Percent Growth')

    # Plot each of the other metrics using the index for the x-axis as scatter plots
    metrics = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']
    for j, metric in enumerate(metrics):
        ax = axs[(j + 1) // 3, (j + 1) % 3]  # Determine subplot position
        ax.scatter(filtered_data['Percent_Growth'].values, filtered_data[metric].values, color='r', marker='o')
        ax.set_title(metric)
        ax.set_xlabel('Percent Growth')
        ax.set_ylabel(f'{metric}')

    # Hide any empty subplots (if any)
    for j in range(len(metrics) + 1, 6):
        fig.delaxes(axs[j // 3, j % 3])

    # Save the plot as an image file
    plt.savefig(f'pics/temp/metric_outlier_reduction_{i:03}.png')
    plt.close()  # Close the figure to avoid memory issues
