import csv
import os

def sort_csv_by_count(csv_file_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        return

    # Read the CSV file into a list of rows
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        data = list(reader)  # Read the rest of the data

    # Sort the data by the 'count' column (index 1) in descending order
    data_sorted = sorted(data, key=lambda row: int(row[1]), reverse=True)

    # Overwrite the CSV file with the sorted data
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write the header
        writer.writerows(data_sorted)  # Write the sorted data

    print(f"File {csv_file_path} has been sorted by count and overwritten.")

if __name__ == '__main__':
    # Specify the path to the CSV file
    csv_file_path = 'clean_data/global_feature_counts.csv'
    
    # Sort and overwrite the CSV file
    sort_csv_by_count(csv_file_path)
