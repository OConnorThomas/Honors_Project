import os
import json
import csv

def count_unique_symbols(data_dir):
    unique_symbols = set()  # Set to store unique symbols

    # Walk through all subdirectories and files in the data directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Open and read the JSON file
            with open(file_path, 'r') as f:
                try:
                    json_data = json.load(f)
                    symbol = json_data.get('symbol')  # Extract the 'symbol' value
                    if symbol:
                        unique_symbols.add(symbol)  # Add to set of unique symbols
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_path}")

    # Return the count of unique symbols
    return unique_symbols

if __name__ == '__main__':
    data_directory = 'data'  # Define the path to the data directory
    unique_symbol_data = count_unique_symbols(data_directory)
    print(f"Number of unique stocks: {len(unique_symbol_data)}")

    output_csv_file = 'reference_files/unique_stocks.csv'

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for symbol in unique_symbol_data:
            writer.writerow([symbol])
