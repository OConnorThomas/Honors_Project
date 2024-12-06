import yfinance as yf
import os
import json
import csv
from tqdm import tqdm

def get_market_sector(symbol):
    """Fetch the market sector for a given stock symbol using Yahoo Finance."""
    try:
        stock_info = yf.Ticker(symbol).info
        return stock_info.get('sector', 'Unknown')  # Default to 'Unknown' if sector not found
    except Exception as e:
        print(f"Error fetching market sector for {symbol}: {e}")
        return 'Unknown'


def gen_unique_sectors_set(data_dir):
    unique_sectors = set()  # Set to store unique sectors

    # Get a list of all directories and files in the data directory
    directories = [(root, files) for root, _, files in os.walk(data_dir)]

    # Outer progress bar for directories
    with tqdm(total=3, desc="Processing Directories", unit="dir") as dir_pbar:
        for i, (root, files) in enumerate(directories):
            if i >= 3:  # Stop after processing the first 3 directories
                break
            # Inner progress bar for files within each directory
            with tqdm(total=len(files), desc=f"Processing Files in {os.path.basename(root)}", leave=False, unit="file") as file_pbar:
                for file in files:
                    file_path = os.path.join(root, file)

                    # Open and read the JSON file
                    with open(file_path, 'r') as f:
                        try:
                            json_data = json.load(f)
                            symbol = json_data.get('symbol')  # Extract the 'symbol' value
                            if symbol:
                                unique_sectors.add(get_market_sector(symbol))
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}")

                    # Update the file progress bar after each file
                    file_pbar.update(1)

            # Update the directory progress bar after all files in the directory have been processed
            dir_pbar.update(1)

    # Return the unique sectors set
    return unique_sectors

if __name__ == '__main__':
    data_directory = 'data'  # Define the path to the data directory
    data = gen_unique_sectors_set(data_directory)
    output_csv_file = 'reference_files/market_sectors.csv'

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for sector in data:
            writer.writerow([sector])
