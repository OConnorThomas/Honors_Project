import os
import json
import csv
from json.decoder import JSONDecodeError
from tqdm import tqdm             # progress bars
from datetime import datetime, timedelta
import yfinance as yf

# Function to process a single JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as json_file:
        try:
            data = json.load(json_file)
        except JSONDecodeError:
            return f"Error decoding JSON in file: {file_path}", None
    
    # Extract required data
    symbol = data['symbol']
    if symbol == 'NONE' or symbol == '': return 'symbol not found', None
    endDate = data['endDate']

    # target label
    date_obj = datetime.strptime(endDate, '%Y-%m-%d')
    new_date_obj = date_obj + timedelta(days=365)  # Assuming a year has 365 days
    new_date_string = new_date_obj.strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start=date_obj, end=new_date_string, progress=False, show_errors=True)
    if stock_data.empty:
        return 'failure', symbol
    else:
        return 'success', None

# Parent directory containing subfolders
parent_dir = 'data/'

# Set to store already written symbols
written_symbols = set()
with open('invalid_symbols.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    written_symbols = set(row[0] for row in reader)

# Iterate through each subfolder
bar = tqdm(total=len(os.listdir(parent_dir)), desc='Generating')
for subdir, _, files in os.walk(parent_dir):
    # Process each JSON file in the subfolder
    bar2 = tqdm(total=len(files), desc='Finding Delsited Symbols', leave=False)
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(subdir, file_name)
            status, result = process_json_file(file_path)
            if status == 'failure' and result not in written_symbols:
                # Write to CSV file
                with open('invalid_symbols.csv', 'a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([result])
                written_symbols.add(result)  # Add symbol to set
        bar2.update(1)
    bar2.close()
    bar.update(1)
bar.close()