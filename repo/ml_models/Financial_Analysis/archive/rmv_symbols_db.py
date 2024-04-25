import os
import json
import csv
from json.decoder import JSONDecodeError
from tqdm import tqdm             # progress bars

# Function to read symbols from the CSV file
def read_symbols(csv_file):
    symbols = set()
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            symbols.add(row[0])
    return symbols

def process_json_file(file_path):
    with open(file_path, 'r') as json_file:
        try:
            data = json.load(json_file)
        except JSONDecodeError:
            return f"Error decoding JSON in file: {file_path}", None
    
    # Extract required data
    return data['symbol']

# Parent directory containing subfolders
parent_dir = 'data/'
symbols_to_check = read_symbols('invalid_symbols.csv')

total = 0

bar = tqdm(total=len(os.listdir(parent_dir)), desc='Generating')
for subdir, _, files in os.walk(parent_dir):
    # Process each JSON file in the subfolder
    bar2 = tqdm(total=len(files), desc='Purging', leave=False)
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(subdir, file_name)
            symbol = process_json_file(file_path)
            if symbol and symbol in symbols_to_check:
                os.remove(file_path)  # Delete the JSON file
                total = total + 1
        bar2.update(1)
    bar2.close()
    bar.update(1)
bar.close()
print(f'Removed {total} files from db')
