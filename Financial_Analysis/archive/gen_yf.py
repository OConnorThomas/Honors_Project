import os
import json
import csv
from json.decoder import JSONDecodeError
from tqdm import tqdm  # progress bars
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
    if symbol == 'NONE' or symbol == '':
        return 'symbol not found', None
    endDate = data['endDate']
    Percent_Growth = 0.0

    # target label
    date_obj = datetime.strptime(endDate, '%Y-%m-%d')
    new_date_obj = date_obj + timedelta(days=252) 
    new_date_string = new_date_obj.strftime('%Y-%m-%d')
    try:
        stock_data = yf.download(symbol, start=date_obj, end=new_date_string, progress=False)
    except Exception as e:
        return f'YFinance Error : {e}', None
    if not stock_data.empty:
        first_close = stock_data.iloc[0]['Close']
        last_close = stock_data.iloc[-1]['Close']
        if first_close == 0:
            return 'YFinance Error', None
        Percent_Growth = ((last_close - first_close) / first_close) * 100
    else:
        return 'YFinance Error', None

    return 'Success', float(Percent_Growth)

# Parent directory containing subfolders
parent_dir = 'data/'

# Output directory for CSV files
output_file = 'clean_data/yf_ref.csv'

# List to store extracted data
data_list = []

# Iterate through each subfolder
bar = tqdm(total=len(os.listdir(parent_dir)), desc='Generating', leave=False)
for subdir, _, files in os.walk(parent_dir):
    bar2 = tqdm(total=len(files), desc='Querying', leave=False)
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(subdir, file_name)
            status, result = process_json_file(file_path)
            if status == 'Success':
                data_list.append([file_name, float(result)])
        bar2.update(1)
    bar2.close()
    bar.update(1)
bar.close()

# Write to CSV file
if data_list:
  with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['file_name', 'percent_growth'])
    writer.writerows(data_list)
    print(f"CSV file '{output_file}' generated successfully.")
