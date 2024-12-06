import os
import json
import csv
from json.decoder import JSONDecodeError
from tqdm import tqdm             # progress bars
from datetime import datetime, timedelta
import yfinance as yf

# Function to process a single JSON file (fetch ratio values)
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
    quarter = data['quarter']
    net_income = 0.0
    stockholders_equity = 0.0
    assets = 0.0
    net_sales = 0.0
    ROE = 0.0
    Percent_Growth = 0.0

    # Net Income
    for item in data['data']['ic']:
        if item['concept'] == 'ProfitLoss' or item['concept'] == 'NetIncomeLoss':
            if item['value'] != 'N/A' and item['value'] != '':
                net_income = float(item['value'])
                
                # Ensure item['label'] is not None before checking for substrings
                if item['label'] and ('less' in item['label'].lower() or 'loss' in item['label'].lower()):
                    net_income *= -1
                
                break

    # Stockholders' Equity
    L = 0.0
    L_S = 0.0
    for item in data['data']['bs']:
        if item['concept'] == 'StockholdersEquity' or item['concept'] == 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest':
            if item['value'] != 'N/A' and item['value'] != '':
                stockholders_equity = float(item['value'])
                break
        elif item['concept'] == 'Liabilities': 
            if item['value'] != 'N/A' and item['value'] != '': L = float(item['value'])
        elif item['concept'] == 'LiabilitiesAndStockholdersEquity': 
            if item['value'] != 'N/A' and item['value'] != '': L_S = float(item['value'])
    # if outside loop and nothing found, but alternatives found, substitute
    if stockholders_equity == 0.0 and L != 0.0 and L_S != 0.0:
        stockholders_equity = float(L_S) - float(L)
    
    # Total assets
    found_assets = False
    for category in ['bs', 'ic']:
        for item in data['data'][category]:
            if item['concept'] == 'Assets':
                if item['value'] != 'N/A' and item['value'] != '':
                    assets = float(item['value'])
                    found_assets = True
                    break
        if found_assets: break

    # Net Sales
    # Sales_words = ['SalesRevenueNet', 'SalesRevenueGoodsNet', 'Net sales', 'Sales', 'Revenue',
    #                'SalesRevenueServicesNet', 'Revenues', 'InterestAndDividendIncomeOperating',
    #                'ContractsRevenue', 'OilAndGasRevenue', 'FinancialServicesRevenue',
    #                'RevenueFromContractWithCustomerExcludingAssessedTax']
    Sales_words = ['SalesRevenueNet', 'SalesRevenueGoodsNet', 'Net sales', 'Sales', 'Revenue',
                   'Revenues', 'SalesRevenueServicesNet']
    
    match_found = False
    for item in data['data']['ic']:
        for keyword in Sales_words:
            if item['concept'] == keyword:
                if item['value'] != 'N/A' and item['value'] != '':
                    net_sales = float(item['value'])
                    match_found = True
                    break
        if match_found: break

    # if all fundamental variables are presnet, perform ratio calculations
    if net_sales != 0.0: Profit_Margin = net_income / net_sales
    else: Profit_Margin = 0.0
    if assets != 0.0: Asset_Turnover = net_sales / assets
    else: Asset_Turnover = 0.0
    ROA = Profit_Margin * Asset_Turnover
    if stockholders_equity != 0.0:
        Financial_Leverage = assets / stockholders_equity
        ROE = net_income / stockholders_equity
    else:
        Financial_Leverage = 0.0
        ROE = 0.0

    # target label
    # fetch from file if available
    found = False
    if os.path.exists('reference_files/yf_ref.csv'):
        yf_master_ref = open('reference_files/yf_ref.csv', 'r')
        reader = csv.reader(yf_master_ref)
        item = os.path.basename(file_path)
        for row in reader:
            if row and row[0] == item:
                Percent_Growth = row[1]
                found = True
                break
    # else query yfinance 
    if not found:
        date_obj = datetime.strptime(endDate, '%Y-%m-%d')
        new_date_obj = date_obj + timedelta(days=252)  # Assuming a fiscal year has 252 days
        new_date_string = new_date_obj.strftime('%Y-%m-%d')
        try:
            stock_data = yf.download(symbol, start=date_obj, end=new_date_string, progress=False)
        except Exception as e:
            return f'YFinance Error : {e}', None
        if not stock_data.empty:
            first_close = stock_data.iloc[0]['Close']
            last_close = stock_data.iloc[-1]['Close']
            if first_close == 0: return 'YFinance Error', None
            Percent_Growth = ((last_close - first_close) / first_close) * 100
            with open('reference_files/yf_ref.csv', 'a', newline='') as saveTarget:
                writer = csv.writer(saveTarget)
                writer.writerow([os.path.basename(file_path), Percent_Growth])
        else:
            return 'YFinance Error', None
    
    return 'Success', (quarter, Profit_Margin, Asset_Turnover, Financial_Leverage, ROA, ROE, Percent_Growth)


def main_prog():
    # Parent directory containing subfolders
    parent_dir = 'data/'

    # Output directory for CSV files
    output_dir = 'clean_data/'

    # Iterate through each subfolder
    subdirectories = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    bar = tqdm(total = len(subdirectories) + 1, desc = 'Generating')
    for subdir, _, files in os.walk(parent_dir):

        bar2 = tqdm(total=len(files), desc='Processing', leave=False)
        
        # Generate CSV file name based on subfolder name
        csv_file_name = os.path.basename(subdir) + '.csv'
        csv_file_path = os.path.join(output_dir, csv_file_name)

        # write to database only if the file does not exist
        if not os.path.exists(csv_file_path):

            # List to store extracted data
            data_list = []

            # Process each JSON file in the subfolder
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(subdir, file_name)
                    status, result = process_json_file(file_path)
                    if status == 'Success':
                        quarter, Profit_Margin, Asset_Turnover, Financial_Leverage, ROA, ROE, Percent_Growth = result
                        data_list.append([quarter, float(Profit_Margin), float(Asset_Turnover), float(Financial_Leverage), 
                                        float(ROA), float(ROE), float(Percent_Growth)])
                    bar2.update(1)
            bar2.close()

            # Write to CSV file
            if data_list:
                with open(csv_file_path, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['Q', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'Percent_Growth'])
                    writer.writerows(data_list)
                    # print(f"CSV file '{csv_file_name}' generated successfully.")
        bar.update(1)
    bar.close()

# Aggregate and clean data

def get_csv_files(parent_directory):
    import os
    return [os.path.join(parent_directory, f) for f in os.listdir(parent_directory) if f.endswith('.csv')]

def write_to_master_file(csv_files, output_qx_file, output_fy_file):
    lines = 0
    with open(output_qx_file, 'w', newline='') as qx_file, open(output_fy_file, 'w', newline='') as fy_file:
        qx_writer = csv.writer(qx_file)
        fy_writer = csv.writer(fy_file)
        
        headers = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'Percent_Growth']
        qx_writer.writerow(headers)
        fy_writer.writerow(headers)
        
        for csv_file in csv_files:
            with open(csv_file, 'r') as csv_input:
                reader = csv.reader(csv_input)
                
                header_row = next(reader)
                if 'Q' not in header_row:
                    print(f"Skipping file {csv_file} as it does not contain 'Q' column.")
                    continue
                
                q_index = header_row.index('Q')
                
                for row in reader:
                    if len(row) > q_index:
                        q_value = row[q_index]
                        row = row[:q_index] + row[q_index + 1:]
                        
                        try:
                            float_row = [float(cell) for cell in row]
                        except ValueError:
                            continue

                        if q_value == 'FY':
                            fy_writer.writerow(float_row)
                        else:
                            qx_writer.writerow(float_row)
                        lines += 1
    return lines

def contains_zero(row):
    return any(cell.strip() == '0.0' for cell in row)

def filter_csv(input_file, output_file):
    removed_count = 0
    with open(input_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        all_rows = list(csvreader)
        filtered_data = [row for row in all_rows if not contains_zero(map(str.strip, row))]
        removed_count = len(all_rows) - len(filtered_data)

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(filtered_data)

    return removed_count

# Main program
if __name__ == '__main__':
    parent_directory = 'clean_data'
    output_qx_file = 'clean_data/bulk_qx_file.csv'
    output_fy_file = 'clean_data/bulk_fy_file.csv'

    # scrape the unfiltered json files : clean them into csv
    main_prog()

    csv_files = get_csv_files(parent_directory)
    entries = write_to_master_file(csv_files, output_qx_file, output_fy_file)
    removed_from_qx = filter_csv(output_qx_file, 'clean_data/filtered_qx_file.csv')
    removed_from_fy = filter_csv(output_fy_file, 'clean_data/filtered_fy_file.csv')

    print(f"All entries have been written to bulk_data with {entries} entries.")
    print(f"Entries removed containing zero: {removed_from_qx + removed_from_fy}")
    print(f"Total remaining entries: {entries - (removed_from_fy + removed_from_qx)}")
