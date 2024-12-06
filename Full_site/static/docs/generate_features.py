import os
import json
import csv
from json.decoder import JSONDecodeError
from tqdm import tqdm             # progress bars
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures

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
    quarter = data['quarter']
    year = int(data['year'])

    # Initialize fundamental attributes and calculated ratios
    net_income = stockholders_equity = total_assets = total_sales = net_sales = 0.0
    cost_of_goods_sold = operating_expenses = operating_assets = operating_liabilities = 0.0
    liabilities = liabilities_and_se = 0.0

    # Expanded lists of synonymous concepts
    net_income_words = ['NetIncomeLoss', 'ProfitLoss', 'Net Earnings', 'NetIncome', 'NetProfit', 'NetLoss']
    stockholders_equity_words = ['StockholdersEquity', 'TotalEquity', 'ShareholdersEquity', 'Equity']
    liabilities_words = ['Liabilities', 'TotalLiabilities']
    liabilities_and_se_words = ['LiabilitiesAndStockholdersEquity', 'TotalLiabilitiesAndEquity', 'LiabilitiesAndEquity']
    assets_words = ['Assets', 'TotalAssets']
    net_sales_words = ['SalesRevenueNet', 'SalesRevenueGoodsNet', 'NetSales', 'NetRevenue', 'SalesRevenueServicesNet']
    total_sales_words = ['Sales', 'Revenue', 'Revenues', 'TotalRevenue', 'GrossSales', 'TotalSales']
    cost_of_goods_sold_words = ['CostOfGoodsSold', 'COGS', 'CostOfSales', 'CostOfRevenue']
    operating_expenses_words = ['OperatingExpenses', 'OPEX', 'OperatingCosts', 'OperatingExpense']
    operating_assets_words = ['AssetsCurrent', 'CurrentAssets', 'OperatingAssets']
    operating_liabilities_words = ['OperatingLiabilities', 'LiabilitiesCurrent', 'CurrentLiabilities']

    market_sector_words = ['Unknown', 'Consumer Cyclical', 'Technology', 'Financial Services', 'Energy',
                            'Communication Services', 'Basic Materials', 'Consumer Defensive', 'Healthcare',
                            'Industrials', 'Utilities', 'Real Estate']

    # Dictionary to store current best matches and their priority indices (least to greatest priority)
    result = {
        'net_income': {'value': 0.0, 'priority': float('inf')},
        'stockholders_equity': {'value': 0.0, 'priority': float('inf')},
        'liabilities': {'value': 0.0, 'priority': float('inf')},
        'liabilities_and_se': {'value': 0.0, 'priority': float('inf')},
        'assets': {'value': 0.0, 'priority': float('inf')},
        'net_sales': {'value': 0.0, 'priority': float('inf')},
        'total_sales': {'value': 0.0, 'priority': float('inf')},
        'cost_of_goods_sold': {'value': 0.0, 'priority': float('inf')},
        'operating_expenses': {'value': 0.0, 'priority': float('inf')},
        'operating_assets': {'value': 0.0, 'priority': float('inf')},
        'operating_liabilities': {'value': 0.0, 'priority': float('inf')}
    }

    # Function to update result based on priority
    def update_result(key, value, word_list, concept):
        if concept in word_list:
            priority = word_list.index(concept)
            if priority < result[key]['priority']:
                result[key]['value'] = value
                result[key]['priority'] = priority

    # Iterate through all categories and items
    for category in ['bs', 'ic', 'cf']:
        for item in data['data'][category]:
            concept = item['concept']
            value = item['value']

            # Skip if value is 'N/A' or empty
            if value == 'N/A' or value == '':
                continue

            value = float(value)

            # Check each concept and update based on priority
            update_result('net_income', value, net_income_words, concept)
            update_result('stockholders_equity', value, stockholders_equity_words, concept)
            update_result('liabilities', value, liabilities_words, concept)
            update_result('liabilities_and_se', value, liabilities_and_se_words, concept)
            update_result('assets', value, assets_words, concept)
            update_result('net_sales', value, net_sales_words, concept)
            update_result('total_sales', value, total_sales_words, concept)
            update_result('cost_of_goods_sold', value, cost_of_goods_sold_words, concept)
            update_result('operating_expenses', value, operating_expenses_words, concept)
            update_result('operating_assets', value, operating_assets_words, concept)
            update_result('operating_liabilities', value, operating_liabilities_words, concept)

    # Access the final results
    net_income = result['net_income']['value']
    stockholders_equity = result['stockholders_equity']['value']
    liabilities = result['liabilities']['value']
    liabilities_and_se = result['liabilities_and_se']['value']
    total_assets = result['assets']['value']
    net_sales = result['net_sales']['value']
    total_sales = result['total_sales']['value']
    cost_of_goods_sold = result['cost_of_goods_sold']['value']
    operating_expenses = result['operating_expenses']['value']
    operating_assets = result['operating_assets']['value']
    operating_liabilities = result['operating_liabilities']['value']

    # Calculate Stockholders' Equity if not directly found
    if stockholders_equity == 0.0 and liabilities != 0.0 and liabilities_and_se != 0.0:
        stockholders_equity = liabilities_and_se - liabilities

    # Init all ratio calculations to 0.0
    ROE = ROA = Profit_Margin = Asset_Turnover = Percent_Growth = Financial_Leverage = RNOA = NOPM = NOAT = 0.0

    if net_sales != 0.0: 
        Profit_Margin = net_income / net_sales
    if total_assets != 0.0: 
        Asset_Turnover = net_sales / total_assets
    ROA = Profit_Margin * Asset_Turnover
    if stockholders_equity != 0.0:
        Financial_Leverage = total_assets / stockholders_equity
        ROE = net_income / stockholders_equity

    # Intermediate Attributes
    Net_Operating_Profit_After_Tax = net_sales - cost_of_goods_sold - operating_expenses
    Average_Net_Operating_Assets = operating_assets - operating_liabilities

    if Average_Net_Operating_Assets != 0.0: 
        RNOA = Net_Operating_Profit_After_Tax / Average_Net_Operating_Assets
        NOAT = total_sales / Average_Net_Operating_Assets

    if total_sales != 0.0:
        NOPM = Net_Operating_Profit_After_Tax / total_sales


    # Target label - fetch from file if available
    found = False
    if os.path.exists('reference_files/yf_ref.csv'):
        with open('reference_files/yf_ref.csv', 'r') as yf_master_ref:
            reader = csv.reader(yf_master_ref)
            item = os.path.basename(file_path)
            for row in reader:
                if row and row[0] == item and len(row) > 1:
                    Percent_Growth = row[1]
                    found = True
                    break

    # If not found, query yfinance 
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
            if first_close == 0: 
                return 'YFinance Error', None
            Percent_Growth = ((last_close - first_close) / first_close) * 100
            with open('reference_files/yf_ref.csv', 'a', newline='') as saveTarget:
                writer = csv.writer(saveTarget)
                writer.writerow([item, Percent_Growth])
        else:
            return 'YFinance Error', None
        

    # market sector - fetch from file if available
    found = False
    if os.path.exists('reference_files/yf_ref2.csv'):
        with open('reference_files/yf_ref2.csv', 'r') as yf_master_ref:
            reader = csv.reader(yf_master_ref)
            item = os.path.basename(file_path)
            for row in reader:
                if row and row[0] == item and len(row) > 1:
                    sector = row[1]
                    found = True
                    break

    # If not found, query yfinance 
    if not found:
        try:
            stock_info = yf.Ticker(symbol).info
            sector = stock_info.get('sector', 'Unknown')  # Default to 'Unknown' if sector not found
        except Exception as e:
            return f'YFinance Error : {e}', None
        
        if sector:
            sector = market_sector_words.index(sector)
            with open('reference_files/yf_ref2.csv', 'a', newline='') as saveTarget:
                writer = csv.writer(saveTarget)
                writer.writerow([item, sector])
        else:
            return 'YFinance Error', None

    return 'Success', (quarter, year - 2000, int(sector), float(Profit_Margin), float(Asset_Turnover),
                       float(Financial_Leverage), float(ROA), float(ROE), 
                       float(RNOA), float(NOAT), float(NOPM), float(Percent_Growth))


def main_prog():
    # Parent directory containing subfolders
    parent_dir = 'data/'

    # Output directory for CSV files
    output_dir = 'clean_data/'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of subdirectories
    subdirectories = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    bar = tqdm(total=len(subdirectories), desc='Generating')

    for subdir in subdirectories:
        subdir_path = os.path.join(parent_dir, subdir)

        # Generate CSV file name based on subfolder name
        csv_file_name = os.path.basename(subdir_path) + '.csv'
        csv_file_path = os.path.join(output_dir, csv_file_name)

        # Write to database only if the file does not exist
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Q', 'year', 'sector', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM', 'Percent_Growth'])

                # Get all JSON files in the current subdirectory
                json_files = [file_name for file_name in os.listdir(subdir_path) if file_name.endswith('.json')]
                bar2 = tqdm(total=len(json_files), desc=f'Processing {subdir}', leave=False)

                # Process JSON files concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    futures = {executor.submit(process_json_file, os.path.join(subdir_path, file_name)): file_name for file_name in json_files}
                    for future in concurrent.futures.as_completed(futures):
                        status, result = future.result()
                        if status == 'Success':
                            writer.writerow(result)
                        bar2.update(1)
                bar2.close()

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
        
        headers = ['year', 'sector', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM', 'Percent_Growth']
        qx_writer.writerow(headers)
        fy_writer.writerow(headers)
        
        for csv_file in csv_files:
            with open(csv_file, 'r') as csv_input:
                reader = csv.reader(csv_input)
                
                header_row = next(reader)
                
                q_index = header_row.index('Q')
                
                for row in reader:
                    if len(row) > q_index:
                        q_value = row[q_index]
                        row = row[:q_index] + row[q_index + 1:]
                        
                        try:
                            # Convert the first two items to int and the rest to float
                            float_row = [int(row[0]), int(row[1])] + [float(cell) for cell in row[2:]]
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
