import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm             # progress bars
from datetime import datetime, timedelta
import numpy as np

def fetch_stock_data(symbol, start_date, end_date, debug = False):
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=debug)
    return stock_data

def generate_pct_diff_label(stock_data, lookback_period=252):
    past_price = stock_data['Close'].shift(-lookback_period)
    first_price = stock_data['Close']
    price_change = ((first_price - past_price) / past_price) * 100
    return price_change

def create_database(output_folder, start_date, end_date, lookback_period=252, debug=False):
    nyse_symbols = pd.read_csv('models/nyse_symbols.csv')
    os.makedirs(output_folder, exist_ok=True)

    bar = tqdm(total = nyse_symbols.size, desc = 'Generating')
    for symbol in nyse_symbols['Symbol']:
        if not os.path.exists(os.path.join('data', f'{symbol}.csv')): 
            try:
                stock_data = fetch_stock_data(symbol, start_date, end_date, debug)
                stock_data = stock_data.reset_index()[['Close']]
                stock_data['Pct Diff 1W'] = generate_pct_diff_label(stock_data, lookback_period=-7)
                stock_data['Pct Diff 2W'] = generate_pct_diff_label(stock_data, lookback_period=-14)
                stock_data['Pct Diff 1M'] = generate_pct_diff_label(stock_data, lookback_period=-21)
                stock_data['Pct Diff 3M'] = generate_pct_diff_label(stock_data, lookback_period=-63)
                stock_data['Pct Diff 6M'] = generate_pct_diff_label(stock_data, lookback_period=-126)
                stock_data['Pct Diff Target'] = generate_pct_diff_label(stock_data, lookback_period)
                # generate more ratios here:
                stock_data = stock_data.iloc[126:]
                output_file = os.path.join(output_folder, f'{symbol}.csv')
                stock_data.to_csv(output_file, index=False, columns=['Close', 'Pct Diff 1W', 'Pct Diff 2W', 'Pct Diff 1M', 'Pct Diff 3M', 'Pct Diff 6M', 'Pct Diff Target'])
                print(f'Downloaded {symbol} stock to {symbol}.csv')

            except Exception as e:
                print(f'Error fetching data for {symbol}: {e}')
                exit()
        bar.update(1)

    bar.close()

if __name__ == "__main__":
    start_date = datetime.now() - pd.to_timedelta(365*41, unit='day')
    end_date = datetime.now()
    output_folder = 'stock_data'

    create_database('data', start_date, end_date, debug=False)
