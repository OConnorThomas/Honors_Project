# Available methods:

# get_financial_data(stock_symbol): # returns tuple with values for NN
# predict(model, scaler, item): # returns float prediction
# load_and_predict(input_data): # loads model and scaler, fetches yfinance data, returns float prediction
# get_full_finances(item): # returns dict with english description of finances


import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime
import sys

market_sector_words = ['Unknown', 'Consumer Cyclical', 'Technology', 'Financial Services', 'Energy',
                       'Communication Services', 'Basic Materials', 'Consumer Defensive', 'Healthcare',
                       'Industrials', 'Utilities', 'Real Estate']

def get_financial_data(item):
    stock = yf.Ticker(item)

    # Access the balance sheet, cash flow, and income statement data
    try:
        # Balance Sheet (using the most recent data)
        general_info = stock.info
        balance_sheet = stock.balance_sheet
        financials = stock.financials

        ROA = general_info['returnOnAssets'] if 'returnOnAssets' in general_info else 0.0
        ROE = general_info['returnOnEquity'] if 'returnOnEquity' in general_info else 0.0
        sector = general_info['sector'] if 'sector' in general_info else 'Unknown'

        total_sales = general_info['totalRevenue'] if 'totalRevenue' in general_info else 0.0

        net_income = financials.loc['Net Income'].values[0] if 'Net Income' in financials.index else 0.0
        cost_of_goods_sold = financials.loc['Cost Of Revenue'].values[0] if 'Cost Of Revenue' in financials.index else 0.0
        operating_expenses = financials.loc['Operating Expense'].values[0] if 'Operating Expense' in financials.index else 0.0
        net_sales = financials.loc['Net Income'].values[0] if 'Net Income' in financials.index else 0.0

        stockholders_equity = balance_sheet.loc['Stockholders Equity'].values[0] if 'Stockholders Equity' in balance_sheet.index else 0.0
        total_assets = balance_sheet.loc['Total Assets'].values[0] if 'Total Assets' in balance_sheet.index else 0.0
        operating_assets = balance_sheet.loc['Working Capital'].values[0] if 'Working Capital' in balance_sheet.index else 0.0
        operating_liabilities = balance_sheet.loc['Current Liabilities'].values[0] if 'Current Liabilities' in balance_sheet.index else 0.0

        year = int(datetime.now().strftime("%y"))
        sector = market_sector_words.index(sector)

        # Init all ratio calculations to 0.0
        ROE = ROA = Profit_Margin = Asset_Turnover = Financial_Leverage = RNOA = NOPM = NOAT = 0.0

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

        return (int(year), int(sector), float(Profit_Margin), float(Asset_Turnover),
                       float(Financial_Leverage), float(ROA), float(ROE), 
                       float(RNOA), float(NOAT), float(NOPM))


    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_full_finances(item):
    data = list(get_financial_data(item))

    # Normalize the year
    data[0] = data[0] + 2000  # normalize to readable year

    # Fetch the sector string
    data[1] = market_sector_words[data[1]]  # fetch string of sector

    # Add the symbol to the front of the data
    data = [item] + data

    # Format each item in data that is a float to 2 decimal places
    data = [f"{x:.2f}" if isinstance(x, float) else x for x in data]

    # Define the names of the fields
    names = ['Symbol', 'Year', 'Sector', 'Profit Margin', 'Asset Turnover', 'Financial Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM']

    # Return a dictionary mapping names to the formatted data
    return dict(zip(names, data))




def load_and_predict(input_data):
    # Load the pre-trained model
    model = load_model('models/py_model/model.keras')
    # querry yfinance for financials of 
    data = get_financial_data(input_data)
    # remove year column
    data = data[1:]
    # reshape to 2D array for prediction
    data = np.array(data).reshape(1, -1)
    # Scale the input features (ensure that scaling is done with the same scaler as during training)
    import pickle
    with open('models/py_model/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    scaled_input = scaler.transform(data)
    # Make the prediction
    prediction = model.predict(scaled_input, verbose=0)
    
    # Return the predicted value as a single float
    return prediction[0][0]


# base structure

# load saved model
from keras.models import load_model
model = load_model('models/py_model/model.keras')

# Use scaler from original training set
import pickle
with open('models/py_model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict(model, scaler, item):
    # querry yfinance for financials of 
    data = get_financial_data(item)
    if data is not None:
        # remove year column
        data = data[1:]
        # reshape to 2D array for prediction
        data = np.array(data).reshape(1, -1)
        # Scale the input features (ensure that scaling is done with the same scaler as during training)
        scaled_input = scaler.transform(data)
        # Make the prediction
        prediction = model.predict(scaled_input, verbose=0)
        
        # Return the predicted value as a single float
        return prediction[0][0]
    else:
        return None


if __name__ == '__main__':
    # Example usage:
    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1]
    else:
        stock_symbol = "AAPL"
    print(predict(model, scaler, stock_symbol))