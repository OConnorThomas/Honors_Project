import os
from tqdm import tqdm             # progress bars
import pandas as pd
import json
from gazpacho import Soup
import requests


nyse_symbols = pd.read_csv('models/nyse_symbols.csv')
os.makedirs('data', exist_ok=True)

bar = tqdm(total = nyse_symbols.size, desc = 'Generating')
for symbol in nyse_symbols['Symbol']:
  if not os.path.exists(os.path.join('data', f'{symbol}.csv')): 
    try:
      url = f"https://roic.ai/financials/{symbol}:US?yearRange=30&valueType=percentage"
      soup = Soup.get(url)
      scrapped_data = soup.find('script', {'id': "__NEXT_DATA__"})
      data = json.loads(scrapped_data.text)
      df = pd.DataFrame(data["props"]["pageProps"]["data"]["data"]["bsq"])
      print(df.head())
    except Exception as e:
      print(f'Error fetching data for {symbol}: {e}')
    bar.update(1)
bar.close()