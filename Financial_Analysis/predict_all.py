from query import predict
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load saved model
from keras.models import load_model
model = load_model('models/py_model/model.keras')

# Use scaler from original training set
import pickle
with open('models/py_model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the NYSE symbols
data = pd.read_csv('reference_files/nyse_symbols.csv')

import os
out_dir = 'out'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Output file
output_file = os.path.join(out_dir, 'results.csv')

# Load existing results and create a set of symbols
try:
    existing_results = pd.read_csv(output_file)
    existing_symbols = set(existing_results['symbol'])  # Create a set with just the symbols
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty set
    existing_symbols = set()

# Open the CSV file in append mode for new results
with open(output_file, 'a') as f:
    # If the file is empty, write the header
    if f.tell() == 0:
        f.write('symbol,result\n')

    # Iterate through symbols and generate predictions
    for item in tqdm(data['Symbol'], desc='Generating predictions'):
        # Check if the symbol is not already in the results
        if item not in existing_symbols:
            result = predict(model, scaler, item)
            print(f'Stock {item} predicted {result}')

            if result is not None:
                f.write(f'{item},{result}\n')

            # Add the symbol to the set of existing symbols
            existing_symbols.add(item)

print(f"Predictions saved to {output_file}")

# Load the results again
results = pd.read_csv(output_file)

# Sort the DataFrame by the 'result' column in ascending order
sorted_results = results.sort_values(by='result', ascending=False)

# Overwrite the original file with the sorted DataFrame
sorted_results.to_csv(output_file, index=False)

print(f"Results file sorted by 'result' in ascending order and saved to {output_file}")

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv(output_file)

# Extract the 'result' column and reverse the order
results = data['result'].sort_values(ascending=True).reset_index(drop=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(results, marker='.', linestyle='-', color='blue')
plt.title('Model Prediction Distribution of NYSE Stocks')
plt.xlabel('Sorted Index')
plt.ylabel('Predicted Growth %')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig('plots/prediction_distribution.png')
plt.close()