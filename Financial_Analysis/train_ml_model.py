import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
# from tensorflowjs import converters as converters
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('clean_data/bulk_qx_file.csv')
# data = pd.read_csv('clean_data/bulk_fy_file.csv')

# remove if sheet is generated correctly as floating point
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['sector'] = pd.to_numeric(data['sector'], errors='coerce')
data['Profit_Margin'] = pd.to_numeric(data['Profit_Margin'], errors='coerce')
data['Asset_Turnover'] = pd.to_numeric(data['Asset_Turnover'], errors='coerce')
data['Financial_Leverage'] = pd.to_numeric(data['Financial_Leverage'], errors='coerce')
data['ROA'] = pd.to_numeric(data['ROA'], errors='coerce')
data['ROE'] = pd.to_numeric(data['ROE'], errors='coerce')
data['RNOA'] = pd.to_numeric(data['RNOA'], errors='coerce')
data['NOAT'] = pd.to_numeric(data['NOAT'], errors='coerce')
data['NOPM'] = pd.to_numeric(data['NOPM'], errors='coerce')
data['Percent_Growth'] = pd.to_numeric(data['Percent_Growth'], errors='coerce')

# print(data.info())

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

features = ['year', 'sector', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM']
labels = 'Percent_Growth'

# Shuffle the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
X_train, y_train = train_data[features], train_data[labels]
X_test, y_test = test_data[features], test_data[labels]

print('Training Stock Model : Gradient Boosting Regressor')
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print('Gradient Boosting Regressor Generated')

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print('Model Performance')
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

actual_df = pd.DataFrame(y_test)
actual_df.rename(columns={actual_df.columns[0]: 'Actual'}, inplace=True)
actual_entry_statistics = actual_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
actual_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print(actual_entry_statistics)

# Create DataFrame for predictions and actual values
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions.flatten()
})

# Sort by actual values
results_df = results_df.sort_values(by='Actual')

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Actual'], results_df['Predicted'], color='blue', label='Predicted vs Actual')
plt.plot([results_df['Actual'].min(), results_df['Actual'].max()], 
         [results_df['Actual'].min(), results_df['Actual'].max()], 
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted Values vs Actual Values')
plt.legend()
plt.grid(True)
plt.savefig('pics/plot.png')

# Statistical summary of predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print(test_entry_statistics)
print("-" * 70)
