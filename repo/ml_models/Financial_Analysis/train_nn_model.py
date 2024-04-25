import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflowjs as tfjs

# Read the data from the CSV file
data = pd.read_csv('clean_data/filtered_qx_file.csv')

# remove if sheet is generated correctly as floating point
data['Profit_Margin'] = pd.to_numeric(data['Profit_Margin'], errors='coerce')
data['Asset_Turnover'] = pd.to_numeric(data['Asset_Turnover'], errors='coerce')
data['Financial_Leverage'] = pd.to_numeric(data['Financial_Leverage'], errors='coerce')
data['ROA'] = pd.to_numeric(data['ROA'], errors='coerce')
data['ROE'] = pd.to_numeric(data['ROE'], errors='coerce')
data['Percent_Growth'] = pd.to_numeric(data['Percent_Growth'], errors='coerce')

# define ratios / features
features = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE']
labels = 'Percent_Growth'

# Shuffle the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
X_train, y_train = train_data[features], train_data[labels]
X_test, y_test = test_data[features], test_data[labels]

# Standardize the features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(X_train)
features_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential([
  Dense(5, activation='relu'),
  Dense(10, activation='relu'),
  Dense(5, activation='relu'),
  Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(features_train_scaled, y_train, epochs=20, batch_size=16, validation_split=0.2)

predictions = model.predict(features_test_scaled)

tfjs.converters.save_keras_model(model, 'models/nnModel')

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print(test_entry_statistics)