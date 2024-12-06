import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
# from tensorflowjs import converters as converters

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Redirect stdout to log file
log_file = open('logs/log_train_dl_model.txt', 'w')
sys.stdout = log_file

# Read the data from the CSV file
data = pd.read_csv('clean_data/filtered_qx_file.csv')

# Ensure numeric conversion of certain columns
data['Profit_Margin'] = pd.to_numeric(data['Profit_Margin'], errors='coerce')
data['Asset_Turnover'] = pd.to_numeric(data['Asset_Turnover'], errors='coerce')
data['Financial_Leverage'] = pd.to_numeric(data['Financial_Leverage'], errors='coerce')
data['ROA'] = pd.to_numeric(data['ROA'], errors='coerce')
data['ROE'] = pd.to_numeric(data['ROE'], errors='coerce')
data['RNOA'] = pd.to_numeric(data['RNOA'], errors='coerce')
data['NOAT'] = pd.to_numeric(data['NOAT'], errors='coerce')
data['NOPM'] = pd.to_numeric(data['NOPM'], errors='coerce')
data['Percent_Growth'] = pd.to_numeric(data['Percent_Growth'], errors='coerce')

# Statistical summary of labels before splitting the data
label_statistics = data['Percent_Growth'].agg(['mean', 'median', 'std', 'min', 'max']).transpose()
label_statistics_df = pd.DataFrame(label_statistics).T
label_statistics_df.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print("Statistical Summary of Labels (Percent_Growth):")
print(label_statistics_df)
print("-" * 70)

# Define features and labels
features = ['Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM']
labels = 'Percent_Growth'

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=2024)
X_train, y_train = train_data[features], train_data[labels]
X_test, y_test = test_data[features], test_data[labels]

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model with 16 dense layers and dropout layers
def build_deep_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=8))
    
    # Add 16 dense layers with some dropout layers for regularization
    for _ in range(8):
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))  # Dropout for regularization
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    return model

# Build and compile the model
model = build_deep_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
predictions = model.predict(X_test_scaled)

# # Save the model for TensorFlow.js
# os.makedirs('models', exist_ok=True)
# converters.save_keras_model(model, 'models/deep_model')

# Calculate and print metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Statistical summary of predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print("Statistical Summary of Predictions:")
print(test_entry_statistics)
print("-" * 70)

# Restore stdout and close the log file
sys.stdout = sys.__stdout__
log_file.close()
