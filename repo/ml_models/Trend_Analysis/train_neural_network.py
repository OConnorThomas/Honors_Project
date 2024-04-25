import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Read the data from the CSV file
data = pd.concat([pd.read_csv(os.path.join('data', file)).iloc[:-252] for file in os.listdir('data') if file.endswith('.csv')])
print(f'{len(os.listdir("data"))} total stocks')
print(f'{len(data):,} total entries')

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Pct Diff 1W'] = pd.to_numeric(data['Pct Diff 1W'], errors='coerce')
data['Pct Diff 2W'] = pd.to_numeric(data['Pct Diff 2W'], errors='coerce')
data['Pct Diff 1M'] = pd.to_numeric(data['Pct Diff 1M'], errors='coerce')
data['Pct Diff 3M'] = pd.to_numeric(data['Pct Diff 3M'], errors='coerce')
data['Pct Diff 6M'] = pd.to_numeric(data['Pct Diff 6M'], errors='coerce')
data['Pct Diff Target'] = pd.to_numeric(data['Pct Diff Target'], errors='coerce')

features = ['Close', 'Pct Diff 1W', 'Pct Diff 2W', 'Pct Diff 1M', 'Pct Diff 3M', 'Pct Diff 6M']
target = 'Pct Diff Target'

# Shuffle the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# Standardize the features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(X_train)
features_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential([
  Dense(128, activation='relu'),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(features_train_scaled, y_train, epochs=50, batch_size=1024, validation_split=0.2)

predictions = model.predict(features_test_scaled)

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