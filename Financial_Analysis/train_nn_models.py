import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Add

# from tensorflowjs import converters as converters

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Redirect stdout to log file
log_file = open('logs/log_QX_all_scalers.txt', 'w')
sys.stdout = log_file

# Read the data from the CSV file
data = pd.read_csv('clean_data/filtered_qx_file.csv')

# Ensure numeric conversion of certain columns
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

# Define scalers and directories
scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler()
}

# Define example architectures
def build_model_1():
    return Sequential([
        Dense(256, activation='relu', input_dim=8),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_2():
    return Sequential([
        Dense(512, activation='relu', input_dim=8),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_3():
    return Sequential([
        Dense(1024, activation='relu', input_dim=8),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_4():
    input_layer = Input(shape=(8,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dense(256, activation='relu')(x)
    residual = Dense(256, activation='relu')(input_layer)
    x = Add()([x, residual])
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def build_model_5():
    return Sequential([
        Dense(256, activation='relu', input_dim=8),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_6():
    return Sequential([
        Dense(512, activation='relu', input_dim=8),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_7():
    return Sequential([
        Dense(512, activation='relu', input_dim=8),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_8():
    return Sequential([
        Dense(512, activation='relu', input_dim=8),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_9():
    return Sequential([
        Dense(1024, activation='relu', input_dim=8),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_model_10():
    return Sequential([
        Dense(512, activation='tanh', input_dim=8),
        Dense(256, activation='tanh'),
        Dense(128, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

models = [
    build_model_1(), build_model_2(), build_model_3(), build_model_4(), build_model_5(),
    build_model_6(), build_model_7(), build_model_8(), build_model_9(), build_model_10()
]

model_names = [f'Model_{i}' for i in range(1, 11)]

# Loop over scalers and models
for scaler_name, scaler in scalers.items():
    # Scale the features
    features_train_scaled = scaler.fit_transform(X_train)
    features_test_scaled = scaler.transform(X_test)

    # Create directory for each scaler
    scaler_dir = f'models/QX_{scaler_name}'
    os.makedirs(scaler_dir, exist_ok=True)

    for i, model in enumerate(models):
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
        # Train the model with verbose output
        model.fit(features_train_scaled, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=0)
        
        # Make predictions
        predictions = model.predict(features_test_scaled)
        
        # Save the model for TensorFlow.js
        # converters.save_keras_model(model, f'{scaler_dir}/{model_names[i]}')
        
        # Calculate and print metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Results for {model_names[i]} with {scaler_name}:")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")
        
        # Statistical summary of predictions
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
        test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
        test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
        print(test_entry_statistics)
        print("-" * 70)

# Restore stdout and close the log file
sys.stdout = sys.__stdout__
log_file.close()
