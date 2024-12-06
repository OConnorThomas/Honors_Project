import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.keras import TqdmCallback
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add

# # ignore all warnings : dump to devnull
# import warnings
# warnings.filterwarnings("ignore")
# sys.stderr = open(os.devnull, 'w')


# from tensorflowjs import converters as converters

# Read the data from the CSV file
# Load the data
# data_qx = pd.read_csv('clean_data/bulk_qx_file.csv')
# data_fy = pd.read_csv('clean_data/bulk_fy_file.csv')

data_qx = pd.read_csv('clean_data/filtered_qx_file.csv')
data_fy = pd.read_csv('clean_data/filtered_fy_file.csv')

# Concatenate row-wise (default for pd.concat)
data = pd.concat([data_qx, data_fy], ignore_index=True)

print(data.head(5))
# data = pd.read_csv('clean_data/filtered_qx_file.csv') # use only entries with non-zero values
# data = pd.read_csv('clean_data/bulk_qx_file.csv') # use entries including zero values

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
label_statistics = data['Percent_Growth'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).transpose()
label_statistics_df = pd.DataFrame(label_statistics).T
label_statistics_df.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count']
print("Statistical Summary of Target Space (Percent_Growth):")
print(label_statistics_df)
print("-" * 80)

# Define features and labels
features = ['sector', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROA', 'ROE', 'RNOA', 'NOAT', 'NOPM']
labels = 'Percent_Growth'

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=2024)
X_train, y_train = train_data[features], train_data[labels]
X_test, y_test = test_data[features], test_data[labels]

# Define scalers and directories
scaler = StandardScaler()

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




###########################
# A bunch of plots #
###########################

import os
image_dir = 'plots'
if not os.path.exists(image_dir):
    os.mkdir(image_dir)


import seaborn as sns
import matplotlib.pyplot as plt

# Combine features and labels for visualization
data_scaled = pd.DataFrame(X_train_scaled, columns=features)

# Bin Percent_Growth into categories for better visualization
data_scaled['Percent_Growth_Category'] = pd.cut(
    y_train.values, 
    bins=[-np.inf, 0, 1000, 2000, 4000, np.inf], 
    labels=['Negative', 'Low', 'Medium', 'High', 'Very High']
)

# Pairplot
sns.pairplot(
    data_scaled, 
    diag_kind="kde", 
    corner=True, 
    hue='Percent_Growth_Category', 
    palette='coolwarm', 
    plot_kws={'alpha': 0.8}  # Adjust transparency for better visibility
)
plt.suptitle("Pairwise Feature Relationships", y=1.02)

# Save the plot
plt.savefig(os.path.join(image_dir, 'pairwise_feature_plot.png'))
plt.close()


# Compute the correlation matrix
correlation_matrix = data[features + [labels]].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Feature and Label Correlation Heatmap")
plt.savefig(os.path.join(image_dir, 'correlation_heatmap.png'))
plt.close()

# Add a categorical plot for sector
plt.figure(figsize=(12, 6))
sns.boxplot(x='sector', y='Percent_Growth', data=data)
plt.title("Distribution of Percent Growth Across Sectors")
plt.xlabel("Sector")
plt.ylabel("Percent Growth")
plt.xticks(rotation=45)
plt.savefig(os.path.join(image_dir, 'sector_vs_pctg.png'))
plt.close()


# KDE plot for each feature
plt.figure(figsize=(14, 8))
for feature in features:
    sns.kdeplot(data[feature], label=feature, fill=True, alpha=0.3)

plt.title("KDE of Feature Distributions")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(image_dir, 'KDE_features_plot.png'))
plt.close()


# Scatterplot for 'ROE' vs. 'Percent_Growth'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='ROE', y='Percent_Growth', data=data, hue='sector', palette='viridis', alpha=0.7)
plt.title("Scatterplot of ROE vs Percent Growth")
plt.xlabel("ROE")
plt.ylabel("Percent Growth")
plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(image_dir, 'roe_vs_pctg.png'))
plt.close()

# Violin plot for each feature per sector
plt.figure(figsize=(12, 8))
sns.violinplot(x='sector', y='ROE', data=data, palette='muted', inner='quartile')
plt.title("Sector-Wise Distribution of ROE")
plt.xlabel("Sector")
plt.ylabel("ROE")
plt.xticks(rotation=45)
plt.savefig(os.path.join(image_dir, 'sector_vs_roe_plot.png'))
plt.close()



# matplotlib for graph creation
import matplotlib.pyplot as plt
import numpy as np

def full_plot(set_plots, labels, file_path, title='Features'):
    """Plot Renyi entropies against time steps. Generates differences plot automatically"""

    plt.figure(figsize=(12, 6))
    colormap = plt.cm.hsv # rainbow colorway
    colors = [colormap(i) for i in np.linspace(0, 1, len(set_plots) + 1)] # rainbow colorway
    for items, label, color in zip(set_plots, labels, colors):
        plt.plot([i for i in range(len(items))], items, linestyle='-', color=color, linewidth=2, markersize=2, label=label)
    plt.title(f'{title} vs Index')
    plt.xlabel('Index')
    plt.ylabel('Features')
    plt.grid(True)
    plt.legend()
    plt.savefig(file_path)
    plt.close()
    return

full_plot(X_train.values.T, features, os.path.join(image_dir, 'feature_distribution.png'), title='Original Features')
full_plot(X_train_scaled.T, features, os.path.join(image_dir, 'scaled_feature_distribution.png'), title='Scaled Features')


def build_model(features):
    model = Sequential([
        Input(shape=(len(features),)),  # Define the input shape using Input layer
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

# use existing model if available
if os.path.exists('models/py_model/model.keras'):
    # Load saved model
    from keras.models import load_model
    model = load_model('models/py_model/model.keras')

    # Use scaler from original training set
    import pickle
    with open('models/py_model/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
# else train new model
else:

    if not os.path.exists('models/py_model'):
        os.mkdir('models/py_model')

    model = build_model(features)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])

    # Train the model with verbose output
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=10000,
        batch_size=512,
        validation_split=0.2,
        verbose=0,  # Turn off Keras' default output
        callbacks=[TqdmCallback(verbose=1)]
    )

    # Extract loss and validation loss from the history object
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 6))
    # plot train_loss vs epochs in blue, circles, connected
    plt.plot(epochs, train_loss, marker='.', linestyle='-', color='b', label='Training Loss')
    # plot val_loss vs epochs in red, circles, connected
    plt.plot(epochs, val_loss, marker='.', linestyle='-', color='r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(image_dir, 'train_val_epochs_plot.png'))
    plt.close()

    # zoom in on recent action
    startpoint = int(len(epochs) * 0.5)
    epochs_second_half = epochs[startpoint:]
    train_loss_second_half = train_loss[startpoint:]
    val_loss_second_half = val_loss[startpoint:]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_second_half, train_loss_second_half, marker='.', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs_second_half, val_loss_second_half, marker='.', linestyle='-', color='r', label='Validation Loss')
    plt.title('Training and Validation Loss (Second Half)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(image_dir, 'train_val_epochs_zoomed.png'))
    plt.close()

    model.save('models/py_model/model.keras')  # Saves the model in updated keras format

    # Save the scaler to a file
    import pickle
    with open('models/py_model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)



    import tensorflowjs as tfjs
    from keras.models import load_model
    import os

    # Define paths
    input_model_path = "models/py_model/model.keras"  # Path to your Keras model
    output_dir = "models/tfjs_model"         # Output directory for the TF.js model

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the Keras model
    print(f"Loading Keras model from: {input_model_path}")
    model = load_model(input_model_path)

    # Convert the model to TensorFlow.js format
    print(f"Converting the model to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, output_dir)

    print(f"Model successfully converted and saved to: {output_dir}")

    import pickle
    import json

    # Load the fitted scaler
    with open("models/py_model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Extract relevant parameters
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    # Save as JSON
    with open("models/tfjs_model/scaler.json", "w") as f:
        json.dump(scaler_params, f)


# Make predictions
predictions = model.predict(X_test_scaled, verbose=0)

# Calculate and print metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Results for NN:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Statistical summary of predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
print(test_entry_statistics)
print("-" * 70)


import shap
import matplotlib.pyplot as plt
import numpy as np

# Create a background dataset for SHAP (using a sample of training data)
background = shap.sample(X_train_scaled, 100)

# Initialize the explainer
explainer = shap.DeepExplainer(model, background)

# Calculate SHAP values for test data
shap_values = explainer.shap_values(X_test_scaled)

# Ensure SHAP values are a numpy array
shap_values = np.array(shap_values)

# Reshape if necessary (for single-output regression models)
if shap_values.ndim > 2:
    shap_values = shap_values.reshape(X_test_scaled.shape[0], -1)

# Plot summary of SHAP values
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=features, plot_type='bar', 
                  show=False, plot_size=(10, 6))
plt.title('Feature Importance via SHAP Values')
plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'shap_summary.png'))
plt.close()

# Plot SHAP summary plot with feature interactions
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=features, 
                  show=False, plot_size=(10, 6))
plt.title('SHAP Summary Plot with Feature Interactions')
plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'shap_interaction_summary.png'))
plt.close()


print('Model Generated and Saved in \'models\' with stats Plotted in \'plots\'')

