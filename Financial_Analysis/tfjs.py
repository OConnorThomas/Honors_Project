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
    "mean": scaler.mean_.tolist(),  # Convert numpy arrays to lists for JSON serialization
    "scale": scaler.scale_.tolist(),
}

# Save as JSON
with open("models/tfjs_model/scaler.json", "w") as f:
    json.dump(scaler_params, f)