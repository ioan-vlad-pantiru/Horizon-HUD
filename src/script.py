import tensorflow as tf
import os

# Get project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the path to the SavedModel directory
saved_model_dir = os.path.join(PROJECT_ROOT, 'models/saved_model')

# Create the converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(os.path.join(PROJECT_ROOT, 'models/1.tflite'), 'wb') as f:
    f.write(tflite_model)