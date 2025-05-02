import tensorflow as tf

# Set the path to the SavedModel directory
saved_model_dir = 'models/saved_model'

# Create the converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('models/1.tflite', 'wb') as f:
    f.write(tflite_model)