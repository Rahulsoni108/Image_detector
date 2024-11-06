#/home/rails/rails_work/Shriffle_2023/lib/scripts/predict_model.py

import tensorflow as tf
import numpy as np
import sys
import json
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Path to your model
# MODEL_PATH = "app/models/ai_image_detector.h5"
MODEL_PATH = "app/models/best_model.h5"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict(flattened_image_array):
    # Convert to the correct tensor shape (1, 128, 128, 3)
    input_data = np.array(flattened_image_array).reshape([1, 128, 128, 3])

    # Run the prediction and interpret the result
    prediction = model.predict(input_data)
    print(f"Raw model prediction: {prediction}")

    return 'AI-generated' if prediction[0][0] >= 0.5 else 'Real'

# Main script logic
if __name__ == "__main__":
    # Read input JSON from stdin
    flattened_image_array = json.loads(sys.stdin.read())

    # Print the prediction result
    result = predict(flattened_image_array)
    print(result)

