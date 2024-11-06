import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# # Paths to model files
# model_dir = 'app/models'
# model_path = os.path.join(model_dir, 'ai_image_detector.keras')
# checkpoint_path = os.path.join(model_dir, 'best_model.h5')

# # Remove old model files if they exist
# for path in [model_path, checkpoint_path]:
#     if os.path.exists(path):
#         os.remove(path)
#         print(f"Removed old model file: {path}")

# Set up model saving directory
model_dir = 'app/models'
os.makedirs(model_dir, exist_ok=True)

# Load and preprocess the dataset
def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))  # Resize image to match model input
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # Random contrast
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def load_dataset():
  (train_data, test_data), ds_info = tfds.load(
      'cats_vs_dogs',  # Replace with any dataset name available in tfds
      split=['train[:80%]', 'train[80%:]'],
      as_supervised=True,
      with_info=True,
  )

  train_data = train_data.map(preprocess_image).batch(32).shuffle(1000)
  test_data = test_data.map(preprocess_image).batch(32)

  return train_data, test_data

# Define the model structure
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
train_data, test_data = load_dataset()

# Create the model
model = create_model()

# Define the model checkpoint callback
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'best_model.h5'),  # Save the best model in the specified directory
    monitor='val_accuracy',       # Metric to monitor
    save_best_only=True,          # Save only the best model
    mode='max',                    # Save when the metric is at its maximum
    verbose=1                     # Verbosity mode
)

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data,
    callbacks=[checkpoint]
)

# Save the trained model
model.save(os.path.join(model_dir, 'ai_image_detector.keras'), save_format='keras')
print("Model training complete and saved.")

# Evaluate model accuracy on the test data
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
