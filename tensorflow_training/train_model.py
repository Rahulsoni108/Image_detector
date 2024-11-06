import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Set up model saving directory
model_dir = 'app/models'
os.makedirs(model_dir, exist_ok=True)

# Preprocess the dataset with data augmentation
def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

# Load custom dataset
def load_custom_dataset(data_dir):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        label_mode='binary',
        image_size=(128, 128),
        batch_size=32
    ).map(preprocess_image)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        label_mode='binary',
        image_size=(128, 128),
        batch_size=32
    ).map(preprocess_image)

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
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load custom dataset
data_dir = 'tensorflow_training'  # Your custom dataset directory
train_data, test_data = load_custom_dataset(data_dir)

# Create the model
model = create_model()

# Define callbacks
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
tensorboard_callback = TensorBoard(log_dir="logs")

# Train the model
history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
    callbacks=[checkpoint, early_stopping, lr_scheduler, tensorboard_callback]
)

# Save the trained model
model.save(os.path.join(model_dir, 'ai_image_detector.keras'), save_format='keras')
print("Model training complete and saved.")

# Evaluate model accuracy on the test data
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
