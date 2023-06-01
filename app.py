# Import the required libraries
import tensorflow as tf
import cv2
import numpy as np
import os

# Define the directory paths
data_path = "dataset/"
train_path = os.path.join(data_path, "train/")
test_path = os.path.join(data_path, "test/")

# Define the classes
classes = ["mask", "no_mask"]

# Define the image size
img_size = 224

# Load the dataset
def load_dataset():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )

    return train_data, test_data

# Define the model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(2)
    ])
    
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    return model

# Train the model
def train_model(model, train_data, test_data):
    history = model.fit(train_data, epochs=10, validation_data=test_data)
    
    return history

# Load the dataset
train_data, test_data = load_dataset()

# Create the model
model = create_model()

# Compile the model
model = compile_model(model)

# Train the model
history = train_model(model, train_data, test_data)
