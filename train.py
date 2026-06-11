import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Class Names
class_names = train_ds.class_names

print("Classes Found:")
print(class_names)

# Build Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Create save folder
os.makedirs("saved_model", exist_ok=True)

# Save model
model.save("saved_model/plant_disease_model.keras")

# Save class names
with open("saved_model/classes.json", "w") as f:
    json.dump(class_names, f)

print("================================")
print("Model Saved Successfully")
print("Location: saved_model/")
print("================================")