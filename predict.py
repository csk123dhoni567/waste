import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "saved_model/plant_disease_model.keras"
)

# Load class names
with open("saved_model/classes.json", "r") as f:
    class_names = json.load(f)

# Image path
img_path = input("Enter image path: ")

# Load image
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print("\nPrediction Result")
print("-----------------")
print("Disease:", class_names[predicted_class])
print("Confidence:", round(confidence, 2), "%")