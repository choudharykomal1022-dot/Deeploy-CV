import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import os

# Load pre-trained ResNet-50 model
model = ResNet50(weights='imagenet')

# Load an image from your local directory (corrected path)
img_path = ("C:/Users/Komal/Desktop/775px-Viceroy_Butterfly.jpg") # Ensure this is the correct file path

# Validate if the file exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"The file at {img_path} does not exist. Please check the file path.")

# Use TensorFlow to load and preprocess the image
img = tf.io.read_file(img_path)  # Read the image file as a tensor
img = tf.image.decode_image(img, channels=3)  # Decode the image (supports various formats)
img = tf.image.resize(img, [224, 224])  # Resize the image to the target size (224x224)
img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array format

# Create a writable copy of the image array
img_array = np.array(img_array, copy=True)

# Expand dimensions to match the input shape of the model
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode and print top 3 predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}. {label}: {score:.2f}")



