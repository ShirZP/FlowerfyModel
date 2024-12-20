import os
import glob
import cv2                      # OpenCV
from PIL import Image           # Pillow
import numpy as np              # NumPy
import pandas as pd             # Pandas
import matplotlib.pyplot as plt # Matplotlib
import seaborn as sns           # Seaborn
import tensorflow as tf         # TensorFlow

# Keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


# scikit-learn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


# Load the weights of the trained model from the specified checkpoint file
model = tf.keras.models.load_model("../model/model.keras")

# If not already done, reload validation data
val_data = '../dataset/val'

labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation", "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]


# Function to extract the class name from the image path
def extract_class_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

# Load and preprocess the image
img_path = '../dataset/val/water_lily/375534490_c9e9a062f4_c.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Interpret the results
predicted_class = np.argmax(predictions)
predicted_class_label = labels[predicted_class]

# Extract the true class name from the image path
true_class_label = extract_class_name(img_path)

# Display the image
#img = Image.open(img_path)
#plt.imshow(img)
#plt.axis('off')
#plt.show()

# Display the results
print(f"True class (Real Name of flower): {true_class_label}")
print(f"Predicted class (Classified name of flower): {predicted_class_label}")
print(f"Predicted probabilities: {predictions[0]}")

# Get the top predicted classes and their probabilities
top_classes = 3  # Set the number of top classes to display
top_indices = np.argsort(predictions[0])[::-1][:top_classes]

print("\nTop predictions:")
for i in range(top_classes):
    index = top_indices[i]
    label = labels[index]
    probability = predictions[0][index]

    # Format the print statement with the complete decimal and limited percentage
    print(f"{i + 1}: {label} ({probability * 100:.2f}% | {probability:.17f})")