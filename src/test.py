import glob
import os
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the weights of the trained model from the specified checkpoint file
model = load_model("../model/modelv2.keras")

labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation", "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

#-----------------------------Make Predictions (Classify Images)---------------------

# Make a prediction on a single image
def extract_class_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

img_path = '../dataset/test/bellflower/14.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0  # Normalize the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
predicted_class_label = labels[predicted_class]
true_class_label = extract_class_name(img_path)


# Display the image and results
plt.imshow(Image.open(img_path))
plt.axis('off')
plt.show()
print(f"True class: {true_class_label}")
print(f"Predicted class: {predicted_class_label}")
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
