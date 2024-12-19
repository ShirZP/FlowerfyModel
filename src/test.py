import glob
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Load the weights of the trained model from the specified checkpoint file
model = load_model("../model/model.keras")

# If not already done, reload validation data
val_data = '../dataset/val'
val_files = [i.replace("\\", "/") for i in glob.glob(val_data + "/*/*")]
np.random.shuffle(val_files)
labels = [os.path.dirname(i).split("/")[-1] for i in val_files]
data = zip(val_files, labels)
validation_data = pd.DataFrame(data, columns=["Path", "Label"])

# Create a test data generator
test_image_data_generator = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_image_data_generator.flow_from_dataframe(
    dataframe=validation_data,
    x_col="Path",             # Column containing file paths
    y_col="Label",            # Column containing class labels
    batch_size=32,            # Batch size for testing
    class_mode="categorical", # Type of classification task
    target_size=(224, 224),   # Target size for images
)

# Evaluate the model on the test data
test_score, test_accuracy = model.evaluate(test_generator)
print('Test Loss = {:.2%}'.format(test_score), '|', test_score)
print('Test Accuracy = {:.2%}'.format(test_accuracy), '|', test_accuracy, '\n')


#-----------------------------Make Predictions (Classify Images)---------------------

# Make a prediction on a single image
def extract_class_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

img_path = '../dataset/val/water_lily/375534490_c9e9a062f4_c.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0  # Normalize the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
predicted_class_label = validation_data['Label'].unique()[predicted_class]
true_class_label = extract_class_name(img_path)

# Display the image and results
plt.imshow(Image.open(img_path))
plt.axis('off')
plt.show()
print(f"True class: {true_class_label}")
print(f"Predicted class: {predicted_class_label}")
print(f"Predicted probabilities: {predictions[0]}")


# Get the top predicted classes and their probabilities
top_classes = 14  # Set the number of top classes to display
top_indices = np.argsort(predictions[0])[::-1][:top_classes]

print("\nTop predictions:")
for i in range(top_classes):
    index = top_indices[i]
    label = validation_data['Label'].unique()[index]
    probability = predictions[0][index]

    # Format the print statement with the complete decimal and limited percentage
    print(f"{i + 1}: {label} ({probability * 100:.2f}% | {probability:.17f})")