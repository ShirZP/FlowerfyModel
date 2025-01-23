import glob
import os
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


model = load_model("../model/modelv2.keras")

labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation", "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

"""
# load validation data
test_data = '../dataset/val'
val_files = [i.replace("\\", "/") for i in glob.glob(test_data + "/*/*")]
np.random.shuffle(val_files)
val_labels = [os.path.dirname(i).split("/")[-1] for i in val_files]
data = zip(val_files, val_labels)
validation_data = pd.DataFrame(data, columns=["Path", "Label"])


# Create an ImageDataGenerator for test data with rescaling
val_image_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,  # Preprocess: scale color values to the range [0, 1]
)

# Create a generator for validation data using a dataframe
val_generator = val_image_data_generator.flow_from_dataframe(
    dataframe=validation_data,
    x_col="Path",             # Column containing file paths
    y_col='Label',            # Column containing class labels
    batch_size=32,            # Batch size for training
    class_mode="categorical", # Type of classification task
    target_size=(224, 224),   # Target size for images
)

# Evaluate the model on the validation data generator
val_score, val_accuracy = model.evaluate(val_generator)

# Display validation loss & accuracy
print('Validation Loss = {:.2%}'.format(val_score), '|', val_score)
print('Validation Accuracy = {:.2%}'.format(val_accuracy), '|', val_accuracy, '\n')

"""

# load test data
test_data = '../dataset/test'
test_files = [i.replace("\\", "/") for i in glob.glob(test_data + "/*/*")]
np.random.shuffle(test_files)
test_labels = [os.path.dirname(i).split("/")[-1] for i in test_files]
data = zip(test_files, test_labels)
testing_data = pd.DataFrame(data, columns=["Path", "Label"])

# Create a test data generator
test_image_data_generator = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_image_data_generator.flow_from_dataframe(
    dataframe=testing_data,
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




"""
#----------------------------Evaluate Model--------------------------

#  Evaluate the model on the validation data generator
validation_score, validation_accuracy = model.evaluate(val_generator)

# Display validation loss & accuracy
print('Validation Loss = {:.2%}'.format(validation_score), '|', validation_score)
print('Validation Accuracy = {:.2%}'.format(validation_accuracy), '|', validation_accuracy, '\n')

# Plot line graphs with training & validation loss on the left, and training & validation accuracy on the right
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_history['loss'],label='Training Loss')
plt.plot(train_history['val_loss'],label='Validation Loss')
plt.title('Training & Validation Loss',fontsize=20)
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_history['accuracy'],label='Training Accuracy')
plt.plot(train_history['val_accuracy'],label='Validation Accuracy')
plt.title('Training & Validation Accuracy',fontsize=20)
plt.legend()


# Create an ImageDataGenerator for test data with rescaling
test_image_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,  # Preprocess: scale color values to the range [0, 1]
)

# Create a generator for test data using a dataframe
test_generator = test_image_data_generator.flow_from_dataframe(
    dataframe=validation_data,
    x_col="Path",             # Column containing file paths
    y_col='Label',            # Column containing class labels
    batch_size=32,            # Batch size for training
    class_mode="categorical", # Type of classification task
    target_size=(224, 224),   # Target size for images
)

# Evaluate the model on the test data generator
test_score, test_accuracy = model.evaluate(test_generator)

# Display test loss & accuracy
print('Test Loss = {:.2%}'.format(test_score), '|', test_score)
print('Test Accuracy = {:.2%}'.format(test_accuracy), '|', test_accuracy, '\n')


# Create a list of tuples representing model evaluation results for validation and test datasets
Accuracy = [('Validation', validation_score, validation_accuracy),
          ('Test', test_score, test_accuracy)
         ]

# Create a DataFrame using the loss & accuracy data of both test & validation
predict_test = pd.DataFrame(data=Accuracy, columns=['Model', 'Loss', 'Accuracy'])
print(predict_test)


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
"""