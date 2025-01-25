import glob
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


model = load_model("../model/modelv2.keras")

labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation", "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

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

