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


# Define the path to the directory containing the images for training
train_data = '../dataset/train'

# Create a Pandas DataFrame with a single column
# The column is populated with the list of file/directory names in the 'train_data' directory
pd.DataFrame(
    os.listdir(train_data),
    columns=['File Name']
)


# Define the path to the directory containing the images for validation
val_data ='../dataset/val'

# Create a Pandas DataFrame with a single column
# The column is populated with the list of file/directory names in the 'validation_data' directory
pd.DataFrame(os.listdir(val_data), columns=['File Name'])


# Get a list of the file paths in the 'train_data' directory
train_files = [i for i in glob.glob(train_data + "/*/*")]
train_files = [i.replace("\\", "/") for i in train_files]

# Randomly shuffle the list of file paths
np.random.shuffle(train_files)

# Extract labels from the directory names of each file path
labels = [os.path.dirname(i).split("/")[-1] for i in train_files]

# Combine file paths & its corresponding labels into a list of tuples
data = zip(train_files, labels)

# Create a Pandas DataFrame with 2 columns
# "Path" column contains file paths, & "Label" column contains corresponding labels
training_data = pd.DataFrame(data, columns=["Path", "Label"])

# Display the contents of the DataFrame
print(training_data)


# Get a list of the file paths in the 'validation_data' directory
val_files = [i for i in glob.glob(val_data + "/*/*")]
val_files = [i.replace("\\", "/") for i in val_files]

# Randomly shuffle the list of file paths
np.random.shuffle(val_files)

# Extract labels from the directory names of each file path
labels = [os.path.dirname(i).split("/")[-1] for i in val_files]

# Combine file paths & its corresponding labels into a list of tuples
data = zip(val_files, labels)

# Create a Pandas DataFrame with 2 columns
# "Path" column contains file paths, & "Label" column contains corresponding labels
validation_data = pd.DataFrame(data, columns=["Path", "Label"])

# Display the contents of the DataFrame
print(validation_data)


# Data Visualization

# Create a countplot() using Seaborn, where x-axis represents the "Label" column of the training_data DataFrame
sns.countplot(x = training_data["Label"])

# Rotate x-axis labels for better visibility
plt.xticks(rotation = 45)


# Create a countplot() using Seaborn, where x-axis represents the "Label" column of the validation_data DataFrame
sns.countplot(x = validation_data["Label"])

# Rotate x-axis labels for better visibility
plt.xticks(rotation = 50);


# ----------------------Split Dataset----------------------------

# Create an ImageDataGenerator for data augmentation
image_data_generator = ImageDataGenerator(
    rescale = 1.0 / 255,          # Rescale pixel values to the range [0, 1]
    rotation_range = 20,          # Randomly rotate images in the range of [-20, 20] degrees
    zoom_range = 0.2,             # Randomly zoom into images by up to 20%
    horizontal_flip = True,       # Randomly flip images horizontally
    validation_split = 0.2        # Reserve 20% of the data for validation
)

# Create a training data generator using the flow_from_dataframe method
train_generator = image_data_generator.flow_from_dataframe(
    dataframe = training_data,    # DataFrame containing file paths & labels for training data
    x_col = "Path",               # Column name containing file paths
    y_col = 'Label',              # Column name containing labels
    batch_size = 32,              # No. of samples per batch
    class_mode = "categorical",   # Class mode for one-hot encoded labels
    subset = "training",          # Use the training subset of the data
    target_size = (224, 224)      # Resize images to the specified target size: 224x224
)

# Create a validation data generator using the flow_from_dataframe method
val_generator = image_data_generator.flow_from_dataframe(
    dataframe = validation_data,  # DataFrame containing file paths & labels for validation data
    x_col = "Path",               # Column name containing file paths
    y_col = 'Label',              # Column name containing labels
    batch_size = 32,              # No. of samples per batch
    class_mode = "categorical",   # Class mode for one-hot encoded labels
    subset = "validation",        # Use the validation subset of the data
    target_size = (224, 224)      # Resize images to the specified target size: 224x224
)

# Get the class indices (mapping of the class names to numerical indices) from the training generator
class_indices = train_generator.class_indices

# Display the keys (class names) from the class_indices dictionary
class_indices.keys()


# Initialize an empty list to store class labels
labels = []

# Iterate through the keys (class names) in the class_indices dictionary
for key in class_indices.keys():
    labels.append(key)  # Append each class name to the labels list

# Calculate the total no. of unique labels
total_labels = len(labels)

# Print the list of class labels and the total no. of unique labels
print("Labels: ", labels)
print("Total no. of unique labels:", total_labels)


#---------------???????????---------------------------
"""
# Set the number of rows and columns for the subplot grid
no_of_rows = 2
no_of_columns = 4

# Create a subplot grid with the specified number of rows and columns
fig, axes = plt.subplots(no_of_rows, no_of_columns, figsize=(12, 8))

# Iterate through the rows
for i in range(no_of_rows):
    # Iterate through the columns
    for j in range(no_of_columns):
        # Calculate the index for accessing the data
        index = i * no_of_columns + j

        # Check if the index is within the bounds of the data
        if index < len(training_data):

            # Open the image using the PIL library
            im = Image.open(training_data.iloc[index]['Path'])

            # Convert the PIL image to a NumPy array
            img = np.array(im)

            # Print the shape of the image array
            print(img.shape)

            # Display the image on the subplot at position (i, j)
            axes[i, j].imshow(img)

            # Turn off axis labels for better visualization
            axes[i, j].axis('off')

            # Get the label for the current image and display it as text
            label = training_data.iloc[index]['Label']
            axes[i, j].text(0.5, -0.1, label, ha='center', transform=axes[i, j].transAxes)
# Show the entire subplot grid
plt.show()
"""

#------------------------------Build Model-----------------------------

# Define the input shape for the model
input_shape = (224, 224, 3)

# Path to the manually downloaded InceptionV3 weights file
weights_path = '../model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Check if the weights file exists locally, otherwise download manually
if not os.path.exists(weights_path):
    print(f"Downloading InceptionV3 weights manually to {weights_path}...")
    # You can download the weights from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    print("Please download the weights file manually and place it in the same directory as your script.")
    exit()

# Load the InceptionV3 model with the manually downloaded weights
base_model = InceptionV3(weights=weights_path, include_top=False, input_shape=input_shape)

# Set all layers of the base model to be trainable
for layer in base_model.layers:
    layer.trainable = True

# Create a sequential model
model = models.Sequential()

# Add the InceptionV3 base model to the sequential model
model.add(base_model)

# Add a global average pooling layer to reduce the spatial dimensions of the output
model.add(layers.GlobalAveragePooling2D())

# Add a dense layer with 256 units and ReLU activation function
model.add(layers.Dense(256, activation='relu'))

# Add a dropout layer to prevent overfitting
model.add(layers.Dropout(0.5))

# Add the final dense layer with the number of labels and softmax activation function
total_labels = 14
model.add(layers.Dense(total_labels, activation='softmax'))

# Display the summary of the model architecture
model.summary()


# Create a ModelCheckpoint callback to save the model's weights during training
# The 'save_best_only' option ensures that only the best model (based on validation performance) is saved
checkpoint = ModelCheckpoint("../model/modelv2.keras", save_best_only = True)

# Create an EarlyStopping callback to stop training if the validation performance doesn't improve for a specified number of epochs (patience)
# The 'restore_best_weights' option restores the best weights when training is stopped
early_stopping = EarlyStopping(patience = 5, restore_best_weights = True)


#--------------------------Compile Model---------------------

# Compile the model with the Adam optimizer, categorical crossentropy loss, & accuracy as the evaluation metric
model.compile(optimizer = 'Adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


#------------------------------Train Model---------------------------------

# Train the model using the fit method
hist = model.fit(
    train_generator,                         # Training data generator
    steps_per_epoch = len(train_generator),  # No. of steps (batches) per training epoch
    epochs = 200,                            # No. of training epochs
    validation_data = val_generator,         # Validation data generator
    validation_steps = len(val_generator),   # No. of steps (batches) per validation epoch
    callbacks = [checkpoint, early_stopping] # Callbacks for model checkpointing and early stopping
)

# Load the weights of the trained model from the specified checkpoint file
model = tf.keras.models.load_model("../model/modelv2.keras")

# Create a Pandas DataFrame containing the training history (metrics) of the model
train_history = pd.DataFrame(hist.history)

# Display the DataFrame
print(train_history)

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
