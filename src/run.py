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

#--------------------------- Read Dataset ----------------------------

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


#----------------------------- Data Visualization ----------------------------

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


# --------------------------------------------Build Model--------------------------------------------

# Define the input shape for the model
input_shape = (224, 224, 3)

# Path to the manually downloaded InceptionV3 weights file
weights_path = '../model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

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


# ---------------------------------------------Compile Model------------------------------------------------------

# Compile the model with the Adam optimizer, categorical crossentropy loss, & accuracy as the evaluation metric
model.compile(optimizer = 'Adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# ----------------------------------------------Train Model-------------------------------------------------------

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