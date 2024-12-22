import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# Class mapping for flower classification
class_mapping_flower = {
    0: 'Astilbe',
    1: 'Bellflower',
    2: 'Black-Eyed Susan',
    3: 'Calendula',
    4: 'California Poppy',
    5: 'Carnation',
    6: 'Common Daisy',
    7: 'Coreopsis',
    8: 'Dandelion',
    9: 'Iris',
    10: 'Rose',
    11: 'Sunflower',
    12: 'Tulip',
    13: 'Water Lily',
}

# Function to load the flower classification model
@st.cache(allow_output_mutation=True)
def load_flower_model():
    # Local path to save the downloaded model file
    local_model_path = './modelv2.keras'  # You can adjust the path as needed

    """
    # Google Drive direct link to the shared model file
    drive_url = 'https://drive.google.com/drive/folders/1aTpSSn11zzGbMZMWbixy1tKJJng6eU0P?usp=sharing'
    
    # Download the model file using gdown
    response = gdown.download(drive_url, output=local_model_path, quiet=False)
    """

    # Load the entire model from the .keras file
    model = tf.keras.models.load_model(local_model_path)

    return model

# Function to preprocess and make predictions for flower classification
def predict_flower(image, model):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize images to the specified target size: 224x224
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label from the mapping
    predicted_class_label = class_mapping_flower.get(predicted_class_index, 'Unknown')

    return predicted_class_label

# Streamlit app
st.title('Flower Image Classification')
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the flower classification model
    flower_model = load_flower_model()

    # Make predictions for flower classification
    predicted_class_flower = predict_flower(image, flower_model)
    st.write(f"Prediction: {predicted_class_flower}")
