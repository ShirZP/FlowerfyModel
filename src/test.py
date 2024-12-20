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


# If not already done, reload validation data
val_data = '../dataset/val'
val_files = [i.replace("\\", "/") for i in glob.glob(val_data + "/*/*")]
np.random.shuffle(val_files)
val_labels = [os.path.dirname(i).split("/")[-1] for i in val_files]
data = zip(val_files, val_labels)
validation_data = pd.DataFrame(data, columns=["Path", "Label"])

results = []

# לולאה על כל השורות ב-validation_data
for idx, row in validation_data.iterrows():
    img_path = row["Path"]
    true_label = row["Label"]

    # טעינת התמונה
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0  # נרמול התמונה

    # הפקת תחזיות
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_label = labels[predicted_class]

    # שמירת התוצאה
    results.append({
        'image_path': img_path,
        'true_label': true_label,
        'predicted_label': predicted_class_label
    })



df = pd.DataFrame(results)
df.to_csv("../model/model_results.csv", index=False)

incorrect_predictions = df[df['true_label'] == df['predicted_label']]
incorrect_predictions.to_csv("../model/correct_predictions.csv", index=False)



"""
#-----------------------------Make Predictions (Classify Images)---------------------

# Make a prediction on a single image
def extract_class_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

img_path = ''
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
top_classes = 14  # Set the number of top classes to display
top_indices = np.argsort(predictions[0])[::-1][:top_classes]

print("\nTop predictions:")
for i in range(top_classes):
    index = top_indices[i]
    label = labels[index]
    probability = predictions[0][index]

    # Format the print statement with the complete decimal and limited percentage
    print(f"{i + 1}: {label} ({probability * 100:.2f}% | {probability:.17f})")
"""