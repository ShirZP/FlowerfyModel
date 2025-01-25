import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# רשימת התוויות
labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
          "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

def extract_class_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

# עיבוד התמונה
def preprocess_image(image_path, input_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
    img = img.resize(input_size)  # Resize to the input size
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)  # Normalize and add batch dimension
    return img_array

# טען את המודל TFLite
tflite_model_path = "../model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# קבל מידע על טנסורים
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# עיבוד התמונה
image_path = "../dataset/test/bellflower/14.jpg"  # נתיב לתמונה
img_array = preprocess_image(image_path)  # עיבוד התמונה

# טען את הקלט למודל
interpreter.set_tensor(input_details[0]['index'], img_array)

# הרץ את המודל
interpreter.invoke()

# קבל את הפלט
predictions = interpreter.get_tensor(output_details[0]['index'])  # Get predictions
predicted_class = np.argmax(predictions)  # Find the class with the highest probability
predicted_class_label = labels[predicted_class]  # Map to the corresponding label

# קבלת התווית האמיתית מתוך הנתיב
true_class_label = extract_class_name(image_path)  # True class label

# הצגת התמונה והתוצאות
plt.imshow(Image.open(image_path))  # Display the image
plt.axis('off')  # Remove axis
plt.show()

# הצגת התוצאות בטקסט
print(f"True class: {true_class_label}")  # Print the true class
print(f"Predicted class: {predicted_class_label}")  # Print the predicted class
print(f"Predicted probabilities: {predictions[0]}")  # Print the prediction probabilities
