import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# רשימת התוויות
labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
          "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

# עיבוד התמונה
def preprocess_image(image_path, input_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)  # Normalized and batched
    return img_array

# טען את המודל TFLite
tflite_model_path = "../model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# קבל מידע על טנסורים
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# טען את הקלט למודל
interpreter.set_tensor(input_details[0]['index'], img_array)

# הרץ את המודל
interpreter.invoke()

# קבל את הפלט
predictions = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(predictions)
predicted_class_label = labels[predicted_class]

