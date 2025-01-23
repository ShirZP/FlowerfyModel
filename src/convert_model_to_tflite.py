import tensorflow as tf
model = tf.keras.models.load_model('model/modelv2.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('../model/model.tflite', 'wb') as f:
    f.write(tflite_model)
