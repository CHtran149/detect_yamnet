import tensorflow as tf

# Đường dẫn đến file mô hình .h5
h5_model_path = "yamnet.h5"

# Đường dẫn lưu file .tflite sau khi convert
tflite_model_path = "yamnet.tflite"

# Load mô hình Keras (.h5)
model = tf.keras.models.load_model(h5_model_path)

# Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Ghi ra file .tflite
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Đã convert xong: {tflite_model_path}")
