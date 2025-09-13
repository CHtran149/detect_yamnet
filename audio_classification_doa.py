import sys
import time
import numpy as np
from multiprocessing.connection import Client
from mic_array import MicArray

print('Loading TensorFlow...')
import tflite_runtime.interpreter as tflite

print('Loading YAMNet TFLite...')

# Load TFLite model (dùng bản đã convert)
interpreter = tflite.Interpreter(model_path="yamnet_compatible.tflite")
interpreter.allocate_tensors()

# Lấy input và output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class map
def load_class_map(csv_file):
    classes = []
    with open(csv_file, "r") as f:
        for line in f.readlines()[1:]:  # bỏ dòng header
            classes.append(line.strip().split(",")[2])  # cột display_name
    return classes

yamnet_classes = load_class_map("yamnet_class_map.csv")

RATE = 16000
CHANNELS = 4
DOA_FRAMES = 1024    # ms
LOUDNESS_THRESHOLD = DOA_FRAMES * 16000
THRESHOLD = 0.4

def run_yamnet_tflite(waveform):
    """Chạy inference trên model yamnet_compatible.tflite"""
    # Đảm bảo float32
    waveform = np.array(waveform, dtype=np.float32).flatten()

    # Thêm batch dimension => [1, N]
    waveform = np.expand_dims(waveform, axis=0)

    # Đưa vào model
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()

    # Lấy output
    scores = interpreter.get_tensor(output_details[0]['index'])  # (frames, 521)

    # Lấy trung bình theo frames để ổn định hơn
    prediction = np.mean(scores, axis=0)
    return prediction

def main():
    # Interprocessing
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'thanhcolonhue')

    try:
        with MicArray(RATE, CHANNELS, RATE * DOA_FRAMES / 1000) as mic:
            start = time.time()
            lastTimeDirection = 0

            for chunk in mic.read_chunks():
                wav_data = chunk[0::4]  # lấy channel 0
                waveform = wav_data / tf.int16.max  # chuẩn hóa [-1, 1]
                waveform = waveform.astype('float32')

                # Dự đoán loại âm thanh
                prediction = run_yamnet_tflite(waveform)

                # Top-5 kết quả
                top5 = np.argsort(prediction)[::-1][:5]
                print(time.ctime().split()[3],
                      ''.join((f" {prediction[i]:.2f} 👉{yamnet_classes[i][:7].ljust(7, '　')}"
                               if prediction[i] >= THRESHOLD else '...') for i in top5))

                # Phát hiện hướng
                loudness = np.sum(np.abs(chunk))
                if loudness >= LOUDNESS_THRESHOLD:
                    direction = int(mic.get_direction(chunk))
                    print('direction', direction)

                    end = time.time()
                    elapsed_time = end - start
                    angle_difference = np.abs(direction - lastTimeDirection)

                    if elapsed_time >= 2.1 and angle_difference > 10:
                        conn.send(str(direction))
                        start = time.time()
                        lastTimeDirection = direction

    except KeyboardInterrupt:
        pass

    conn.close()


if __name__ == '__main__':
    main()
