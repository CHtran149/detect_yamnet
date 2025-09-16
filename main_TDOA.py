import pyaudio
import numpy as np
import math

# ===============================
# Cấu hình ReSpeaker 4 Mic Array
# ===============================
CHANNELS = 4
RATE = 16000
CHUNK = 1024
DEVICE_INDEX = 2   # card index (xem bằng "arecord -l")

MIC_DISTANCE = 0.05  # 5 cm giữa các mic
SPEED_SOUND = 343.0  # tốc độ âm thanh (m/s)

# ===============================
# Hàm GCC-PHAT
# ===============================
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R) + 1e-15), n=(interp * nfft))
    max_shift = int(interp * nfft / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

# ===============================
# Hàm tính DOA với 4 mic
# ===============================
def estimate_doa(frame, fs, mic_distance):
    # Mic bố trí: 0 (trái), 1 (trước), 2 (phải), 3 (sau)
    mic0 = frame[:, 0]
    mic1 = frame[:, 1]
    mic2 = frame[:, 2]
    mic3 = frame[:, 3]

    # Tính TDOA trên trục X (mic0 vs mic2) và Y (mic1 vs mic3)
    tau_x = gcc_phat(mic0, mic2, fs, max_tau=mic_distance/SPEED_SOUND)
    tau_y = gcc_phat(mic1, mic3, fs, max_tau=mic_distance/SPEED_SOUND)

    # Đổi TDOA -> góc
    dx = tau_x * SPEED_SOUND
    dy = tau_y * SPEED_SOUND

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

# ===============================
# Main Loop
# ===============================
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("Bắt đầu phát hiện hướng âm thanh 0°–360°... (Ctrl+C để dừng)")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        frame = audio_data.reshape(-1, CHANNELS)

        angle = estimate_doa(frame, RATE, MIC_DISTANCE)
        print(f"Hướng ước lượng: {angle:.2f}°")

except KeyboardInterrupt:
    print("Kết thúc chương trình.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
