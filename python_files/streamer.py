import numpy as np
from scipy.signal import decimate
from load_intan_rhd_format import read_data
import os
import serial
import random
import glob

PORT = "COM3"
BAUD = 500_000
DOWNSAMPLE_FACTOR = 50
N_CHANNELS = 56
WINDOW = 50
CHUNK_SIZE = 256

EPOCHS = 10

CLASSES = {
    "DORSIFLEXION": 0,
    "PLANTARFLEXION": 1,
    "PRICKING": 2
}
N_CLASSES = len(CLASSES)


def load_and_downsample(filename, num_channels=N_CHANNELS, downsample_factor=DOWNSAMPLE_FACTOR):
    print(f'Loading {filename}')
    result = read_data(filename)

    amplifier_data = result['amplifier_data']
    t = result['t_amplifier']

    total_channels = amplifier_data.shape[0]
    if num_channels > total_channels:
        raise ValueError(f'Requested {num_channels} channels but file only has {total_channels}')

    channels = amplifier_data[:num_channels, :]

    print(f'Downsampling by factor {downsample_factor}')
    num_samples_downsampled = len(decimate(channels[0], downsample_factor))
    downsampled = np.zeros((num_channels, num_samples_downsampled))

    for i in range(num_channels):
        downsampled[i] = decimate(channels[i], downsample_factor)

    t_downsampled = t[::downsample_factor][:num_samples_downsampled]

    actual_windows = downsampled.shape[1] // WINDOW

    data = downsampled[:, :actual_windows * WINDOW].reshape(num_channels, actual_windows, WINDOW)
    return t_downsampled, data


def save_model_to_file(ser, filename="trained_weights.h"):
    print(f"Requesting weight dump from Arduino")
    ser.reset_input_buffer()

    ser.write(b'EX')
    ser.flush()

    with open(filename, "w") as f:
        started = False
        while True:
            line = readline(ser)
            if "START_EXPORT" in line:
                started = True
                print("Receiving weights")
                continue
            if "END_EXPORT" in line:
                print(f"Model saved to {filename}")
                break
            if started:
                f.write(line + "\n")

    ser.write(b'R')
    ser.flush()


def get_label(path):
    folder = os.path.basename(os.path.dirname(path)).upper()
    for class_name, idx in CLASSES.items():
        if class_name in folder:
            one_hot = [0] * N_CLASSES
            one_hot[idx] = 1
            return one_hot
    return [0] * N_CLASSES


def readline(ser):
    try:
        return ser.readline().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def wait_for_answer(ser):
    while True:
        line = readline(ser)
        if line and "Probs" in line:
            print(f"Arduino: {line}")
        if line.startswith("TRAIN"):
            return "ok"
        if not line:
            return "timeout"


def stream_window(ser, windows, labels):
    for i, (window, label) in enumerate(zip(windows, labels)):
        trigger = ""
        while trigger != "SEND":
            trigger = readline(ser)

        window = (window - window.mean()) / (window.std() + 1e-8)
        data_bytes = window.tobytes()
        for j in range(0, len(data_bytes), CHUNK_SIZE):
            chunk = data_bytes[j:j + CHUNK_SIZE]
            ser.write(chunk)
            ser.flush()
            ack = readline(ser)
            if ack != "ACK":
                print(f"Window {i}, chunk {j // CHUNK_SIZE}: Expected ACK, got: '{ack}'")
                return

        ser.write(bytes(label))
        ser.flush()
        wait_for_answer(ser)
        if i % 10 == 0:
            print(f"Window {i} done")


def stream_epoch(ser, epoch_idx, windows, labels):
    print(f"Starting Epoch {epoch_idx + 1}/{EPOCHS}")

    combined = list(zip(windows, labels))
    random.shuffle(combined)

    for i, (window, label) in enumerate(combined):
        trigger = ""
        while trigger != "SEND":
            trigger = readline(ser)

        window = (window - window.mean()) / (window.std() + 1e-8)
        data_bytes = window.tobytes()

        for j in range(0, len(data_bytes), CHUNK_SIZE):
            chunk = data_bytes[j:j + CHUNK_SIZE]
            ser.write(chunk)
            ser.flush()
            ack = readline(ser)
            if ack != "ACK":
                print(f"Err Ack: {ack}")
                return

        ser.write(bytes(label))
        ser.flush()
        wait_for_answer(ser)

        if i % 10 == 0:
            print(f"Window {i}/{len(windows)} sent")


if __name__ == '__main__':
    search_pattern = "./data/RAT*_*"
    all_folders = glob.glob(search_pattern)

    files = [
        "./data/RAT4_DORSIFLEXION/dorsi_170306_123322.rhd",
        "./data/RAT4_PLANTARFLEXION/plantar_170306_123936.rhd",
        "./data/RAT4_PRICKING/pricking_170306_125402.rhd",
    ]

    """files = []
    for folder in all_folders:
        folder_name = os.path.basename(folder).upper()
        if any(cls in folder_name for cls in CLASSES.keys()):
            rhd_files = glob.glob(os.path.join(folder, "*.rhd"))
            files.extend(rhd_files)"""

    all_windows = []
    all_labels = []

    for filename in files:
        _, data = load_and_downsample(filename)
        label = get_label(filename)
        num_windows = data.shape[1]
        for frame in range(num_windows):
            all_windows.append(data[:, frame, :].flatten().astype(np.float32))
            all_labels.append(label)

    ser = serial.Serial(PORT, BAUD, timeout=5)

    arduino_answer = ""
    for _ in range(10):
        line = readline(ser)
        if line == "READY":
            arduino_answer = "READY"
            break

    if arduino_answer == "READY":
        ser.write(b"GO\n")
        ser.flush()
        for epoch in range(EPOCHS):
            stream_epoch(ser, epoch, all_windows, all_labels)
            save_model_to_file(ser, "trained_weights.h")
        print("All epochs completed and model saved.")
    else:
        print("Arduino not ready.")
        ser.close()
