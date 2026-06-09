import numpy as np
from scipy.signal import decimate
from load_intan_rhd_format import read_data
import os
import serial
import random
from sklearn.preprocessing import StandardScaler
import time

PORT = "COM5"
BAUD = 115200

DOWNSAMPLE_FACTOR = 50
N_CHANNELS = 56
SEQ_LEN = 16
CHUNK_SIZE = 256
EPOCHS = 10
SUBSAMPLE_RATE = 20

CLASSES = {
    "DORSIFLEXION": 0,
    "PLANTARFLEXION": 1,
    "PRICKING": 2
}

N_CLASSES = len(CLASSES)

def load_and_downsample(filename, num_channels=N_CHANNELS, downsample_factor=DOWNSAMPLE_FACTOR):
    print(f"Loading {filename}")
    result = read_data(filename)
    data = result['amplifier_data'][:num_channels]
    t = result['t_amplifier']

    print("Downsampling")
    downsampled = np.stack([
        decimate(data[i], downsample_factor, ftype="fir", zero_phase=True)
        for i in range(num_channels)
    ])

    t_ds = t[::downsample_factor][:downsampled.shape[1]]
    WINDOW_SIZE = 1
    n_windows = downsampled.shape[1] // WINDOW_SIZE

    downsampled = downsampled[:, :n_windows * WINDOW_SIZE]
    t_ds = t_ds[:n_windows * WINDOW_SIZE]

    windows = downsampled.reshape(num_channels, n_windows, WINDOW_SIZE)
    windows = np.transpose(windows, (1, 0, 2))

    return t_ds, windows

def readline(ser):
    try:
        return ser.readline().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""

def wait_for_answer(ser, window_idx):
    while True:
        line = readline(ser)
        if line.startswith("Probs") and window_idx % 100 == 0:
            print(f"Arduino: {line}")
        if line.startswith("TRAIN"):
            return "ok"
        if not line:
            return "timeout"

def save_model_to_file(ser, filename="trained_weights.h"):
    print("Requesting final weight dump from Arduino.")
    ser.reset_input_buffer()
    ser.write(b'EX')
    ser.flush()

    with open(filename, "w") as f:
        started = False
        while True:
            line = readline(ser)
            if "START_EXPORT" in line:
                started = True
                continue
            if "END_EXPORT" in line:
                break
            if started:
                f.write(line + "\n")
    print(f"Final model saved to {filename}")
    ser.write(b'R')
    ser.flush()

def stream_epoch(ser, epoch_idx, windows, labels, first_send_consumed=False):
    print(f"Starting Epoch {epoch_idx + 1}/{EPOCHS}")
    combined = list(zip(windows, labels))
    random.shuffle(combined)

    for i, (window, label) in enumerate(combined):
        if i == 0 and first_send_consumed:
            pass

        data_bytes = window.astype(np.float32).tobytes()
        for j in range(0, len(data_bytes), CHUNK_SIZE):
            chunk = data_bytes[j:j + CHUNK_SIZE]
            ser.write(chunk)
            ser.flush()
            readline(ser)

        one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
        one_hot[label] = 1
        ser.write(one_hot.tobytes())
        ser.flush()

        wait_for_answer(ser, i)

        if i % 100 == 0:
            print(f"Progress: Window {i}/{len(windows)}")

def prepare_dataset(rms_data, angles_ds):
    rms_data = rms_data[:N_CHANNELS, :]
    X = rms_data.T

    y = np.zeros(len(angles_ds), dtype=np.int64)
    y[angles_ds > 2.0] = 1
    y[angles_ds < -2.0] = 2

    X_seq, y_seq = [], []

    for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
        X_seq.append(X[i - SEQ_LEN:i].T)
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

def load_rat(path):
    data = np.load(path)
    return data["rms_data"], data["a"]

def build_dataset(data_dir, rat_ids):
    X_all, y_all = [], []
    for r in rat_ids:
        path = os.path.join(data_dir, f"rat{r}.npz")
        if os.path.exists(path):
            rms, ang = load_rat(path)
            X, y = prepare_dataset(rms, ang)
            X_all.append(X)
            y_all.append(y)
    return np.concatenate(X_all), np.concatenate(y_all)

if __name__ == "__main__":
    data_dir = "./dataset_rats_50w"
    rat_ids = list(range(4, 10))

    print("Building subsampled dataset...")
    X, y = build_dataset(data_dir, rat_ids)
    print(f"New Dataset Size: {len(X)} windows")

    n_channels = X.shape[1]
    X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
    scaler = StandardScaler()
    X_2d = scaler.fit_transform(X_2d)
    X = X_2d.reshape(X.shape[0], X.shape[2], n_channels).transpose(0, 2, 1)

    np.savez("scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    print("opening serial port...")
    ser = serial.Serial(PORT, BAUD, timeout=3)

    ser.setDTR(True)
    ser.setRTS(True)


    print("Waiting for Arduino USB stack to stabilize...")
    time.sleep(3)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    arduino_ready = False
    print("Sending 'X' handshake to Arduino...")

    for i in range(10):  #
        print(f"Ping {i+1}/10...")
        ser.write(b"X")
        ser.flush()

        time.sleep(0.5)
        response = readline(ser)
        print(f"Arduino responded: '{response}'")

        if "READY" in response:
            arduino_ready = True
            print("Handshake successful! Connection established!")

            ser.reset_input_buffer()
            ser.reset_output_buffer()
            break
        time.sleep(0.5)

    if not arduino_ready:
        print("Arduino Connection Failed.")
        ser.close()
        exit()

    try:
        for epoch in range(EPOCHS):
            stream_epoch(ser, epoch, X, y, first_send_consumed=(epoch == 0))
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    save_model_to_file(ser, "trained_weights.h")
    ser.close()
