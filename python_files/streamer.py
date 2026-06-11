import numpy as np
from scipy.signal import decimate
from utils import read_rhd
import os
import serial
import random
import time

PORT = "COM3"
BAUD = 1_000_000

DOWNSAMPLE_FACTOR = 50
N_CHANNELS = 56
SEQ_LEN = 16
CHUNK_SIZE = 256
EPOCHS = 20
SUBSAMPLE_RATE = 20
RANDOM_SEED = 10

minimum_val_loss = np.inf
val_loss_counter = 0

CLASSES = {
    "NOSIGNAL": 0,
    "DORSIFLEXION": 1,
    "PLANTARFLEXION": 2,
}

N_CLASSES = len(CLASSES)

def load_and_downsample(filename, num_channels=N_CHANNELS, downsample_factor=DOWNSAMPLE_FACTOR):
    print(f"Loading {filename}")
    result = read_rhd(filename)
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

def save_model_to_file(ser, filename="cpp_part/trained_weights.h"):
    print("Requesting weight dump from Arduino.")
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
    print(f"Model saved to {filename}")
    ser.write(b'R')
    ser.flush()

def stream_epoch(ser, epoch_idx, windows, labels, first_send_consumed=False):
    print(f"Starting Epoch {epoch_idx + 1}/{EPOCHS}")
    combined = list(zip(windows, labels))
    random.shuffle(combined)

    for i, (window, label) in enumerate(combined):
        if i == 0 and first_send_consumed:
            pass
        else:
            trigger = ""
            while trigger != "SEND":
                trigger = readline(ser).strip()

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
        path = os.path.join(data_dir, f"RAT{r}.npz")
        if os.path.exists(path):
            rms, ang = load_rat(path)
            X, y = prepare_dataset(rms, ang)
            X_all.append(X)
            y_all.append(y)
    return np.concatenate(X_all), np.concatenate(y_all)

def load_or_build_splits(data_dir, rat_ids, out_dir="./splits", seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"{s}.npz") for s in ("train", "val", "test")}

    if all(os.path.exists(p) for p in paths.values()):
        print("Splits already exist, loading")
        return {s: np.load(p) for s, p in paths.items()}

    print("Building dataset")
    X, y = build_dataset(data_dir, rat_ids)
    print(f"Total windows: {X.shape}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    train_end = int(0.70 * len(idx))
    val_end = int(0.85 * len(idx))

    splits = {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }

    for name, i in splits.items():
        np.savez(paths[name], X=X[i], y=y[i])
        print(f"{name}: {X[i].shape}")

    return {s: np.load(p) for s, p in paths.items()}

def run_validation(ser, X_val, y_val):
    print("Starting validation")

    ser.reset_input_buffer()
    ser.write(b'V')
    ser.flush()
    while readline(ser).strip() != "SEND":
        pass
    first_send_consumed = True

    for i, (window, label) in enumerate(zip(X_val, y_val)):
        if not first_send_consumed:
            while readline(ser).strip() != "SEND":
                pass
        first_send_consumed = False

        data_bytes = window.astype(np.float32).tobytes()
        for j in range(0, len(data_bytes), CHUNK_SIZE):
            ser.write(data_bytes[j:j + CHUNK_SIZE])
            ser.flush()
            readline(ser)

        one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
        one_hot[label] = 1
        ser.write(one_hot.tobytes())
        ser.flush()
        wait_for_answer(ser, i)

        if i % 100 == 0:
            print(f"Validation progress: {i}/{len(X_val)}")

    ser.write(b'D')
    ser.flush()

    deadline = time.time() + 10
    lines = []
    while time.time() < deadline:
        line = readline(ser)
        if line.startswith("VAL_LOSS") or line.startswith("VAL_ACC"):
            print(line)
            lines.append(line)
            with open("results_val.txt", "a") as f:
                f.writelines(line + '\n')
                f.close()
        if line.startswith("VAL_ACC"):
            break

    return lines

def early_stopping(ser, lines):
    global minimum_val_loss
    global val_loss_counter
    for line in lines:
        line = line.strip()
        if "VAL_LOSS" in line:
            val_loss = float(line.split(":")[1])

            if val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
                val_loss_counter = 0
                save_model_to_file(ser, "trained_weights.h")
            else:
                val_loss_counter += 1

if __name__ == "__main__":
    data_dir = "./dataset_rats_50w"
    rat_ids = list(range(4, 10))

    splits = load_or_build_splits(data_dir, rat_ids)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]

    ser = serial.Serial(PORT, BAUD, timeout=5)

    arduino_ready = False
    for _ in range(10):
        if readline(ser) == "READY":
            ser.write(b"GO\n")
            ser.flush()
            arduino_ready = True
            break

    if not arduino_ready:
        print("Arduino Connection Failed.")
        exit()

    while readline(ser).strip() != "SEND":
        pass

    try:
        for epoch in range(EPOCHS):
            stream_epoch(ser, epoch, X_train, y_train, first_send_consumed=(epoch == 0))
            lines = run_validation(ser, X_val, y_val)
            early_stopping(ser, lines)
            if val_loss_counter >= 5:
                print("Early stopping triggered")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    ser.close()
