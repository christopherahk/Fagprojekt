import numpy as np
from scipy.signal import decimate
from load_intan_rhd_format import read_data
import os
import serial
import random
import time
import csv
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

PORT = "COM7"
BAUD = 115200

DOWNSAMPLE_FACTOR = 50
N_CHANNELS = 56
SEQ_LEN = 32
CHUNK_SIZE = 256
EPOCHS = 20
SUBSAMPLE_RATE = 10 # lower to increase data

CLASSES = {
    "DORSIFLEXION": 0,
    "PLANTARFLEXION": 1,
    "PRICKING": 2
}

N_CLASSES = len(CLASSES)
ARDUINO_CLASS_NAMES = ("dorsi", "plantar", "none")
ARDUINO_NAME_TO_LABEL = {name: idx for idx, name in enumerate(ARDUINO_CLASS_NAMES)}

UNLABELED_LABEL = -1
PSEUDO_LABEL_CONFIDENCE = 0.85
SEMI_SUPERVISED_UNLABELED_FRACTION = 0.0
SEMI_SUPERVISED_SEED = 42


@dataclass
class ArduinoResult:
    status: str
    probs: np.ndarray | None = None
    pred_name: str | None = None
    pred_idx: int | None = None

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

def parse_probs(line):
    _, raw_values = line.split(":", 1)
    return np.array([float(value.strip()) for value in raw_values.split(",")], dtype=np.float32)


def parse_pred(line):
    raw = line.split(":", 1)[1].strip()
    name = raw.split("|", 1)[0].strip()
    return name


def wait_for_result(ser, window_idx, timeout_s=1.0):
    start_time = time.monotonic()
    result = ArduinoResult(status="timeout", probs=None, pred_name=None, pred_idx=None)

    while (time.monotonic() - start_time) < timeout_s:
        if ser.in_waiting == 0:
            time.sleep(0.001)
            continue

        line = readline(ser)
        if not line:
            continue

        if "Probs" in line:
            result.probs = parse_probs(line)
            continue
        if "Pred" in line:
            pred_name = parse_pred(line)
            result.pred_name = pred_name
            result.pred_idx = ARDUINO_NAME_TO_LABEL.get(pred_name) if pred_name else None
            continue

        if "TRAIN" in line:
            result.status = "TRAIN"
            return result
        if "INFER" in line:
            result.status = "INFER"
            return result
        if "UNLABELED" in line:
            result.status = "UNLABELED"
            return result
        if "INVALID" in line:
            result.status = "INVALID_LABEL"
            return result

    return result


def send_window(ser, window, label_idx):
    ser.write(b"D")
    ser.flush()

    data_bytes = window.astype(np.float32).tobytes()

    for j in range(0, len(data_bytes), CHUNK_SIZE):
        chunk = data_bytes[j:j + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        time.sleep(0.002)

    if label_idx is None or label_idx < 0:
        ser.write(b"U")
        ser.flush()
        return

    one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
    one_hot[label_idx] = 1
    ser.write(b"L")
    ser.write(one_hot.tobytes())
    ser.flush()

def maybe_enable_semi_supervised(labels, unlabeled_fraction=SEMI_SUPERVISED_UNLABELED_FRACTION, seed=SEMI_SUPERVISED_SEED):
    if unlabeled_fraction <= 0.0:
        return labels

    rng = np.random.default_rng(seed)
    labels = labels.copy()
    mask = rng.random(len(labels)) < unlabeled_fraction
    labels[mask] = UNLABELED_LABEL
    return labels

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


def open_accuracy_log(path="training_accuracy.csv"):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "window_index",
        "label",
        "pred",
        "confidence",
        "running_correct",
        "running_total",
        "running_accuracy",
        "status",
    ])
    return f, writer


def log_accuracy_row(writer, epoch, window_index, label, pred, confidence, running_correct, running_total, status):
    running_accuracy = (running_correct / running_total) if running_total > 0 else 0.0
    writer.writerow([
        epoch,
        window_index,
        label,
        pred,
        f"{confidence:.6f}",
        running_correct,
        running_total,
        f"{running_accuracy:.6f}",
        status,
    ])

# def prepare_dataset(rms_data, angles_ds):
#     rms_data = rms_data[:N_CHANNELS, :]
#     X = rms_data.T

#     y = np.zeros(len(angles_ds), dtype=np.int64)
#     y[angles_ds > 2.0] = 1
#     y[angles_ds < -2.0] = 2

#     X_seq, y_seq = [], []

#     for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
#         X_seq.append(X[i - SEQ_LEN:i].T)
#         y_seq.append(y[i])

#     return np.array(X_seq), np.array(y_seq)

def prepare_dataset(rms_data, angles_ds):
    rms_data = rms_data[:N_CHANNELS, :]
    X = rms_data.T

    y = np.zeros(len(angles_ds), dtype=np.int64)
    y[angles_ds > 2.0]  = 1
    y[angles_ds < -2.0] = 2

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
        window_labels = y[i - SEQ_LEN:i]

        # majority label instead of last
        counts = np.bincount(window_labels, minlength=3)
        majority = int(np.argmax(counts))

        # skip below threshold
        if counts[majority] < SEQ_LEN * 0.6:
            continue

        X_seq.append(X[i - SEQ_LEN:i].T)
        y_seq.append(majority)

    return np.array(X_seq), np.array(y_seq)


def make_semi_supervised(labels, unlabeled_fraction=SEMI_SUPERVISED_UNLABELED_FRACTION, seed=SEMI_SUPERVISED_SEED):
    return maybe_enable_semi_supervised(labels, unlabeled_fraction=unlabeled_fraction, seed=seed)

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

    y = make_semi_supervised(y)
    print(
        f"Supervised labels: {(y >= 0).sum()} | Unlabeled: {(y < 0).sum()} | "
        f"Pseudo-label threshold: {PSEUDO_LABEL_CONFIDENCE:.2f}"
    )

    n_channels = X.shape[1]
    X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
    scaler = RobustScaler() # testing if better than standard scaler
    X_2d = scaler.fit_transform(X_2d)
    X = X_2d.reshape(X.shape[0], X.shape[2], n_channels).transpose(0, 2, 1)

    scaler_mean = np.asarray(scaler.center_)
    scaler_scale = np.asarray(scaler.scale_)
    np.savez("scaler.npz", mean=scaler_mean, scale=scaler_scale)

    accuracy_log_file, accuracy_writer = open_accuracy_log("training_accuracy.csv")

    print("opening serial port...")
    ser = serial.Serial(PORT, BAUD, timeout=3)

    ser.dtr = True
    ser.rts = True

    print("Waiting for Arduino USB stack to stabilize...")
    time.sleep(3)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    arduino_ready = False
    print("Sending 'X' handshake to Arduino...")

    for i in range(10):
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
        running_correct = 0
        running_total = 0


        epoch_pbar = tqdm(range(EPOCHS), desc="Training Process", unit="epoch")
        for epoch in epoch_pbar:
            combined = list(zip(X, y))
            random.shuffle(combined)


            window_pbar = tqdm(
                enumerate(combined),
                total=len(combined),
                desc=f"Epoch {epoch + 1}/{EPOCHS}",
                unit="win",
                leave=False
            )

            for i, (window, label) in window_pbar:
                is_labeled = label is not None and int(label) >= 0
                send_window(ser, window, int(label) if is_labeled else UNLABELED_LABEL)
                result = wait_for_result(ser, i, timeout_s=2.0)

                confidence = float(np.max(result.probs)) if result.probs is not None else 0.0
                pred_name = result.pred_name if result.pred_name is not None else ""
                pred_idx = result.pred_idx if result.pred_idx is not None else -1

                if is_labeled and pred_idx >= 0:
                    running_total += 1
                    if pred_idx == int(label):
                        running_correct += 1

                log_accuracy_row(
                    accuracy_writer,
                    epoch,
                    i,
                    int(label) if is_labeled else UNLABELED_LABEL,
                    pred_idx,
                    confidence,
                    running_correct,
                    running_total,
                    result.status,
                )

                if not is_labeled:
                    if pred_idx >= 0 and confidence >= PSEUDO_LABEL_CONFIDENCE:

                        tqdm.write(f"Pseudo-labeling unlabeled sample as {pred_name} (conf={confidence:.3f})")
                        send_window(ser, window, pred_idx)
                        train_result = wait_for_result(ser, i, timeout_s=2.0)
                        log_accuracy_row(
                            accuracy_writer,
                            epoch,
                            i,
                            UNLABELED_LABEL,
                            pred_idx,
                            confidence,
                            running_correct,
                            running_total,
                            f"pseudo-{train_result.status}",
                        )
                    else:
                        pass


                if running_total > 0:
                    current_acc = (running_correct / running_total) * 100
                    window_pbar.set_postfix({"Acc": f"{current_acc:.2f}%"})
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    accuracy_log_file.flush()
    accuracy_log_file.close()

    save_model_to_file(ser, "trained_weights.h")
    ser.close()
