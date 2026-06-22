import numpy as np
import os
import serial
import random
import time
import csv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PORT = "COM8"
BAUD = 1000000
RANDOM_SEED = 10

DOWNSAMPLE_FACTOR = 50
N_CHANNELS        = 56
SEQ_LEN           = 16
CHUNK_SIZE        = 256
EPOCHS            = 1
SUBSAMPLE_RATE    = 1

N_CLASSES = 3
CLASS_NAMES = ("dorsi", "plantar", "none")

minimum_val_loss = np.inf
val_loss_counter = 0

def readline(ser):
    try:
        return ser.readline().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""

def wait_for_send(ser, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = readline(ser)
        if line == "SEND":
            return True
    return False

def wait_for_answer(ser, timeout_s=5.0):
    probs    = None
    pred_idx = -1
    is_correct = -1
    status   = "timeout"
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        line = readline(ser)
        if not line:
            continue
        if line.startswith("Probs:"):
            try:
                base_part = line.split("|")[0] if "|" in line else line
                raw  = base_part.split(":", 1)[1].strip()
                probs = np.array([float(v.strip()) for v in raw.split(",")], dtype=np.float32)

                if "Correct:" in line:
                    is_correct = int(line.split("Correct:")[1].strip())
            except Exception:
                pass
        elif line.startswith("Pred:"):
            try:
                name = line.split(":", 1)[1].strip()
                mapping = {n: i for i, n in enumerate(CLASS_NAMES)}
                pred_idx = mapping.get(name, -1)
            except Exception:
                pass
        elif line == "TRAIN":
            status = "train"
            return probs, pred_idx, is_correct, status
        elif line == "INFER":
            status = "infer"
            return probs, pred_idx, is_correct, status

    return probs, pred_idx, is_correct, status

def send_signal(ser, window):
    data_bytes = window.astype(np.float32).tobytes()
    for j in range(0, len(data_bytes), CHUNK_SIZE):
        chunk = data_bytes[j:j + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        readline(ser)
    time.sleep(0.002)

def prepare_dataset(rms_data, angles_ds):
    rms_data = rms_data[:N_CHANNELS, :]
    X = rms_data.T

    y = np.full(len(angles_ds), 2, dtype=np.int64)
    y[angles_ds >  2.0] = 1
    y[angles_ds < -2.0] = 0

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
        X_seq.append(X[i - SEQ_LEN:i].T)
        y_seq.append(y[i - 1])

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

def load_or_build_splits(data_dir, pretrain_rats, live_rats, out_dir="./splits_6rats", seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"{s}.npz") for s in ("train", "val", "live")}

    if all(os.path.exists(p) for p in paths.values()):
        print("Splits already exist, loading...")
        return {s: np.load(p) for s, p in paths.items()}

    print("Building Pretrain dataset (Rats 4-9)...")
    X_base, y_base = build_dataset(data_dir, pretrain_rats)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_base))
    train_end = int(0.85 * len(idx))

    splits = {
        "train": idx[:train_end],
        "val": idx[train_end:]
    }

    print("Building Live dataset (Rat 10)...")
    X_live, y_live = build_dataset(data_dir, live_rats)

    scaler = StandardScaler()
    n_ch = X_base.shape[1]

    def scale_data(raw_data, fit=False):
        flat = raw_data.transpose(0, 2, 1).reshape(-1, n_ch)
        flat = scaler.fit_transform(flat) if fit else scaler.transform(flat)
        return flat.reshape(raw_data.shape[0], raw_data.shape[2], n_ch).transpose(0, 2, 1)

    X_tr = scale_data(X_base[splits["train"]], fit=True)
    X_va = scale_data(X_base[splits["val"]], fit=False)
    X_li = scale_data(X_live, fit=False)

    np.savez(paths["train"], X=X_tr, y=y_base[splits["train"]])
    np.savez(paths["val"],   X=X_va, y=y_base[splits["val"]])
    np.savez(paths["live"],  X=X_li, y=y_live)

    np.savez("scaler.npz", mean=np.asarray(scaler.mean_), scale=np.asarray(scaler.scale_))

    return {s: np.load(p) for s, p in paths.items()}

def calculate_macro_f1(cm):
    f1s = []
    for i in range(N_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2*p*r / (p+r) if (p+r) > 0 else 0.0)
    return float(np.mean(f1s))

def stream_epoch(ser, epoch_idx, windows, labels, csv_writer):
    print(f"\n--- Epoch {epoch_idx + 1}/{EPOCHS} ---")
    combined = list(zip(windows, labels))
    random.shuffle(combined)

    running_correct = 0
    running_total   = 0
    running_cm      = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    pbar = tqdm(enumerate(combined), total=len(combined),
                desc=f"Epoch {epoch_idx+1}", unit="win", leave=False)

    for i, (window, label) in pbar:
        if not wait_for_send(ser):
            tqdm.write(f"  Timeout waiting for SEND at window {i}")
            continue

        ser.write(b"D")
        ser.flush()
        send_signal(ser, window)

        one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
        one_hot[label] = 1
        ser.write(b"L")
        ser.write(one_hot.tobytes())
        ser.flush()

        probs, pred_idx, is_correct, status = wait_for_answer(ser)
        confidence = float(np.max(probs)) if probs is not None else 0.0

        if pred_idx >= 0:
            running_total += 1
            if pred_idx == int(label):
                running_correct += 1
            running_cm[int(label), pred_idx] += 1

        f1 = calculate_macro_f1(running_cm)
        acc = running_correct / running_total if running_total > 0 else 0.0

        csv_writer.writerow([
            epoch_idx, i, int(label), pred_idx,
            f"{confidence:.4f}", running_correct, running_total,
            f"{acc:.6f}", f"{f1:.6f}", status,
        ])

        if running_total > 0:
            pbar.set_postfix({"Acc": f"{acc*100:.2f}%", "F1": f"{f1:.4f}"})

    final_acc = running_correct / running_total if running_total > 0 else 0.0
    final_f1  = calculate_macro_f1(running_cm)
    print(f"Epoch {epoch_idx+1} final -- Acc: {final_acc*100:.2f}%  F1: {final_f1:.4f}")

def run_validation(ser, X_val, y_val):
    print("\nValidation pass...")
    ser.reset_input_buffer()

    ser.write(b"V")
    ser.flush()

    val_running_correct = 0
    val_running_total = 0
    val_running_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    val_pbar = tqdm(enumerate(zip(X_val, y_val)), total=len(X_val),
                    desc="Validating", unit="win", leave=False)

    for i, (window, label) in val_pbar:
        if not wait_for_send(ser):
            continue

        ser.write(b"D")
        ser.flush()
        send_signal(ser, window)

        one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
        one_hot[label] = 1
        ser.write(b"L")
        ser.write(one_hot.tobytes())
        ser.flush()

        probs, pred_idx, is_correct, status = wait_for_answer(ser)

        if pred_idx >= 0:
            val_running_total += 1
            if pred_idx == int(label):
                val_running_correct += 1
            val_running_cm[int(label), pred_idx] += 1

        val_f1 = calculate_macro_f1(val_running_cm)
        val_acc = val_running_correct / val_running_total if val_running_total > 0 else 0.0

        if val_running_total > 0:
            val_pbar.set_postfix({"Val_Acc": f"{val_acc*100:.2f}%", "Val_F1": f"{val_f1:.4f}"})

    ser.write(b"F")
    ser.flush()

    val_lines = []
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        line = readline(ser)
        if line.startswith("VAL_LOSS") or line.startswith("VAL_ACC"):
            print(f"  [VAL] {line}")
            val_lines.append(line)
            with open("results_val.txt", "a") as f:
                f.write(line + "\n")
        if line.startswith("VAL_ACC"):
            break

    return val_lines

def stream_live(ser, windows, labels, csv_writer):
    print(f"\n--- STARTING LIVE ONLINE LEARNING (Chronological) ---")
    combined = list(zip(windows, labels))

    running_correct = 0
    running_total   = 0
    running_cm      = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    correctness_list = []

    pbar = tqdm(enumerate(combined), total=len(combined), desc="Live Streaming", unit="win")

    for i, (window, label) in pbar:
        if not wait_for_send(ser):
            tqdm.write(f"  Timeout waiting for SEND at window {i}")
            continue

        ser.write(b"D")
        ser.flush()
        send_signal(ser, window)

        one_hot = np.zeros(N_CLASSES, dtype=np.uint8)
        one_hot[label] = 1

        ser.write(b"L")
        ser.write(one_hot.tobytes())
        ser.flush()

        probs, pred_idx, is_correct, status = wait_for_answer(ser)

        if probs is not None and len(probs) == 3:
            weights = np.array([1, 1, 1])
            weighted_probs = probs * weights
            pred_idx = int(np.argmax(weighted_probs))
            confidence = float(np.max(probs))
        else:
            confidence = 0.0

        if pred_idx >= 0:
            running_total += 1
            if pred_idx == int(label):
                running_correct += 1
            running_cm[int(label), pred_idx] += 1

        if is_correct != -1:
            correctness_list.append(is_correct)

        f1 = calculate_macro_f1(running_cm)
        acc = running_correct / running_total if running_total > 0 else 0.0

        csv_writer.writerow([
            "LIVE", i, int(label), pred_idx,
            f"{confidence:.4f}", running_correct, running_total,
            f"{acc:.6f}", f"{f1:.6f}", status,
        ])

    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "model_MF_correctness.txt"), np.array(correctness_list, dtype=int), fmt="%d")

    final_acc = running_correct / running_total if running_total > 0 else 0.0
    final_f1  = calculate_macro_f1(running_cm)
    print(f"\nLive Streaming Finished -- Final Acc: {final_acc*100:.2f}%  Final F1: {final_f1:.4f}")

if __name__ == "__main__":
    data_dir = "./dataset_rats_50w"

    PRETRAIN_RATS = [9]
    LIVE_RAT      = [10]

    splits   = load_or_build_splits(data_dir, PRETRAIN_RATS, LIVE_RAT)
    X_train  = splits["train"]["X"]
    y_train  = splits["train"]["y"]
    X_val    = splits["val"]["X"]
    y_val    = splits["val"]["y"]
    X_live   = splits["live"]["X"]
    y_live   = splits["live"]["y"]

    f_csv   = open("training_accuracy.csv", "w", newline="", encoding="utf-8")
    writer  = csv.writer(f_csv)
    writer.writerow(["epoch", "window_index", "label", "pred", "confidence",
                     "running_correct", "running_total",
                     "running_accuracy", "running_f1", "status"])

    print("Opening serial port...")
    ser = serial.Serial(PORT, BAUD, timeout=5)

    arduino_ready = False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        line = readline(ser)
        if line == "READY":
            ser.write(b"G\n")
            ser.flush()
            arduino_ready = True
            print("Handshake complete!")
            break
        time.sleep(0.1)

    if not arduino_ready:
        print("Arduino not responding.")
        f_csv.close(); ser.close(); exit()

    time.sleep(0.5)
    ser.reset_input_buffer()

    try:
        print("\n=== PHASE 2: LIVE ONLINE LEARNING ON RAT 10 ===")
        stream_live(ser, X_live, y_live, writer)
        f_csv.flush()
        val_lines = run_validation(ser, X_val, y_val)
        for line in val_lines:
            print(f"  --> {line}")
    except KeyboardInterrupt:
        print("\nInterrupted.")

    f_csv.close()
    ser.close()
    print("Done.")
