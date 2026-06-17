import numpy as np
import os
import serial
import random
import time
import csv
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from tqdm import tqdm

PORT = "COM7"
BAUD = 1000000

DOWNSAMPLE_FACTOR = 50
N_CHANNELS        = 56
SEQ_LEN           = 16 # from 32
CHUNK_SIZE        = 256
EPOCHS            = 1 # mondrian is a one-pass
SUBSAMPLE_RATE    = 8 # from 10
RANDOM_SEED       = 10

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
    status   = "timeout"
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        line = readline(ser)
        if not line:
            continue
        if line.startswith("Probs:"):
            try:
                raw  = line.split(":", 1)[1].strip()
                probs = np.array([float(v.strip()) for v in raw.split(",")], dtype=np.float32)
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
            return probs, pred_idx, status
        elif line == "INFER":
            status = "infer"
            return probs, pred_idx, status

    return probs, pred_idx, status

def send_signal(ser, window):
    """Sender float32 arrays i chunks af 256 bytes (kræver ACK tjek pga. main branch cnn adfærd)."""
    data_bytes = window.astype(np.float32).tobytes()
    for j in range(0, len(data_bytes), CHUNK_SIZE):
        chunk = data_bytes[j:j + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        readline(ser)
    time.sleep(0.002)


# def prepare_dataset(rms_data, angles_ds): # HARD limit version, not to great, reduces sample count by too much
#     rms_data = rms_data[:N_CHANNELS, :]
#     X = rms_data.T

#     y = np.full(len(angles_ds), 2, dtype=np.int64)
#     y[angles_ds >  2.0] = 1   # plantar
#     y[angles_ds < -2.0] = 0   # dorsi

#     X_seq, y_seq = [], []
#     for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
#         window_labels = y[i - SEQ_LEN:i]
#         counts   = np.bincount(window_labels, minlength=3)
#         majority = int(np.argmax(counts))
#         if counts[majority] < SEQ_LEN * 0.6:
#             continue
#         X_seq.append(X[i - SEQ_LEN:i].T)
#         y_seq.append(majority)

#     return np.array(X_seq), np.array(y_seq)

# def prepare_dataset(rms_data, angles_ds):
#     rms_data = rms_data[:N_CHANNELS, :]
#     X = rms_data.T

#     y = np.full(len(angles_ds), 2, dtype=np.int64)
#     y[angles_ds >  2.0] = 1   # plantar
#     y[angles_ds < -2.0] = 0   # dorsi

#     X_seq, y_seq = [], []
#     for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
#         X_seq.append(X[i - SEQ_LEN:i].T)
#         y_seq.append(y[i])

#     return np.array(X_seq), np.array(y_seq)


def prepare_dataset(rms_data, angles_ds): # majority voting
    rms_data = rms_data[:N_CHANNELS, :]
    X = rms_data.T

    y = np.full(len(angles_ds), 2, dtype=np.int64)  # Default: None (2)
    y[angles_ds >  2.0] = 1   # Plantar (1)
    y[angles_ds < -2.0] = 0   # Dorsi (0)

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
        window_labels = y[i - SEQ_LEN:i]

        counts = np.bincount(window_labels, minlength=3)

        majority = int(np.argmax(counts))

        if counts[majority] < (SEQ_LEN * 0.6):
            continue

        if majority == 2 and counts[2] < (SEQ_LEN * 0.4):
            continue

        X_seq.append(X[i - SEQ_LEN:i].T)
        y_seq.append(majority)

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
            X_all.append(X); y_all.append(y)
    return np.concatenate(X_all), np.concatenate(y_all)

def load_or_build_splits(data_dir, pretrain_rats, live_rats, out_dir="./splits", seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"{s}.npz") for s in ("train", "val", "live")}

    if all(os.path.exists(p) for p in paths.values()):
        print("Splits already exist, loading...")
        return {s: np.load(p) for s, p in paths.items()}

    print("Building Pretrain dataset (Rats 4-9)...")
    X_base, y_base = build_dataset(data_dir, pretrain_rats)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_base))
    train_end = int(0.85 * len(idx)) # 85% train, 15% val

    splits = {
        "train": idx[:train_end],
        "val": idx[train_end:]
    }

    print("Building Live dataset (Rat 10)...")
    X_live, y_live = build_dataset(data_dir, live_rats)


    scaler = MinMaxScaler()
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
    np.savez("scaler.npz", mean=np.asarray(scaler.min_), scale=np.asarray(scaler.scale_))

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

        probs, pred_idx, status = wait_for_answer(ser)
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

        probs, pred_idx, status = wait_for_answer(ser)

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


def save_model(ser, filename="trained_weights.h"):
    print("Requesting model export...")
    ser.reset_input_buffer()
    ser.write(b"EX")
    ser.flush()
    with open(filename, "w") as f:
        started = False
        while True:
            line = readline(ser)
            if "START_EXPORT" in line: started = True; continue
            if "END_EXPORT"   in line: break
            if started: f.write(line + "\n")
    print(f"Model saved to {filename}")
    ser.write(b"R")
    ser.flush()

def early_stopping_check(ser, val_lines):
    global minimum_val_loss, val_loss_counter
    for line in val_lines:
        if "VAL_LOSS" in line:
            try:
                loss = float(line.split(":")[1])
                if loss < minimum_val_loss:
                    minimum_val_loss = loss
                    val_loss_counter = 0
                    save_model(ser)
                else:
                    val_loss_counter += 1
            except Exception:
                pass


def stream_live(ser, windows, labels, csv_writer):
    print(f"\n--- STARTING LIVE ONLINE LEARNING (Chronological) ---")
    combined = list(zip(windows, labels))


    running_correct = 0
    running_total   = 0
    running_cm      = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

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

        probs, pred_idx, status = wait_for_answer(ser)
        CONFIDENCE_THRESHOLD = 0.45


        confidence = float(np.max(probs)) if probs is not None else 0.0
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
        # if probs is not None:
        #     confidence = float(np.max(probs))

        #     if confidence < CONFIDENCE_THRESHOLD:
        #         pred_idx = -1
        # else:
        #     confidence = 0.0

        # if pred_idx >= 0:
        #     running_total += 1
        #     if pred_idx == int(label):
        #         running_correct += 1
        #     running_cm[int(label), pred_idx] += 1

        f1 = calculate_macro_f1(running_cm)
        acc = running_correct / running_total if running_total > 0 else 0.0


        csv_writer.writerow([
            "LIVE", i, int(label), pred_idx,
            f"{confidence:.4f}", running_correct, running_total,
            f"{acc:.6f}", f"{f1:.6f}", status,
        ])

        if running_total > 0:
            pbar.set_postfix({"Live_Acc": f"{acc*100:.2f}%", "Live_F1": f"{f1:.4f}"})

    final_acc = running_correct / running_total if running_total > 0 else 0.0
    final_f1  = calculate_macro_f1(running_cm)
    print(f"\nLive Streaming Finished -- Final Acc: {final_acc*100:.2f}%  Final F1: {final_f1:.4f}")



if __name__ == "__main__":
    data_dir = "./dataset_rats_50w"
    splits_dir = "./splits_rat10_only"
    PRETRAIN_RATS = [9]
    LIVE_RAT      = [10]

    splits   = load_or_build_splits(data_dir, PRETRAIN_RATS, LIVE_RAT)
    X_train  = splits["train"]["X"]
    y_train  = splits["train"]["y"]
    X_val    = splits["val"]["X"]
    y_val    = splits["val"]["y"]
    X_live   = splits["live"]["X"]
    y_live   = splits["live"]["y"]

    # For testing train - test - stream on rat 10
    # X_train = np.load(f"{splits_dir}/train.npz")["X"] 
    # y_train = np.load(f"{splits_dir}/train.npz")["y"]
    # X_val   = np.load(f"{splits_dir}/val.npz")["X"]
    # y_val   = np.load(f"{splits_dir}/val.npz")["y"]
    # X_live  = np.load(f"{splits_dir}/live.npz")["X"]
    # y_live  = np.load(f"{splits_dir}/live.npz")["y"]


    f_csv   = open("training_accuracy.csv", "w", newline="", encoding="utf-8")
    writer  = csv.writer(f_csv)
    writer.writerow(["epoch", "window_index", "label", "pred", "confidence",
                     "running_correct", "running_total",
                     "running_accuracy", "running_f1", "status"])
    
    # saving seperately for rat10 only test
    # f_csv   = open("training_accuracy_rat10.csv", "w", newline="", encoding="utf-8")
    # writer  = csv.writer(f_csv)
    # writer.writerow(["epoch", "window_index", "label", "pred", "confidence",
    #                  "running_correct", "running_total",
    #                  "running_accuracy", "running_f1", "status"])

    print("Opening serial port...")
    ser = serial.Serial(PORT, BAUD, timeout=5)

    arduino_ready = False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        line = readline(ser)
        if line == "READY":
            ser.write(b"G")
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
        # Her streamer vi kronologisk til en HELT BLANK model!
        stream_live(ser, X_live, y_live, writer)
        f_csv.flush()

    except KeyboardInterrupt:
        print("\nInterrupted.")


    # try:
    #     print("\n=== PHASE 1: PRETRAIN ON RATS 4-9 ===")
    #     for epoch in range(EPOCHS):
    #         stream_epoch(ser, epoch, X_train, y_train, writer)
    #         f_csv.flush()

    #         val_lines = run_validation(ser, X_val, y_val)
    #         early_stopping_check(ser, val_lines)

    #         if val_loss_counter >= 5:
    #             print("\nEarly stopping triggered. Model pretraining stopped.")
    #             break

    #     print("\n=== PHASE 2: LIVE ONLINE LEARNING ON RAT 10 ===")

    #     stream_live(ser, X_live, y_live, writer)
    #     f_csv.flush()

    # except KeyboardInterrupt:
    #     print("\nInterrupted.")

    f_csv.close()
    ser.close()
    print("Done.")
