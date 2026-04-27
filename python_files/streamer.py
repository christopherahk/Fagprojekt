import os
import time

import numpy as np
import serial
import tqdm
import spikeinterface.extractors as se
from scipy.signal import decimate

PORT = "COM3"
BAUD = 115200

DATA_FOLDER = "data"
SINGLE_FILE = None

STREAM_ID = "0"
RAW_FS = 30_000
DOWNSAMPLE = 100
N_CHANNELS = 56
WINDOW = 100

CLASSES = {
    "DORSIFLEXION":   0,
    "PLANTARFLEXION": 1,
    "PRICKING":       2,
}
N_CLASSES = len(CLASSES)

SEND_LABELS = True

LINE_GAP_MIN = 0.003
LINE_GAP_MAX = 0.100
LINE_GAP     = 0.005

ACK_TIMEOUT  = 5.0
RESET_PAUSE  = 0.20


def find_rhd_files(root):
    abs_root = os.path.abspath(root)
    print(f"  Scanning: {abs_root}")

    if not os.path.isdir(abs_root):
        print(f"  [ERROR] Folder not found: {abs_root}")
        return []

    try:
        all_dirs = sorted(os.listdir(abs_root))
    except PermissionError as e:
        print(f"  [ERROR] Cannot list folder: {e}")
        return []

    part2_dirs = [d for d in all_dirs if d.lower().endswith("dorsiflexion")]
    other_dirs = [d for d in all_dirs if not d.lower().endswith("dorsiflexion")]

    print(f"  Subfolders ending with 'dorsiflexion' ({len(part2_dirs)}): {part2_dirs}")
    print(f"  Subfolders skipped ({len(other_dirs)}): {other_dirs}")

    paths = []
    for folder in part2_dirs:
        folder_path = os.path.join(abs_root, folder)
        rhd_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".rhd"))
        print(f"  {folder}: {len(rhd_files)} .rhd file(s)")
        for name in rhd_files:
            paths.append(os.path.join(folder_path, name))

    return paths


def infer_label(path):
    folder = os.path.basename(os.path.dirname(path)).upper()
    for class_name, idx in CLASSES.items():
        if class_name in folder:
            one_hot = [0] * N_CLASSES
            one_hot[idx] = 1
            return one_hot
    print(f"  [LABEL_SKIP] Cannot infer class from folder: {os.path.dirname(path)}")
    return None


def load_and_preprocess(path):
    recording = se.read_intan(path, stream_id=STREAM_ID)
    n_samples  = recording.get_num_samples()
    n_ch       = recording.get_num_channels()

    if n_ch < N_CHANNELS:
        print(f"  [SKIP] {path}: only {n_ch} channels (need {N_CHANNELS})")
        return None

    raw_window = WINDOW * DOWNSAMPLE
    n_frames = n_samples // raw_window
    if n_frames == 0:
        print(f"  [SKIP] {path}: too short ({n_samples} samples)")
        return None

    frames = []
    for i in range(n_frames):
        start = i * raw_window
        end = start + raw_window

        raw = recording.get_traces(start_frame=start, end_frame=end)
        down = decimate(raw, DOWNSAMPLE, axis=0).astype(np.float32)
        down = down[:, :N_CHANNELS]
        down /= 32768.0
        frames.append(down.T)

    return np.stack(frames, axis=0)


def readline(ser):
    try:
        return ser.readline().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def wait_for_ack(ser):
    deadline = time.monotonic() + ACK_TIMEOUT
    while time.monotonic() < deadline:
        if ser.in_waiting:
            line = readline(ser)
            if line:
                print(f"  Arduino: {line}")
            if line.startswith("OUT,") or line.startswith("TRAIN,"):
                return "ok"
            if line.startswith("ERR_FRAME_RESET"):
                ser.reset_input_buffer()
                time.sleep(RESET_PAUSE)
                return "reset"
        else:
            time.sleep(0.005)
    ser.reset_input_buffer()
    return "timeout"


_line_gap = LINE_GAP


def send_frame(ser, frame, label):
    global _line_gap

    lines = []
    for ch in range(N_CHANNELS):
        values = ",".join(f"{v:.5f}" for v in frame[ch])
        lines.append((values + "\n").encode("utf-8"))

    if label is not None and SEND_LABELS:
        lines.append((",".join(str(v) for v in label) + "\n").encode("utf-8"))

    ser.write(b"".join(lines))
    ser.flush()

    result = wait_for_ack(ser)

    if result == "ok":
        _line_gap = max(LINE_GAP_MIN, _line_gap * 0.90)
    else:
        _line_gap = min(LINE_GAP_MAX, _line_gap * 2.0)
        print(f"  [GAP] line gap now {_line_gap*1000:.1f} ms (kept for label send)")

    return result == "ok"


def stream_file(ser, path):
    print(f"\nLoading: {path}")
    data = load_and_preprocess(path)
    if data is None:
        return 0, 0

    label   = infer_label(path)
    n_frames = data.shape[0]
    ok = dropped = 0

    for i in range(n_frames):
        success = send_frame(ser, data[i], label)
        if success:
            ok += 1
        else:
            dropped += 1

    print(f"  Done: {ok} ok, {dropped} dropped out of {n_frames} frames")
    return ok, dropped


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    try:
        print(f"Opened {PORT} at {BAUD} baud.")
        print("Waiting for Arduino to boot (2 s)...")
        time.sleep(2.0)
        while ser.in_waiting:
            print(f"  Arduino: {readline(ser)}")

        if SINGLE_FILE:
            files = [SINGLE_FILE]
        else:
            files = find_rhd_files(DATA_FOLDER)
            print(f"Found {len(files)} .rhd files under '{DATA_FOLDER}'.")

        total_ok = total_dropped = 0
        for path in tqdm.tqdm(files, desc="Files"):
            ok, dropped = stream_file(ser, path)
            total_ok += ok
            total_dropped += dropped

        print(f"\nFinished. Total frames - ok: {total_ok}, dropped: {total_dropped}")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
