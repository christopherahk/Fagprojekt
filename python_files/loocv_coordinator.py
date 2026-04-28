"""
Leave-One-Rat-Out Cross-Validation Coordinator.

Reads .rhd files, groups by rat, performs Leave-One-Rat-Out validation
by sending train/test data to Arduino via Serial and collecting results.

Usage:
    python loocv_coordinator.py --data-root data --port COM3 --baudrate 115200

How to use:
1) Flash Arduino with LOOCV-compatible firmware (for example streaming_loocv.cpp + Fagprojekt.ino).
2) Ensure the board prints NET_READY on serial startup.
3) Run this script with the correct serial port.
4) After completion, inspect analysis_outputs/loocv_results.json.
95% confidence interval for accuracy
95% confidence interval for loss
One-sided p-værdi for accuracy vs chance baseline (1/3) med one-sample t-test
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import serial
import spikeinterface.extractors as se
from scipy import stats
from tqdm import tqdm

@dataclass
class RatRecording:
    """Metadata for a single .rhd recording."""
    file_path: str
    rat: str
    label: str
    channels: int
    sampling_hz: float
    num_samples: int
    duration_s: float


@dataclass
class FoldResult:
    """Result of a single LOOCV fold."""
    test_rat: str
    loss_final: float
    accuracy: float
    confusion_matrix: list  # 3x3 for 3 classes
    num_train_samples: int
    num_test_samples: int
    timestamp: str


def mean_std_ci(values: list[float], confidence: float = 0.95) -> dict:
    """Return mean, std, and two-sided CI for a sample mean."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "confidence": confidence,
        }

    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1:
        se = std / np.sqrt(n)
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
        margin = t_crit * se
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = mean
        ci_high = mean

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
    }


def one_sample_pvalue_greater(values: list[float], baseline: float) -> dict:
    """One-sample t-test p-value for H1: mean(values) > baseline."""
    if not values:
        return {"baseline": baseline, "t_stat": 0.0, "p_value": 1.0, "n": 0}

    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n < 2:
        # Not enough folds for a reliable variance estimate.
        return {"baseline": baseline, "t_stat": 0.0, "p_value": 1.0, "n": n}

    t_stat, p_two_sided = stats.ttest_1samp(arr, popmean=baseline)
    t_stat = float(t_stat)
    p_two_sided = float(p_two_sided)

    # Convert to one-sided p-value for mean > baseline.
    if t_stat >= 0.0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - (p_two_sided / 2.0)

    return {
        "baseline": baseline,
        "t_stat": t_stat,
        "p_value": float(p_one_sided),
        "n": n,
    }


def parse_rat_and_label(file_path: str) -> tuple[str, str]:
    """Extract rat and label from file path."""
    parent = os.path.basename(os.path.dirname(file_path))
    parts = parent.split("_")
    rat = parts[0] if parts else "UNKNOWN"
    label = "_".join(parts[1:]) if len(parts) > 1 else parent
    return rat, label


def collect_rhd_paths(root: str) -> list[str]:
    """Recursively find all .rhd files."""
    paths = []
    for dir_path, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".rhd"):
                paths.append(os.path.join(dir_path, name))
    return sorted(paths)


def load_recording_info(file_path: str, stream_id: str = "0") -> RatRecording:
    """Load metadata from .rhd file."""
    rec = se.read_intan(file_path, stream_id=stream_id)
    sampling_hz = float(rec.get_sampling_frequency())
    num_samples = int(rec.get_num_samples())
    channels = int(rec.get_num_channels())
    duration_s = (num_samples / sampling_hz) if sampling_hz else 0.0
    rat, label = parse_rat_and_label(file_path)

    return RatRecording(
        file_path=file_path,
        rat=rat,
        label=label,
        channels=channels,
        sampling_hz=sampling_hz,
        num_samples=num_samples,
        duration_s=duration_s,
    )


def extract_features(rec_path: str, label: str, stream_id: str = "0") -> tuple[np.ndarray, str]:
    """
    Extract features from .rhd recording.

    Returns: (features_array [56x100], label_str)

    The features array is normalized across channels and downsampled to 100 time-points.
    """
    rec = se.read_intan(rec_path, stream_id=stream_id)
    traces = rec.get_traces()  # shape: (num_samples, num_channels)

    # Ensure we have 56 channels (pad or trim)
    if traces.shape[1] < 56:
        padding = np.zeros((traces.shape[0], 56 - traces.shape[1]), dtype=traces.dtype)
        traces = np.hstack([traces, padding])
    elif traces.shape[1] > 56:
        traces = traces[:, :56]

    # Normalize each channel
    traces = traces.astype(np.float32)
    channel_means = np.mean(traces, axis=0, keepdims=True)
    channel_stds = np.std(traces, axis=0, keepdims=True)
    channel_stds[channel_stds < 1e-6] = 1.0  # Avoid div-by-zero
    traces = (traces - channel_means) / channel_stds

    # Downsample to 100 time-points (56x100 as expected by streaming.cpp)
    target_samples = 100
    if traces.shape[0] > target_samples:
        indices = np.linspace(0, traces.shape[0] - 1, target_samples, dtype=int)
        features = traces[indices, :].T  # shape: (56, 100)
    else:
        # Pad with zeros if too short
        features = np.zeros((56, target_samples), dtype=np.float32)
        features[:, :traces.shape[0]] = traces.T

    return features, label


def map_label_to_class_index(label: str) -> int:
    """Map label string to class index."""
    label_map = {
        "DORSIFLEXION": 0,
        "PLANTARFLEXION": 1,
        "DORSI_PLANTAR_PART1": 2,
        "DORSI_PLANTAR_PART2": 2,  # Group with DORSI_PLANTAR_PART1
        "PRICKING": 1,  # Binary-ish for now: DORSI=0, (PLANTAR+PRICKING)=1, DORSI_PLANTAR=2
    }
    return label_map.get(label, 0)


def format_feature_row_csv(feature_row: np.ndarray) -> str:
    """Format a single feature row (100 values) as CSV line."""
    return ",".join(f"{v:.6f}" for v in feature_row)


def format_label_csv(class_idx: int) -> str:
    """Format one-hot label as CSV."""
    one_hot = [0, 0, 0]
    one_hot[class_idx] = 1
    return ",".join(str(v) for v in one_hot)


def open_serial_port(port: str, baudrate: int, timeout: float = 5.0) -> serial.Serial:
    """Open serial connection to Arduino."""
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset
        return ser
    except Exception as e:
        print(f"ERROR: Could not open serial port {port}: {e}")
        sys.exit(1)


def send_training_data(
    ser: serial.Serial,
    train_features_labels: list[tuple[np.ndarray, int]],
    fold_idx: int,
    num_folds: int,
) -> None:
    """Send training data to Arduino."""
    print(f"\n[Fold {fold_idx+1}/{num_folds}] Sending training data ({len(train_features_labels)} samples)...")

    ser.write(b"\n")
    time.sleep(0.1)
    ser.write(f"START_FOLD {fold_idx}\n".encode())
    time.sleep(0.5)

    for idx, (features, class_idx) in enumerate(train_features_labels):
        # Send each of the 56 rows
        for row_idx in range(56):
            row_data = format_feature_row_csv(features[row_idx])
            ser.write(f"{row_data}\n".encode())
            if (idx + 1) % 10 == 0:
                print(f"  Sent {idx+1}/{len(train_features_labels)} train samples...")

        # Send label after all 56 rows
        label_data = format_label_csv(class_idx)
        ser.write(f"LABEL,{label_data}\n".encode())
        time.sleep(0.01)

    print("  Training data sent.")


def send_test_data(
    ser: serial.Serial,
    test_features_labels: list[tuple[np.ndarray, int]],
) -> None:
    """Send test data to Arduino."""
    print(f"Sending test data ({len(test_features_labels)} samples)...")

    ser.write(b"START_EVAL\n")
    time.sleep(0.5)

    for idx, (features, class_idx) in enumerate(test_features_labels):
        for row_idx in range(56):
            row_data = format_feature_row_csv(features[row_idx])
            ser.write(f"{row_data}\n".encode())

        # Send label for reference (Arduino may use for evaluation)
        label_data = format_label_csv(class_idx)
        ser.write(f"LABEL,{label_data}\n".encode())
        time.sleep(0.01)

    print("  Test data sent.")

    # Request results
    time.sleep(0.5)
    ser.write(b"GET_RESULTS\n")
    time.sleep(0.2)


def read_fold_result(ser: serial.Serial, timeout: float = 120.0) -> dict:
    """Read fold result from Arduino."""
    print("Waiting for Arduino result...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("RESULT"):
                # Parse: "RESULT,loss=X,acc=Y,correct=C,total=T"
                try:
                    parts = line.split(",")
                    result = {}
                    for part in parts[1:]:
                        if "=" in part:
                            key, val = part.split("=")
                            result[key.strip()] = val.strip()
                    print(f"  Got result: {result}")
                    return result
                except Exception as e:
                    print(f"  Parse error: {e}")
                    continue
            # Also log confusion matrix if it arrives
            elif line.startswith("CONFUSION"):
                print(f"  Confusion matrix: {line}")

        time.sleep(0.1)

    print("  Timeout waiting for result!")
    return {}


def run_loocv(data_root: str, port: str, baudrate: int, output_dir: str) -> None:
    """Run Leave-One-Rat-Out cross-validation."""

    # Collect all .rhd files
    print("Scanning for .rhd files...")
    rhd_paths = collect_rhd_paths(data_root)
    if not rhd_paths:
        print(f"No .rhd files found in {data_root}")
        return

    print(f"Found {len(rhd_paths)} recordings.")

    # Load metadata and group by rat
    print("Loading recording metadata...")
    recordings_by_rat = defaultdict(list)
    for path in rhd_paths:
        try:
            info = load_recording_info(path)
            recordings_by_rat[info.rat].append(info)
        except Exception as e:
            print(f"  Skipped {path}: {e}")

    rats = sorted(recordings_by_rat.keys())
    print(f"Found {len(rats)} unique rats: {rats}")

    # Check for outliers
    outlier_rats = []
    for rat in rats:
        labels_set = set(info.label for info in recordings_by_rat[rat])
        if len(recordings_by_rat[rat]) < 5:
            outlier_rats.append((rat, len(recordings_by_rat[rat]), len(labels_set)))

    if outlier_rats:
        print("\nWARNING: Outlier rats detected:")
        for rat, num_files, num_labels in outlier_rats:
            print(f"  {rat}: {num_files} files, {num_labels} unique label(s)")
        print("  These rats may have insufficient data for robust evaluation.\n")

    # Extract features for all recordings
    print("Extracting features from all recordings...")
    features_by_recording = {}
    for rat in tqdm(rats):
        for info in recordings_by_rat[rat]:
            try:
                features, _ = extract_features(info.file_path, info.label)
                features_by_recording[info.file_path] = features
            except Exception as e:
                print(f"  Failed to extract features from {info.file_path}: {e}")

    # Open serial connection
    print(f"Opening serial connection to {port}...")
    ser = open_serial_port(port, baudrate)

    # Run LOOCV: Leave-One-Rat-Out
    fold_results = []
    num_folds = len(rats)

    for fold_idx, test_rat in enumerate(rats):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{num_folds}: Test rat = {test_rat}")
        print(f"{'='*70}")

        # Warn if testing on outlier
        if test_rat in [r[0] for r in outlier_rats]:
            print(f"WARNING: {test_rat} is an outlier with limited data!")
            print(f"  Results may not be statistically meaningful.")

        # Prepare training and test data
        train_data = []
        test_data = []

        for rat in rats:
            for info in recordings_by_rat[rat]:
                if info.file_path not in features_by_recording:
                    continue

                features = features_by_recording[info.file_path]
                class_idx = map_label_to_class_index(info.label)

                if rat == test_rat:
                    test_data.append((features, class_idx))
                else:
                    train_data.append((features, class_idx))

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        # Send data and run on Arduino
        send_training_data(ser, train_data, fold_idx, num_folds)
        send_test_data(ser, test_data)

        # Read result
        result = read_fold_result(ser)

        fold_result = FoldResult(
            test_rat=test_rat,
            loss_final=float(result.get("loss", 0.0)),
            accuracy=float(result.get("acc", 0.0)),
            confusion_matrix=[],  # TODO: parse from Arduino if sent
            num_train_samples=len(train_data),
            num_test_samples=len(test_data),
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        fold_results.append(fold_result)

        print(f"Fold result: loss={fold_result.loss_final:.4f}, acc={fold_result.accuracy:.4f}")

    ser.close()

    # Summarize results
    print(f"\n{'='*70}")
    print("LOOCV Summary")
    print(f"{'='*70}")

    # Split outliers from main results
    main_results = [r for r in fold_results if r.test_rat not in [o[0] for o in outlier_rats]]
    outlier_results = [r for r in fold_results if r.test_rat in [o[0] for o in outlier_rats]]

    accuracies = [r.accuracy for r in main_results]
    losses = [r.loss_final for r in main_results]
    acc_stats = mean_std_ci(accuracies, confidence=0.95)
    loss_stats = mean_std_ci(losses, confidence=0.95)

    # Accuracy significance versus chance-level baseline for 3 classes.
    chance_baseline = 1.0 / 3.0
    acc_vs_chance = one_sample_pvalue_greater(accuracies, chance_baseline)

    print(f"\nMain results ({len(main_results)} folds):")
    print(f"Mean Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
    print(
        f"Accuracy 95% CI: [{acc_stats['ci_low']:.4f}, {acc_stats['ci_high']:.4f}]"
    )
    print(f"Mean Loss: {loss_stats['mean']:.4f} ± {loss_stats['std']:.4f}")
    print(f"Loss 95% CI: [{loss_stats['ci_low']:.4f}, {loss_stats['ci_high']:.4f}]")
    print(
        f"Accuracy vs chance (>{chance_baseline:.3f}) p-value: "
        f"{acc_vs_chance['p_value']:.6f}"
    )
    print("\nPer-rat results (main):")
    for r in main_results:
        print(f"  {r.test_rat}: acc={r.accuracy:.4f}, loss={r.loss_final:.4f}")

    if outlier_results:
        print("\nOutlier results (limited data, use with caution):")
        for r in outlier_results:
            print(f"  {r.test_rat}: acc={r.accuracy:.4f}, loss={r.loss_final:.4f}, n_test={r.num_test_samples}")

    # Write results to file
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "loocv_results.json")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_folds": len(fold_results),
        "num_outlier_folds": len(outlier_results),
        "outlier_rats": [(o[0], o[1], o[2]) for o in outlier_rats],
        "main_fold_stats": {
            "accuracy": acc_stats,
            "loss": loss_stats,
        },
        "p_values": {
            "accuracy_greater_than_chance": acc_vs_chance,
        },
        "fold_results": [asdict(r) for r in fold_results],
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Leave-One-Rat-Out LOOCV coordinator for Arduino training."
    )
    parser.add_argument("--data-root", default="data", help="Path to folder with .rhd files.")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux).")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Folder where results are written.")

    args = parser.parse_args()

    run_loocv(
        data_root=args.data_root,
        port=args.port,
        baudrate=args.baudrate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
