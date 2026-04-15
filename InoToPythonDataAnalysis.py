"""Script overview.

Purpose:
- Read CSV cells that contain 3 softmax values + 3 one-hot labels.
- Decode each valid cell into a flat sample table.
- Compute classification statistics, calibration metrics, and stream-drift metrics.
- Save the results as JSON, CSV, and plots in one output folder.

High-level flow:
1) "parse_args" defines CLI options, including optional "--input-path".
2) "collect_csv_files" selects the intended CSV file or scans a folder with include/exclude globs.
3) "read_and_decode_csv" parses each CSV and normalizes valid cells into rows.
4) "compute_and_save_statistics" calculates metrics, writes JSON/CSV, and creates plots.
5) "main" ties everything together and shows a progress bar.

Class order (fixed):
- 0: dorsiflexion
- 1: plantarflexion
- 2: pricking

Main outputs:
- "softmax_label_stats.json"
- "decoded_softmax_labels.csv"
- "confusion_matrix.png"
- "class_distribution_true_vs_pred.png"
- "true_class_confidence_hist.png"
- "true_class_confidence_boxplot.png"
- "reliability_diagram.png"
- "stream_drift_by_chunk.png"

Tip:
- Set "TARGET_PATH" below for a no-argument run, or override with "--input-path".
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


CLASS_NAMES = ["dorsiflexion", "plantarflexion", "pricking"]
CLASS_INDEX_TO_NAME = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Default input target used when --input-path is not provided.
# Can be either a CSV file path or a directory path.
TARGET_PATH = Path("softmax_test_120.csv")


def infer_csv_delimiter(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)

    # Try common delimiters first. Falls back to comma if no strong signal.
    candidates = [",", ";", "\t", "|"]
    counts = {c: sample.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def is_one_hot(values: list[float], tol: float = 1e-6) -> bool:
    if len(values) != 3:
        return False
    rounded = [1 if abs(v - 1.0) <= tol else 0 if abs(v) <= tol else -1 for v in values]
    return rounded.count(1) == 1 and rounded.count(0) == 2


def is_probability_triplet(values: list[float], tol: float = 1e-3) -> bool:
    if len(values) != 3:
        return False
    if any(v < -tol or v > 1.0 + tol for v in values):
        return False
    return abs(sum(values) - 1.0) <= tol


def try_parse_bracketed_groups(cell_text: str) -> tuple[list[float], list[int]] | None:
    groups = re.findall(r"\[([^\]]+)\]", cell_text)
    if len(groups) < 2:
        return None

    parsed_groups: list[list[float]] = []
    for group in groups:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", group)]
        if len(nums) >= 3:
            parsed_groups.append(nums[:3])

    if len(parsed_groups) < 2:
        return None

    first, second = parsed_groups[0], parsed_groups[1]
    if is_probability_triplet(first) and is_one_hot(second):
        return first, [int(round(x)) for x in second]
    if is_probability_triplet(second) and is_one_hot(first):
        return second, [int(round(x)) for x in first]

    return None


def parse_prediction_label_cell(cell_value: Any) -> tuple[list[float], list[int]] | None:
    if cell_value is None:
        return None

    cell_text = str(cell_value).strip()
    if cell_text == "" or cell_text.lower() in {"nan", "none", "null"}:
        return None

    bracketed = try_parse_bracketed_groups(cell_text)
    if bracketed is not None:
        return bracketed

    numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell_text)]
    if len(numbers) < 6:
        return None

    for i in range(0, len(numbers) - 5):
        a = numbers[i : i + 3]
        b = numbers[i + 3 : i + 6]
        if is_probability_triplet(a) and is_one_hot(b):
            return a, [int(round(x)) for x in b]
        if is_probability_triplet(b) and is_one_hot(a):
            return b, [int(round(x)) for x in a]

    return None


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=float)
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std_val,
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def wilson_ci(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}

    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    return {
        "lower": float(max(0.0, center - margin)),
        "upper": float(min(1.0, center + margin)),
    }


def bootstrap_ci(values: np.ndarray, rounds: int = 500, seed: int = 42) -> dict[str, float]:
    if values.size == 0:
        return {"lower": 0.0, "upper": 0.0}

    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(rounds, dtype=float)
    for i in range(rounds):
        sample = values[rng.integers(0, n, n)]
        means[i] = float(np.mean(sample))
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
    }


def multiclass_brier_score(prob_matrix: np.ndarray, true_classes: np.ndarray) -> float:
    one_hot = np.zeros_like(prob_matrix)
    one_hot[np.arange(true_classes.size), true_classes] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def expected_calibration_error(pred_conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = pred_conf.size
    if n == 0:
        return 0.0, []

    bin_rows: list[dict[str, float]] = []
    ece = 0.0
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i == bins - 1:
            mask = (pred_conf >= left) & (pred_conf <= right)
        else:
            mask = (pred_conf >= left) & (pred_conf < right)

        count = int(np.sum(mask))
        if count == 0:
            bin_rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                    "abs_gap": 0.0,
                }
            )
            continue

        mean_conf = float(np.mean(pred_conf[mask]))
        acc = float(np.mean(correct[mask]))
        gap = abs(acc - mean_conf)
        ece += (count / n) * gap
        bin_rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "abs_gap": float(gap),
            }
        )

    return float(ece), bin_rows


def plot_reliability_diagram(bin_rows: list[dict[str, float]], output_path: Path) -> None:
    if not bin_rows:
        return

    centers = [(r["bin_left"] + r["bin_right"]) / 2.0 for r in bin_rows]
    acc = [r["accuracy"] for r in bin_rows]
    conf = [r["mean_confidence"] for r in bin_rows]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.3, label="Perfect calibration")
    ax.plot(centers, acc, marker="o", color="#1E88E5", label="Empirical accuracy")
    ax.plot(centers, conf, marker="s", color="#E4572E", label="Mean confidence")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy / Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stream_drift(decoded_df: pd.DataFrame, output_path: Path, chunks: int = 10) -> list[dict[str, float]]:
    n = len(decoded_df)
    if n == 0:
        return []

    chunk_ids = np.minimum((np.arange(n) * chunks) // max(1, n), chunks - 1)
    decoded_df = decoded_df.copy()
    decoded_df["chunk_id"] = chunk_ids

    rows: list[dict[str, float]] = []
    for chunk in range(chunks):
        chunk_df = decoded_df.loc[decoded_df["chunk_id"] == chunk]
        if chunk_df.empty:
            continue
        rows.append(
            {
                "chunk": int(chunk),
                "n_samples": int(len(chunk_df)),
                "accuracy": float(chunk_df["correct"].mean()),
                "mean_true_confidence": float(chunk_df["true_class_confidence"].mean()),
                "mean_pred_confidence": float(chunk_df["pred_class_confidence"].mean()),
            }
        )

    if rows:
        x = [r["chunk"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        tc = [r["mean_true_confidence"] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(x, acc, marker="o", label="Accuracy", color="#3A7D44")
        ax.plot(x, tc, marker="s", label="Mean true-class confidence", color="#2E86AB")
        ax.set_title("Stream Drift by Chunk")
        ax.set_xlabel("Chunk index (stream order)")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    return rows


def read_and_decode_csv(csv_path: Path) -> tuple[list[dict[str, Any]], int]:
    delimiter = infer_csv_delimiter(str(csv_path))
    df = pd.read_csv(csv_path, sep=delimiter, dtype=str, keep_default_na=False)

    decoded_rows: list[dict[str, Any]] = []
    undecoded_cells = 0

    for row_idx, row in df.iterrows():
        for col_name, cell in row.items():
            parsed = parse_prediction_label_cell(cell)
            if parsed is None:
                undecoded_cells += 1
                continue

            probs, labels = parsed
            true_class = int(np.argmax(labels))
            pred_class = int(np.argmax(probs))
            sorted_probs = sorted(probs, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1])

            decoded_rows.append(
                {
                    "source_file": str(csv_path),
                    "row_index": int(row_idx),
                    "column_name": str(col_name),
                    "p0": float(probs[0]),
                    "p1": float(probs[1]),
                    "p2": float(probs[2]),
                    "y0": int(labels[0]),
                    "y1": int(labels[1]),
                    "y2": int(labels[2]),
                    "true_class": true_class,
                    "true_class_name": CLASS_INDEX_TO_NAME[true_class],
                    "pred_class": pred_class,
                    "pred_class_name": CLASS_INDEX_TO_NAME[pred_class],
                    "true_class_confidence": float(probs[true_class]),
                    "pred_class_confidence": float(max(probs)),
                    "margin_top1_top2": margin,
                    "correct": int(pred_class == true_class),
                }
            )

    return decoded_rows, undecoded_cells


def plot_confusion_matrix(conf_mat: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat, cmap="Blues")
    ax.set_title("Confusion Matrix (Pred vs True)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, int(conf_mat[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_histogram_true_confidence(values: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30, color="#2E86AB", edgecolor="black", alpha=0.85)
    ax.set_title("Distribution of True-Class Confidence")
    ax.set_xlabel("Softmax confidence for true class")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_distribution(true_counts: np.ndarray, pred_counts: np.ndarray, output_path: Path) -> None:
    x = np.arange(3)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, true_counts, width=width, label="True", color="#6CA965")
    ax.bar(x + width / 2, pred_counts, width=width, label="Predicted", color="#D1495B")
    ax.set_title("Class Distribution: True vs Predicted")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_by_true_class(df: pd.DataFrame, output_path: Path) -> None:
    data = [
        df.loc[df["true_class"] == cls, "true_class_confidence"].values
        for cls in [0, 1, 2]
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
    colors = ["#F4A259", "#5B8E7D", "#3D5A80"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("True-Class Confidence by Label")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_and_save_statistics(decoded_rows: list[dict[str, Any]], output_dir: Path, input_paths: list[Path], undecoded_cells: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded_df = pd.DataFrame(decoded_rows)
    decoded_df.to_csv(output_dir / "decoded_softmax_labels.csv", index=False)

    true_classes = decoded_df["true_class"].to_numpy(dtype=int)
    pred_classes = decoded_df["pred_class"].to_numpy(dtype=int)
    conf_true = decoded_df["true_class_confidence"].to_numpy(dtype=float)
    conf_pred = decoded_df["pred_class_confidence"].to_numpy(dtype=float)
    margin = decoded_df["margin_top1_top2"].to_numpy(dtype=float)
    prob_matrix = decoded_df[["p0", "p1", "p2"]].to_numpy(dtype=float)
    accuracy = float((decoded_df["correct"] == 1).mean())

    confusion = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    class_counts_true = np.bincount(true_classes, minlength=3)
    class_counts_pred = np.bincount(pred_classes, minlength=3)

    per_class_metrics: dict[str, Any] = {}
    class_recalls: list[float] = []
    class_f1_values: list[float] = []
    for cls in [0, 1, 2]:
        tp = int(confusion[cls, cls])
        fn = int(np.sum(confusion[cls, :]) - tp)
        fp = int(np.sum(confusion[:, cls]) - tp)
        support = int(np.sum(confusion[cls, :]))
        predicted_count = int(np.sum(confusion[:, cls]))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        class_recalls.append(recall)
        class_f1_values.append(f1)

        per_class_metrics[CLASS_INDEX_TO_NAME[cls]] = {
            "class_index": cls,
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "precision_ci95_wilson": wilson_ci(tp, tp + fp),
            "recall": recall,
            "recall_ci95_wilson": wilson_ci(tp, tp + fn),
            "f1": f1,
        }

    balanced_accuracy = float(np.mean(class_recalls))
    macro_f1 = float(np.mean(class_f1_values))
    accuracy_ci = bootstrap_ci(decoded_df["correct"].to_numpy(dtype=float), rounds=500, seed=42)
    ece, calibration_bins = expected_calibration_error(
        pred_conf=decoded_df["pred_class_confidence"].to_numpy(dtype=float),
        correct=decoded_df["correct"].to_numpy(dtype=float),
        bins=10,
    )
    brier = multiclass_brier_score(prob_matrix=prob_matrix, true_classes=true_classes)
    stream_drift_rows = plot_stream_drift(decoded_df, output_dir / "stream_drift_by_chunk.png", chunks=10)

    class_stats: dict[str, Any] = {}
    for cls in [0, 1, 2]:
        cls_df = decoded_df.loc[decoded_df["true_class"] == cls]
        class_name = CLASS_INDEX_TO_NAME[cls]
        class_stats[class_name] = {
            "class_index": cls,
            "count": int(len(cls_df)),
            "true_class_confidence": summary_stats(cls_df["true_class_confidence"].astype(float).tolist()),
            "pred_class_confidence": summary_stats(cls_df["pred_class_confidence"].astype(float).tolist()),
            "margin_top1_top2": summary_stats(cls_df["margin_top1_top2"].astype(float).tolist()),
        }

    overall_stats = {
        "samples_decoded": int(len(decoded_df)),
        "input_files": [str(p) for p in input_paths],
        "undecoded_cells": int(undecoded_cells),
        "accuracy": accuracy,
        "accuracy_ci95_bootstrap": accuracy_ci,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "class_order": CLASS_NAMES,
        "classification_metrics": {
            "per_class": per_class_metrics,
        },
        "calibration": {
            "expected_calibration_error": ece,
            "multiclass_brier_score": brier,
            "bins": calibration_bins,
        },
        "stream_drift_chunks": stream_drift_rows,
        "overall": {
            "true_class_confidence": summary_stats(conf_true.tolist()),
            "pred_class_confidence": summary_stats(conf_pred.tolist()),
            "margin_top1_top2": summary_stats(margin.tolist()),
        },
        "distribution": {
            "true_class_counts": {CLASS_NAMES[i]: int(class_counts_true[i]) for i in range(3)},
            "pred_class_counts": {CLASS_NAMES[i]: int(class_counts_pred[i]) for i in range(3)},
        },
        "confusion_matrix": confusion.tolist(),
        "per_true_class": class_stats,
    }

    with open(output_dir / "softmax_label_stats.json", "w", encoding="utf-8") as handle:
        json.dump(overall_stats, handle, indent=2)

    plot_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
    plot_histogram_true_confidence(conf_true, output_dir / "true_class_confidence_hist.png")
    plot_class_distribution(class_counts_true, class_counts_pred, output_dir / "class_distribution_true_vs_pred.png")
    plot_confidence_by_true_class(decoded_df, output_dir / "true_class_confidence_boxplot.png")
    plot_reliability_diagram(calibration_bins, output_dir / "reliability_diagram.png")


def collect_csv_files(input_path: Path, include_glob: str, exclude_globs: list[str]) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path]
    if input_path.is_dir():
        all_csv = sorted(input_path.rglob(include_glob))
        selected: list[Path] = []
        for csv_path in all_csv:
            rel_path = csv_path.relative_to(input_path).as_posix()
            if any(csv_path.match(pattern) or Path(rel_path).match(pattern) for pattern in exclude_globs):
                continue
            selected.append(csv_path)
        return selected
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode CSV cells with [softmax(3) + one-hot label(3)] and create stats/plots."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional override for TARGET_PATH. Can be one CSV file or a folder.",
    )
    parser.add_argument(
        "--include-glob",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV discovery when input-path is a directory.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[
            "analysis_outputs/**",
            "softmax_csv_analysis/**",
            ".venv/**",
            "tests/**",
        ],
        help="Glob pattern to exclude (repeat flag for multiple patterns).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("softmax_csv_analysis"),
        help="Output folder for JSON stats and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path if args.input_path is not None else TARGET_PATH

    csv_files = collect_csv_files(input_path, args.include_glob, args.exclude_glob)
    if not csv_files:
        print(f"No CSV files found from input path: {input_path}")
        print(f"include_glob={args.include_glob} exclude_glob={args.exclude_glob}")
        return

    all_decoded: list[dict[str, Any]] = []
    undecoded_total = 0

    for csv_path in tqdm(csv_files, desc="Decoding CSV files", unit="file"):
        decoded_rows, undecoded_cells = read_and_decode_csv(csv_path)
        all_decoded.extend(decoded_rows)
        undecoded_total += undecoded_cells

    if not all_decoded:
        print("No valid prediction/label cells could be decoded. Check string format in CSV cells.")
        return

    compute_and_save_statistics(all_decoded, args.output_dir, csv_files, undecoded_total)

    print("Analysis complete.")
    print(f"Input path used: {input_path}")
    print(f"CSV files processed: {len(csv_files)}")
    print(f"Decoded samples: {len(all_decoded)}")
    print(f"Undecoded cells: {undecoded_total}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
