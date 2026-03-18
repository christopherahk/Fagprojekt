import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import spikeinterface.extractors as se


@dataclass
class RecordingInfo:
    path: str
    rat: str
    label: str
    channels: int
    sampling_hz: float
    samples: int
    duration_s: float
    bytes_size: int


def parse_rat_and_label(file_path: str) -> tuple[str, str]:
    parent = os.path.basename(os.path.dirname(file_path))
    parts = parent.split("_")
    rat = parts[0] if parts else "UNKNOWN"
    label = "_".join(parts[1:]) if len(parts) > 1 else parent
    return rat, label


def collect_rhd_paths(root: str) -> list[str]:
    paths: list[str] = []
    for dir_path, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".rhd"):
                paths.append(os.path.join(dir_path, name))
    return sorted(paths)


def load_recording_info(path: str, stream_id: str) -> RecordingInfo:
    rec = se.read_intan(path, stream_id=stream_id)
    sampling_hz = float(rec.get_sampling_frequency())
    samples = int(rec.get_num_samples())
    channels = int(rec.get_num_channels())
    duration_s = (samples / sampling_hz) if sampling_hz else 0.0
    bytes_size = os.path.getsize(path)
    rat, label = parse_rat_and_label(path)

    return RecordingInfo(
        path=path,
        rat=rat,
        label=label,
        channels=channels,
        sampling_hz=sampling_hz,
        samples=samples,
        duration_s=duration_s,
        bytes_size=bytes_size,
    )


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1))
    return sorted_values[idx]


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze(data_root: str, stream_id: str, short_limit_s: float, very_short_limit_s: float) -> None:
    paths = collect_rhd_paths(data_root)
    if not paths:
        print(f"No .rhd files found under: {data_root}")
        return

    infos: list[RecordingInfo] = []
    failed: list[tuple[str, str]] = []

    for path in paths:
        try:
            infos.append(load_recording_info(path, stream_id=stream_id))
        except Exception as exc:  # Keep analysis running if a single file fails.
            failed.append((path, str(exc)))

    if not infos:
        print("No files could be loaded successfully.")
        if failed:
            print("First error:")
            print(f"  {failed[0][0]} :: {failed[0][1]}")
        return

    durations = sorted(x.duration_s for x in infos)
    fs_counter = Counter(x.sampling_hz for x in infos)
    ch_counter = Counter(x.channels for x in infos)
    total_bytes = sum(x.bytes_size for x in infos)

    print_header("DATASET OVERVIEW")
    print(f"Data root: {data_root}")
    print(f"Loaded .rhd files: {len(infos)}")
    print(f"Failed files: {len(failed)}")
    print(f"Total size (bytes): {total_bytes}")
    print(f"Total size (GB): {total_bytes / (1024 ** 3):.3f}")
    print(f"Total duration (minutes): {sum(durations) / 60:.2f}")
    print(f"Total duration (hours): {sum(durations) / 3600:.3f}")
    print(f"Sampling frequencies: {dict(fs_counter)}")
    print(f"Channel counts: {dict(ch_counter)}")

    print_header("DURATION SUMMARY (seconds)")
    print(f"Min: {min(durations):.3f}")
    print(f"P10: {percentile(durations, 0.10):.3f}")
    print(f"P50: {percentile(durations, 0.50):.3f}")
    print(f"P90: {percentile(durations, 0.90):.3f}")
    print(f"Max: {max(durations):.3f}")

    by_label_count = Counter(x.label for x in infos)
    by_label_bytes = defaultdict(int)
    by_label_seconds = defaultdict(float)
    for x in infos:
        by_label_bytes[x.label] += x.bytes_size
        by_label_seconds[x.label] += x.duration_s

    print_header("BY LABEL")
    print("Label\tCount\tGB\tMinutes\tHours")
    for label in sorted(by_label_count):
        count = by_label_count[label]
        gb = by_label_bytes[label] / (1024 ** 3)
        minutes = by_label_seconds[label] / 60
        hours = by_label_seconds[label] / 3600
        print(f"{label}\t{count}\t{gb:.3f}\t{minutes:.2f}\t{hours:.3f}")

    by_rat_count = Counter(x.rat for x in infos)
    by_rat_bytes = defaultdict(int)
    by_rat_seconds = defaultdict(float)
    for x in infos:
        by_rat_bytes[x.rat] += x.bytes_size
        by_rat_seconds[x.rat] += x.duration_s

    print_header("BY RAT")
    print("Rat\tCount\tGB\tMinutes")
    for rat in sorted(by_rat_count):
        count = by_rat_count[rat]
        gb = by_rat_bytes[rat] / (1024 ** 3)
        minutes = by_rat_seconds[rat] / 60
        print(f"{rat}\t{count}\t{gb:.3f}\t{minutes:.2f}")

    short_files = [x for x in infos if x.duration_s < short_limit_s]
    very_short_files = [x for x in infos if x.duration_s < very_short_limit_s]

    print_header("SHORT RECORDINGS")
    print(f"Files shorter than {short_limit_s:.1f}s: {len(short_files)}")
    print(f"Files shorter than {very_short_limit_s:.1f}s: {len(very_short_files)}")

    short_by_label = Counter(x.label for x in short_files)
    if short_by_label:
        print("Short files by label:")
        for label in sorted(short_by_label):
            print(f"  {label}: {short_by_label[label]}")

    if short_files:
        print("Examples (up to 20):")
        for x in sorted(short_files, key=lambda i: i.duration_s)[:20]:
            print(
                f"  {x.duration_s:7.3f}s | ch={x.channels:2d} | {x.rat:>5s} | {x.label:<20s} | {os.path.basename(x.path)}"
            )

    if failed:
        print_header("FAILED FILES")
        for path, err in failed[:20]:
            print(f"{path} :: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze original .rhd dataset statistics.")
    parser.add_argument("--data-root", default="data", help="Path to folder with .rhd files.")
    parser.add_argument("--stream-id", default="0", help="Intan stream_id to load.")
    parser.add_argument("--short-limit", type=float, default=59.0, help="Short recording threshold in seconds.")
    parser.add_argument("--very-short-limit", type=float, default=10.0, help="Very short recording threshold in seconds.")
    args = parser.parse_args()

    analyze(
        data_root=args.data_root,
        stream_id=args.stream_id,
        short_limit_s=args.short_limit,
        very_short_limit_s=args.very_short_limit,
    )


if __name__ == "__main__":
    main()
