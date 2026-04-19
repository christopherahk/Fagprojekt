import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime

import spikeinterface.extractors as se


@dataclass
class RecordingMeta:
    file_path: str
    rat: str
    label: str
    channels: int
    sampling_hz: float
    num_samples: int
    duration_s: float
    file_size_bytes: int


def collect_rhd_files(data_root: str) -> list[str]:
    rhd_files: list[str] = []
    for dir_path, _, file_names in os.walk(data_root):
        for file_name in file_names:
            if file_name.lower().endswith('.rhd'):
                rhd_files.append(os.path.join(dir_path, file_name))
    return sorted(rhd_files)


def parse_rat_label(file_path: str) -> tuple[str, str]:
    parent = os.path.basename(os.path.dirname(file_path))
    parts = parent.split('_')
    rat = parts[0] if parts else 'UNKNOWN'
    label = '_'.join(parts[1:]) if len(parts) > 1 else parent
    return rat, label


def load_metadata(file_path: str, stream_id: str) -> RecordingMeta:
    rec = se.read_intan(file_path, stream_id=stream_id)
    sampling_hz = float(rec.get_sampling_frequency())
    num_samples = int(rec.get_num_samples())
    channels = int(rec.get_num_channels())
    duration_s = (num_samples / sampling_hz) if sampling_hz else 0.0
    rat, label = parse_rat_label(file_path)
    file_size_bytes = os.path.getsize(file_path)

    return RecordingMeta(
        file_path=file_path,
        rat=rat,
        label=label,
        channels=channels,
        sampling_hz=sampling_hz,
        num_samples=num_samples,
        duration_s=duration_s,
        file_size_bytes=file_size_bytes,
    )


def percentiles(sorted_values: list[float]) -> dict[str, float]:
    if not sorted_values:
        return {'p10': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0}

    def pick(p: float) -> float:
        idx = int(p * (len(sorted_values) - 1))
        return float(sorted_values[idx])

    return {
        'p10': pick(0.10),
        'p25': pick(0.25),
        'p50': pick(0.50),
        'p75': pick(0.75),
        'p90': pick(0.90),
    }


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p10': 0.0,
            'p25': 0.0,
            'p50': 0.0,
            'p75': 0.0,
            'p90': 0.0,
        }

    values_sorted = sorted(values)
    pct = percentiles(values_sorted)
    std = statistics.stdev(values) if len(values) > 1 else 0.0

    return {
        'count': len(values),
        'mean': float(statistics.mean(values)),
        'median': float(statistics.median(values)),
        'std': float(std),
        'min': float(min(values)),
        'max': float(max(values)),
        **pct,
    }


def write_csv(file_path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_rows_for_group_stats(group_name: str, stats_map: dict[str, dict[str, float]]) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(stats_map.keys()):
        item = {group_name: key}
        item.update(stats_map[key])
        rows.append(item)
    return rows


def run_analysis(data_root: str, stream_id: str, output_dir: str, short_limit: float, very_short_limit: float) -> None:
    rhd_files = collect_rhd_files(data_root)
    if not rhd_files:
        print(f'No .rhd files found under: {data_root}')
        return

    metadata: list[RecordingMeta] = []
    failed: list[dict] = []

    for file_path in rhd_files:
        try:
            metadata.append(load_metadata(file_path, stream_id=stream_id))
        except Exception as exc:
            failed.append({'file_path': file_path, 'error': str(exc)})

    durations = [m.duration_s for m in metadata]
    sample_counts = [float(m.num_samples) for m in metadata]
    file_sizes = [float(m.file_size_bytes) for m in metadata]
    channels = [float(m.channels) for m in metadata]

    by_label_durations: defaultdict[str, list[float]] = defaultdict(list)
    by_rat_durations: defaultdict[str, list[float]] = defaultdict(list)
    for m in metadata:
        by_label_durations[m.label].append(m.duration_s)
        by_rat_durations[m.rat].append(m.duration_s)

    label_stats = {label: summary_stats(vals) for label, vals in by_label_durations.items()}
    rat_stats = {rat: summary_stats(vals) for rat, vals in by_rat_durations.items()}

    channel_distribution = Counter(int(m.channels) for m in metadata)
    sampling_distribution = Counter(m.sampling_hz for m in metadata)

    short_files = [m for m in metadata if m.duration_s < short_limit]
    very_short_files = [m for m in metadata if m.duration_s < very_short_limit]

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'data_root': data_root,
        'stream_id': stream_id,
        'total_rhd_files_found': len(rhd_files),
        'total_loaded_files': len(metadata),
        'failed_files': len(failed),
        'overall_stats': {
            'duration_seconds': summary_stats(durations),
            'num_samples': summary_stats(sample_counts),
            'file_size_bytes': summary_stats(file_sizes),
            'channels': summary_stats(channels),
        },
        'total_duration_minutes': sum(durations) / 60 if durations else 0.0,
        'total_duration_hours': sum(durations) / 3600 if durations else 0.0,
        'total_size_gb': (sum(file_sizes) / (1024 ** 3)) if file_sizes else 0.0,
        'channel_distribution': dict(channel_distribution),
        'sampling_distribution_hz': dict(sampling_distribution),
        'short_threshold_seconds': short_limit,
        'very_short_threshold_seconds': very_short_limit,
        'short_file_count': len(short_files),
        'very_short_file_count': len(very_short_files),
        'short_files_by_label': dict(Counter(m.label for m in short_files)),
        'label_duration_stats_seconds': label_stats,
        'rat_duration_stats_seconds': rat_stats,
    }

    os.makedirs(output_dir, exist_ok=True)

    summary_json_path = os.path.join(output_dir, 'descriptive_stats_summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    failed_json_path = os.path.join(output_dir, 'failed_files.json')
    with open(failed_json_path, 'w', encoding='utf-8') as handle:
        json.dump(failed, handle, indent=2)

    label_csv_rows = build_rows_for_group_stats('label', label_stats)
    rat_csv_rows = build_rows_for_group_stats('rat', rat_stats)

    stats_fields = ['count', 'mean', 'median', 'std', 'min', 'max', 'p10', 'p25', 'p50', 'p75', 'p90']
    write_csv(
        os.path.join(output_dir, 'label_duration_stats_seconds.csv'),
        label_csv_rows,
        ['label'] + stats_fields,
    )
    write_csv(
        os.path.join(output_dir, 'rat_duration_stats_seconds.csv'),
        rat_csv_rows,
        ['rat'] + stats_fields,
    )

    write_csv(
        os.path.join(output_dir, 'channel_distribution.csv'),
        [{'channels': key, 'file_count': channel_distribution[key]} for key in sorted(channel_distribution)],
        ['channels', 'file_count'],
    )
    write_csv(
        os.path.join(output_dir, 'sampling_distribution_hz.csv'),
        [{'sampling_hz': key, 'file_count': sampling_distribution[key]} for key in sorted(sampling_distribution)],
        ['sampling_hz', 'file_count'],
    )

    short_rows = [
        {
            'file_path': m.file_path,
            'rat': m.rat,
            'label': m.label,
            'channels': m.channels,
            'duration_s': round(m.duration_s, 6),
            'num_samples': m.num_samples,
            'sampling_hz': m.sampling_hz,
        }
        for m in sorted(short_files, key=lambda x: x.duration_s)
    ]
    write_csv(
        os.path.join(output_dir, 'short_files.csv'),
        short_rows,
        ['file_path', 'rat', 'label', 'channels', 'duration_s', 'num_samples', 'sampling_hz'],
    )

    print('Analysis complete.')
    print(f'Loaded files: {len(metadata)} / {len(rhd_files)}')
    print(f'Failed files: {len(failed)}')
    print(f'Output folder: {output_dir}')
    print(f'Total duration (hours): {summary["total_duration_hours"]:.3f}')
    print(f'Channel distribution: {dict(channel_distribution)}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute descriptive statistics for .rhd dataset metadata.')
    parser.add_argument('--data-root', default='data', help='Root folder to scan recursively for .rhd files.')
    parser.add_argument('--stream-id', default='0', help='Intan stream_id used when loading .rhd files.')
    parser.add_argument('--output-dir', default='analysis_outputs', help='Folder where result files are written.')
    parser.add_argument('--short-limit', type=float, default=59.0, help='Threshold in seconds for short files.')
    parser.add_argument('--very-short-limit', type=float, default=10.0, help='Threshold in seconds for very short files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        data_root=args.data_root,
        stream_id=args.stream_id,
        output_dir=args.output_dir,
        short_limit=args.short_limit,
        very_short_limit=args.very_short_limit,
    )


if __name__ == '__main__':
    main()
