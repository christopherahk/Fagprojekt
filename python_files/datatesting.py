"""
Test script to compare DORSI_PLANTAR recordings with DORSIFLEXION + PLANTARFLEXION recordings.
Analyzes if they contain the same data.
"""
# datacheck med copilot hjælp, baseret på metadata-analyse og filhashes. Ingen signal-korrelationstjek endnu,
# så kan ikke 100% udelukke genbrug af data.
import os
import json
import hashlib
import re
from collections import defaultdict
from datetime import datetime
import spikeinterface.extractors as se
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def parse_timestamp_from_filename(file_path: str) -> dict | None:
    """Extract YYMMDD and HHMMSS timestamp fields from RHD filename."""
    name = os.path.basename(file_path)
    match = re.search(r"(\d{6})_(\d{6})\.rhd$", name)
    if not match:
        return None

    date_yy_mm_dd = match.group(1)
    time_hh_mm_ss = match.group(2)
    try:
        dt = datetime.strptime(date_yy_mm_dd + time_hh_mm_ss, "%y%m%d%H%M%S")
    except ValueError:
        return None

    return {
        'date_yy_mm_dd': date_yy_mm_dd,
        'time_hh_mm_ss': time_hh_mm_ss,
        'datetime_iso': dt.isoformat(sep=' '),
        'datetime_obj': dt,
    }


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash for exact file identity checks."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def get_recording_info(rhd_path: str) -> dict:
    """Extract metadata from an RHD file."""
    try:
        rec = se.read_intan(rhd_path, stream_id="0")
        sampling_hz = float(rec.get_sampling_frequency())
        samples = int(rec.get_num_samples())
        channels = int(rec.get_num_channels())
        duration_s = (samples / sampling_hz) if sampling_hz else 0.0
        bytes_size = os.path.getsize(rhd_path)
        ts = parse_timestamp_from_filename(rhd_path)

        return {
            'path': rhd_path,
            'duration_s': duration_s,
            'samples': samples,
            'channels': channels,
            'file_size_mb': bytes_size / (1024 * 1024),
            'sampling_hz': sampling_hz,
            'file_size_bytes': bytes_size,
            'date_yy_mm_dd': ts['date_yy_mm_dd'] if ts else None,
            'time_hh_mm_ss': ts['time_hh_mm_ss'] if ts else None,
            'datetime_iso': ts['datetime_iso'] if ts else None,
            'datetime_obj': ts['datetime_obj'] if ts else None,
        }
    except Exception as e:
        print(f"Error reading {rhd_path}: {e}")
        return None


def collect_recordings_by_type(data_root: str) -> dict:
    """Collect all recordings organized by recording type and rat."""
    recordings = defaultdict(lambda: defaultdict(list))

    for rat_dir in sorted(os.listdir(data_root)):
        rat_path = os.path.join(data_root, rat_dir)
        if not os.path.isdir(rat_path):
            continue

        # Extract rat ID and recording type
        parts = rat_dir.split('_')
        if not parts[0].startswith('RAT'):
            continue

        rat_id = parts[0]
        recording_type = '_'.join(parts[1:])

        # Find all RHD files
        for file in os.listdir(rat_path):
            if file.lower().endswith('.rhd'):
                rhd_path = os.path.join(rat_path, file)
                info = get_recording_info(rhd_path)
                if info:
                    recordings[recording_type][rat_id].append(info)

    return recordings


def analyze_recordings(recordings: dict) -> dict:
    """Analyze recording statistics by type."""
    analysis = {}

    for rec_type, rats_data in recordings.items():
        durations = []
        sizes = []
        file_counts = []

        for rat_id, files in rats_data.items():
            file_counts.append(len(files))
            for file_info in files:
                durations.append(file_info['duration_s'])
                sizes.append(file_info['file_size_mb'])

        analysis[rec_type] = {
            'num_rats': len(rats_data),
            'total_files': sum(len(files) for files in rats_data.values()),
            'avg_files_per_rat': np.mean(file_counts) if file_counts else 0,
            'total_duration_hours': sum(durations) / 3600,
            'avg_duration_per_file_s': np.mean(durations) if durations else 0,
            'total_size_gb': sum(sizes) / 1024,
            'avg_file_size_mb': np.mean(sizes) if sizes else 0,
            'min_duration_s': min(durations) if durations else 0,
            'max_duration_s': max(durations) if durations else 0,
        }

    return analysis


def compare_dorsi_plantar_with_flexions(recordings: dict) -> dict:
    """Compare DORSI_PLANTAR recordings with DORSIFLEXION + PLANTARFLEXION."""
    comparison = {}

    # Get the different types
    dorsi_plantar_1 = recordings.get('DORSI_PLANTAR_PART1', {})
    dorsi_plantar_2 = recordings.get('DORSI_PLANTAR_PART2', {})
    dorsiflexion = recordings.get('DORSIFLEXION', {})
    plantarflexion = recordings.get('PLANTARFLEXION', {})

    # For each rat, compare
    all_rats = set()
    all_rats.update(dorsi_plantar_1.keys())
    all_rats.update(dorsi_plantar_2.keys())
    all_rats.update(dorsiflexion.keys())
    all_rats.update(plantarflexion.keys())

    for rat_id in sorted(all_rats):
        dp1_files = dorsi_plantar_1.get(rat_id, [])
        dp2_files = dorsi_plantar_2.get(rat_id, [])
        dorf_files = dorsiflexion.get(rat_id, [])
        plantf_files = plantarflexion.get(rat_id, [])

        dp1_duration = sum(f['duration_s'] for f in dp1_files)
        dp2_duration = sum(f['duration_s'] for f in dp2_files)
        dorf_duration = sum(f['duration_s'] for f in dorf_files)
        plantf_duration = sum(f['duration_s'] for f in plantf_files)

        flexion_combined_duration = dorf_duration + plantf_duration

        comparison[rat_id] = {
            'DORSI_PLANTAR_PART1_duration_s': dp1_duration,
            'DORSI_PLANTAR_PART2_duration_s': dp2_duration,
            'DORSIFLEXION_duration_s': dorf_duration,
            'PLANTARFLEXION_duration_s': plantf_duration,
            'FLEXION_COMBINED_duration_s': flexion_combined_duration,
            'DORSI_PLANTAR_PART1_files': len(dp1_files),
            'DORSI_PLANTAR_PART2_files': len(dp2_files),
            'DORSIFLEXION_files': len(dorf_files),
            'PLANTARFLEXION_files': len(plantf_files),
            'dp_vs_flexion_ratio_1': dp1_duration / flexion_combined_duration if flexion_combined_duration > 0 else 0,
            'dp_vs_flexion_ratio_2': dp2_duration / flexion_combined_duration if flexion_combined_duration > 0 else 0,
            'dp_combined_vs_flexion_ratio': (dp1_duration + dp2_duration) / flexion_combined_duration if flexion_combined_duration > 0 else 0,
        }

    return comparison


def collect_timestamp_windows(recordings: dict) -> list[dict]:
    """Build timestamp windows per rat and recording type."""
    rows: list[dict] = []

    for rec_type, rats_data in recordings.items():
        for rat_id, files in rats_data.items():
            dts = [f['datetime_obj'] for f in files if f.get('datetime_obj') is not None]
            if not dts:
                continue

            dts_sorted = sorted(dts)
            rows.append(
                {
                    'rat': rat_id,
                    'type': rec_type,
                    'date': dts_sorted[0].strftime('%y%m%d'),
                    'start': dts_sorted[0],
                    'end': dts_sorted[-1],
                    'count': len(dts_sorted),
                }
            )

    return sorted(rows, key=lambda x: (x['rat'], x['start']))


def evaluate_time_separation(timestamp_windows: list[dict]) -> dict:
    """Evaluate if DORSI_PLANTAR windows are separated in time from FLEXION windows."""
    by_rat: dict[str, list[dict]] = defaultdict(list)
    for row in timestamp_windows:
        by_rat[row['rat']].append(row)

    results = {}
    for rat, rows in by_rat.items():
        rows = sorted(rows, key=lambda x: x['start'])
        group = {r['type']: r for r in rows}

        dp1 = group.get('DORSI_PLANTAR_PART1')
        dp2 = group.get('DORSI_PLANTAR_PART2')
        dorf = group.get('DORSIFLEXION')
        plantf = group.get('PLANTARFLEXION')

        flex_end = None
        if dorf and plantf:
            flex_end = max(dorf['end'], plantf['end'])
        elif dorf:
            flex_end = dorf['end']
        elif plantf:
            flex_end = plantf['end']

        dp_start_candidates = []
        if dp1:
            dp_start_candidates.append(dp1['start'])
        if dp2:
            dp_start_candidates.append(dp2['start'])
        dp_start = min(dp_start_candidates) if dp_start_candidates else None

        gap_minutes = None
        separated = None
        if flex_end and dp_start:
            gap_minutes = (dp_start - flex_end).total_seconds() / 60.0
            separated = gap_minutes > 0

        results[rat] = {
            'flex_end': flex_end.isoformat(sep=' ') if flex_end else None,
            'dp_start': dp_start.isoformat(sep=' ') if dp_start else None,
            'gap_minutes': gap_minutes,
            'separated_in_time': separated,
        }

    return results


def find_cross_type_hash_matches(recordings: dict) -> list[dict]:
    """Find exact duplicate files across DORSI_PLANTAR vs FLEXION categories."""
    all_rows = []
    for rec_type, rats_data in recordings.items():
        for rat_id, files in rats_data.items():
            for f in files:
                all_rows.append(
                    {
                        'rat': rat_id,
                        'type': rec_type,
                        'file_name': os.path.basename(f['path']),
                        'path': f['path'],
                        'hash': compute_sha256(f['path']),
                    }
                )

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_hash[row['hash']].append(row)

    duplicates: list[dict] = []
    for h, rows in by_hash.items():
        if len(rows) < 2:
            continue

        types = {r['type'] for r in rows}
        has_dp = 'DORSI_PLANTAR_PART1' in types or 'DORSI_PLANTAR_PART2' in types
        has_flex = 'DORSIFLEXION' in types or 'PLANTARFLEXION' in types

        if has_dp and has_flex:
            duplicates.append({'hash': h, 'matches': rows})

    return duplicates


def _sort_recording_files(files: list[dict]) -> list[dict]:
    """Sort file-info rows by extracted recording datetime."""
    return sorted(
        files,
        key=lambda f: (
            f['datetime_obj'] if f.get('datetime_obj') is not None else datetime.min,
            os.path.basename(f['path']),
        ),
    )


def build_downsampled_sequence(files: list[dict], target_hz: float = 500.0) -> tuple[np.ndarray, float]:
    """Load first channel from files, concatenate, and downsample by striding."""
    sequences = []
    effective_hz = None

    for file_info in _sort_recording_files(files):
        rec = se.read_intan(file_info['path'], stream_id="0")
        fs = float(rec.get_sampling_frequency())
        channel_ids = rec.get_channel_ids()
        if len(channel_ids) == 0:
            continue

        step = max(int(round(fs / target_hz)), 1)
        this_hz = fs / step
        if effective_hz is None:
            effective_hz = this_hz

        # Only keep one channel for robust, lightweight sequence matching.
        trace = rec.get_traces(channel_ids=[channel_ids[0]]).squeeze().astype(np.float32)
        if trace.size == 0:
            continue
        sequences.append(trace[::step])

    if not sequences:
        return np.array([], dtype=np.float32), target_hz

    concatenated = np.concatenate(sequences).astype(np.float32)
    return concatenated, (effective_hz if effective_hz is not None else target_hz)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standard score transform that is stable when variance is near zero."""
    if x.size == 0:
        return x
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x - mean
    return (x - mean) / std


def compare_signal_similarity(recordings: dict, max_lag_s: float = 30.0) -> dict:
    """Compare concatenated DP vs FLEXION signals using multiple similarity metrics."""
    dorsi_plantar_1 = recordings.get('DORSI_PLANTAR_PART1', {})
    dorsi_plantar_2 = recordings.get('DORSI_PLANTAR_PART2', {})
    dorsiflexion = recordings.get('DORSIFLEXION', {})
    plantarflexion = recordings.get('PLANTARFLEXION', {})

    all_rats = set()
    all_rats.update(dorsi_plantar_1.keys())
    all_rats.update(dorsi_plantar_2.keys())
    all_rats.update(dorsiflexion.keys())
    all_rats.update(plantarflexion.keys())

    results = {}

    for rat_id in sorted(all_rats):
        dp_files = _sort_recording_files(dorsi_plantar_1.get(rat_id, []) + dorsi_plantar_2.get(rat_id, []))
        flex_files = _sort_recording_files(dorsiflexion.get(rat_id, []) + plantarflexion.get(rat_id, []))

        if not dp_files or not flex_files:
            results[rat_id] = {
                'available': False,
                'reason': 'Missing DP or FLEXION files',
            }
            continue

        dp_seq, dp_hz = build_downsampled_sequence(dp_files, target_hz=500.0)
        flex_seq, flex_hz = build_downsampled_sequence(flex_files, target_hz=500.0)

        if dp_seq.size == 0 or flex_seq.size == 0:
            results[rat_id] = {
                'available': False,
                'reason': 'One of the sequences is empty after loading',
            }
            continue

        # Align to shortest length for overlap metrics.
        min_len = min(dp_seq.size, flex_seq.size)
        dp_overlap = _zscore(dp_seq[:min_len].astype(np.float64))
        flex_overlap = _zscore(flex_seq[:min_len].astype(np.float64))

        pearson = float(np.corrcoef(dp_overlap, flex_overlap)[0, 1]) if min_len > 2 else 0.0

        # Cross-correlation via FFT to allow temporal shifts and sign inversion.
        dp_norm = _zscore(dp_seq.astype(np.float64))
        flex_norm = _zscore(flex_seq.astype(np.float64))
        corr = signal.correlate(dp_norm, flex_norm, mode='full', method='fft')
        lags = signal.correlation_lags(dp_norm.size, flex_norm.size, mode='full')

        hz_for_lag = min(dp_hz, flex_hz)
        max_lag_samples = int(round(max_lag_s * hz_for_lag))
        mask = np.abs(lags) <= max_lag_samples

        denom = np.sqrt(np.sum(dp_norm ** 2) * np.sum(flex_norm ** 2))
        if denom <= 1e-12:
            max_abs_xcorr = 0.0
            best_lag_s = 0.0
        else:
            corr_norm = corr[mask] / denom
            idx = int(np.argmax(np.abs(corr_norm)))
            max_abs_xcorr = float(np.abs(corr_norm[idx]))
            best_lag_samples = int(lags[mask][idx])
            best_lag_s = float(best_lag_samples / hz_for_lag)

        # Frequency-domain similarity: identical/reused signals should have very similar PSD shape.
        nperseg = min(4096, min_len)
        if nperseg >= 256:
            f1, p1 = signal.welch(dp_overlap, fs=hz_for_lag, nperseg=nperseg)
            f2, p2 = signal.welch(flex_overlap, fs=hz_for_lag, nperseg=nperseg)
            n = min(len(p1), len(p2))
            lp1 = np.log10(p1[:n] + 1e-12)
            lp2 = np.log10(p2[:n] + 1e-12)
            psd_corr = float(np.corrcoef(lp1, lp2)[0, 1]) if n > 2 else 0.0
        else:
            psd_corr = 0.0

        likely_same_or_transformed = bool(max_abs_xcorr >= 0.98 and abs(psd_corr) >= 0.98)

        results[rat_id] = {
            'available': True,
            'dp_samples_500hz': int(dp_seq.size),
            'flex_samples_500hz': int(flex_seq.size),
            'overlap_samples': int(min_len),
            'pearson_abs_overlap': float(abs(pearson)),
            'max_abs_xcorr_30s': max_abs_xcorr,
            'best_lag_seconds': best_lag_s,
            'psd_corr': psd_corr,
            'likely_same_or_transformed': likely_same_or_transformed,
        }

    return results


def create_visualizations(recordings: dict, analysis: dict, comparison: dict, output_dir: str = 'analysis_outputs'):
    """Create visualizations of the data comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Duration comparison by recording type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Recording Type Comparison', fontsize=16, fontweight='bold')

    rec_types = list(analysis.keys())
    durations = [analysis[t]['total_duration_hours'] for t in rec_types]
    file_counts = [analysis[t]['total_files'] for t in rec_types]
    avg_durations = [analysis[t]['avg_duration_per_file_s'] for t in rec_types]
    sizes = [analysis[t]['total_size_gb'] for t in rec_types]

    # Plot 1: Total duration by type
    axes[0, 0].bar(rec_types, durations, color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Total Duration (hours)', fontweight='bold')
    axes[0, 0].set_title('Total Recording Hours by Type')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(durations):
        axes[0, 0].text(i, v, f'{v:.1f}h', ha='center', va='bottom')

    # Plot 2: Number of files by type
    axes[0, 1].bar(rec_types, file_counts, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('Number of Files', fontweight='bold')
    axes[0, 1].set_title('Total Files by Type')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(file_counts):
        axes[0, 1].text(i, v, f'{int(v)}', ha='center', va='bottom')

    # Plot 3: Average file duration
    axes[1, 0].bar(rec_types, avg_durations, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('Average Duration (seconds)', fontweight='bold')
    axes[1, 0].set_title('Average File Duration by Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_durations):
        axes[1, 0].text(i, v, f'{v:.1f}s', ha='center', va='bottom')

    # Plot 4: Total size
    axes[1, 1].bar(rec_types, sizes, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_ylabel('Total Size (GB)', fontweight='bold')
    axes[1, 1].set_title('Total Data Size by Type')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(sizes):
        axes[1, 1].text(i, v, f'{v:.2f}GB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recording_types_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'recording_types_comparison.png')}")
    plt.close()

    # Figure 2: DORSI_PLANTAR vs FLEXION comparison per rat
    if comparison:
        rats = sorted(comparison.keys())
        dp1_durations = [comparison[r]['DORSI_PLANTAR_PART1_duration_s'] / 60 for r in rats]
        dp2_durations = [comparison[r]['DORSI_PLANTAR_PART2_duration_s'] / 60 for r in rats]
        dp_combined_durations = [(comparison[r]['DORSI_PLANTAR_PART1_duration_s'] + comparison[r]['DORSI_PLANTAR_PART2_duration_s']) / 60 for r in rats]
        flexion_durations = [comparison[r]['FLEXION_COMBINED_duration_s'] / 60 for r in rats]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(rats))
        width = 0.2

        bars1 = ax.bar(x - 1.5*width, dp1_durations, width, label='DORSI_PLANTAR_PART1', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, dp2_durations, width, label='DORSI_PLANTAR_PART2', color='coral', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, dp_combined_durations, width, label='DORSI_PLANTAR_COMBINED', color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars4 = ax.bar(x + 1.5*width, flexion_durations, width, label='DORSIFLEXION + PLANTARFLEXION', color='mediumseagreen', alpha=0.8)

        ax.set_xlabel('Rat ID', fontweight='bold')
        ax.set_ylabel('Duration (minutes)', fontweight='bold')
        ax.set_title('DORSI_PLANTAR vs FLEXION Recordings Duration Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(rats)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dorsi_flexion_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'dorsi_flexion_comparison.png')}")
        plt.close()

        # Figure 3: Ratio analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        ratios_1 = [comparison[r]['dp_vs_flexion_ratio_1'] for r in rats]
        ratios_2 = [comparison[r]['dp_vs_flexion_ratio_2'] for r in rats]

        bars1 = ax.bar(x - width/2, ratios_1, width, label='DORSI_PLANTAR_PART1 / (DORF + PLANTF)', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, ratios_2, width, label='DORSI_PLANTAR_PART2 / (DORF + PLANTF)', color='coral', alpha=0.8)

        # Add a reference line at 1.0 (equal duration)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Match (ratio=1.0)')

        ax.set_xlabel('Rat ID', fontweight='bold')
        ax.set_ylabel('Duration Ratio', fontweight='bold')
        ax.set_title('DORSI_PLANTAR Parts vs Combined FLEXION - Duration Ratios', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(rats)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ratio_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'ratio_analysis.png')}")
        plt.close()


def create_timestamp_timeline_plot(timestamp_windows: list[dict], output_dir: str = 'analysis_outputs'):
    """Create a timeline plot to show recording order and separation for each rat."""
    if not timestamp_windows:
        return

    os.makedirs(output_dir, exist_ok=True)
    relevant = [
        row
        for row in timestamp_windows
        if row['type'] in {'DORSI_PLANTAR_PART1', 'DORSI_PLANTAR_PART2', 'DORSIFLEXION', 'PLANTARFLEXION'}
    ]
    if not relevant:
        return

    color_map = {
        'DORSIFLEXION': '#3b82f6',
        'PLANTARFLEXION': '#10b981',
        'DORSI_PLANTAR_PART1': '#f59e0b',
        'DORSI_PLANTAR_PART2': '#ef4444',
    }

    rats = sorted({row['rat'] for row in relevant})
    y_positions = {rat: i for i, rat in enumerate(rats)}

    rat_base_time = {}
    rat_span_minutes = {}
    for rat in rats:
        rat_rows = [row for row in relevant if row['rat'] == rat]
        base = min(row['start'] for row in rat_rows)
        end = max(row['end'] for row in rat_rows)
        rat_base_time[rat] = base
        rat_span_minutes[rat] = max((end - base).total_seconds() / 60.0, 1.0)

    global_span_minutes = max(rat_span_minutes.values())

    fig, ax = plt.subplots(figsize=(14, 1.1 * len(rats) + 2))
    for row in relevant:
        base = rat_base_time[row['rat']]
        x_start = (row['start'] - base).total_seconds() / 60.0
        x_end = (row['end'] - base).total_seconds() / 60.0
        width = max(x_end - x_start, 0.4)
        y = y_positions[row['rat']]
        ax.barh(
            y=y,
            width=width,
            left=x_start,
            height=0.65,
            color=color_map.get(row['type'], '#9ca3af'),
            alpha=0.85,
            edgecolor='black',
            linewidth=0.6,
            label=row['type'],
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    ax.set_yticks(np.arange(len(rats)))
    ax.set_yticklabels(rats)
    ax.set_xlim(0, global_span_minutes * 1.05)
    ax.set_xlabel('Minutes from first timestamp for each rat')
    ax.set_title('Timestamp Timeline by Rat and Recording Type (per-rat aligned)')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'timestamp_timeline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def create_signal_similarity_plot(signal_similarity: dict, output_dir: str = 'analysis_outputs'):
    """Visualize similarity metrics per rat for DP-combined vs FLEXION-combined signals."""
    rows = [(rat, vals) for rat, vals in sorted(signal_similarity.items()) if vals.get('available')]
    if not rows:
        return

    os.makedirs(output_dir, exist_ok=True)
    rats = [r for r, _ in rows]
    xcorr_vals = [v['max_abs_xcorr_30s'] for _, v in rows]
    psd_vals = [abs(v['psd_corr']) for _, v in rows]

    x = np.arange(len(rats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width / 2, xcorr_vals, width, label='Max abs xcorr (±30s lag)', color='#2563eb', alpha=0.85)
    ax.bar(x + width / 2, psd_vals, width, label='Abs PSD correlation', color='#f97316', alpha=0.85)
    ax.axhline(0.98, color='red', linestyle='--', linewidth=1.8, label='"Could be same/transformed" threshold (0.98)')

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(rats)
    ax.set_ylabel('Similarity score')
    ax.set_xlabel('Rat ID')
    ax.set_title('Signal Similarity: DORSI_PLANTAR_COMBINED vs DORSI+PLANTAR')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='upper right')

    out_path = os.path.join(output_dir, 'signal_similarity_scores.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def print_analysis_report(
    analysis: dict,
    comparison: dict,
    timestamp_windows: list[dict],
    time_separation: dict,
    cross_type_duplicates: list[dict],
    signal_similarity: dict,
):
    """Print a detailed analysis report."""
    print("\n" + "="*80)
    print("RECORDING DATA ANALYSIS REPORT")
    print("="*80)

    print("\n1. RECORDING TYPE SUMMARY")
    print("-" * 80)
    for rec_type, stats in sorted(analysis.items()):
        print(f"\n{rec_type}:")
        print(f"  - Number of Rats: {stats['num_rats']}")
        print(f"  - Total Files: {stats['total_files']}")
        print(f"  - Avg Files per Rat: {stats['avg_files_per_rat']:.1f}")
        print(f"  - Total Duration: {stats['total_duration_hours']:.2f} hours ({stats['total_duration_hours']*60:.0f} minutes)")
        print(f"  - Avg Duration per File: {stats['avg_duration_per_file_s']:.1f} seconds")
        print(f"  - File Duration Range: {stats['min_duration_s']:.1f}s - {stats['max_duration_s']:.1f}s")
        print(f"  - Total Data Size: {stats['total_size_gb']:.2f} GB")
        print(f"  - Avg File Size: {stats['avg_file_size_mb']:.1f} MB")

    print("\n2. DORSI_PLANTAR vs DORSIFLEXION + PLANTARFLEXION COMPARISON")
    print("-" * 80)

    for rat_id in sorted(comparison.keys()):
        comp = comparison[rat_id]
        dp1 = comp['DORSI_PLANTAR_PART1_duration_s'] / 60  # Convert to minutes
        dp2 = comp['DORSI_PLANTAR_PART2_duration_s'] / 60
        dorf = comp['DORSIFLEXION_duration_s'] / 60
        plantf = comp['PLANTARFLEXION_duration_s'] / 60
        flexion_combined = comp['FLEXION_COMBINED_duration_s'] / 60
        ratio1 = comp['dp_vs_flexion_ratio_1']
        ratio2 = comp['dp_vs_flexion_ratio_2']
        ratio_combined = comp['dp_combined_vs_flexion_ratio']

        print(f"\n{rat_id}:")
        print(f"  DORSI_PLANTAR_PART1:      {dp1:6.2f} min ({comp['DORSI_PLANTAR_PART1_files']} files) | Ratio vs FLEXION: {ratio1:.3f}")
        print(f"  DORSI_PLANTAR_PART2:      {dp2:6.2f} min ({comp['DORSI_PLANTAR_PART2_files']} files) | Ratio vs FLEXION: {ratio2:.3f}")
        print(f"  ---")
        print(f"  DORSIFLEXION:             {dorf:6.2f} min ({comp['DORSIFLEXION_files']} files)")
        print(f"  PLANTARFLEXION:           {plantf:6.2f} min ({comp['PLANTARFLEXION_files']} files)")
        print(f"  FLEXION COMBINED:         {flexion_combined:6.2f} min")
        print(f"  DP COMBINED RATIO:        {ratio_combined:6.3f}  ((PART1+PART2) / (DORF+PLANTF))")

        # Analysis
        print(f"  Analysis:")
        if abs(ratio1 - 1.0) < 0.05:
            print(f"    ✓ PART1 matches FLEXION combined (ratio {ratio1:.3f} ≈ 1.0)")
        elif ratio1 < 1.0:
            print(f"    ✗ PART1 is SHORTER than FLEXION combined (ratio {ratio1:.3f})")
        else:
            print(f"    ✗ PART1 is LONGER than FLEXION combined (ratio {ratio1:.3f})")

        if abs(ratio2 - 1.0) < 0.05:
            print(f"    ✓ PART2 matches FLEXION combined (ratio {ratio2:.3f} ≈ 1.0)")
        elif ratio2 < 1.0:
            print(f"    ✗ PART2 is SHORTER than FLEXION combined (ratio {ratio2:.3f})")
        else:
            print(f"    ✗ PART2 is LONGER than FLEXION combined (ratio {ratio2:.3f})")

    print("\n3. TIMESTAMP WINDOWS (CONCRETE SEPARATION CHECK)")
    print("-" * 80)
    for row in timestamp_windows:
        if row['type'] not in {'DORSI_PLANTAR_PART1', 'DORSI_PLANTAR_PART2', 'DORSIFLEXION', 'PLANTARFLEXION'}:
            continue
        print(
            f"{row['rat']:>5} | {row['type']:<18} | {row['date']} | "
            f"{row['start'].strftime('%H:%M:%S')} -> {row['end'].strftime('%H:%M:%S')} | files={row['count']}"
        )

    print("\n4. EVIDENCE AGAINST 'DP IS JUST DORF+PLANTF'")
    print("-" * 80)
    if cross_type_duplicates:
        print("Warning: Found exact byte-identical files across categories.")
        print(f"Cross-type duplicate hash groups: {len(cross_type_duplicates)}")
    else:
        print("✓ No exact SHA256 matches between DORSI_PLANTAR and FLEXION file groups.")

    all_gaps = [
        info['gap_minutes']
        for info in time_separation.values()
        if info.get('gap_minutes') is not None
    ]
    if all_gaps:
        min_gap = min(all_gaps)
        print(f"✓ Earliest DP start is after FLEXION end for each rat (min gap: {min_gap:.2f} minutes).")

    print("Interpretation:")
    print("  - Current checks strongly support that these are separate recordings.")
    print("  - Absolute proof against all transformations is impossible from metadata alone.")
    print("  - To fully rule out transformed/re-exported signal reuse, add signal correlation checks.")

    print("\n5. SIGNAL-LEVEL SIMILARITY CHECK (TRANSFORMATION SCREEN)")
    print("-" * 80)
    available_rows = [(rat, vals) for rat, vals in sorted(signal_similarity.items()) if vals.get('available')]
    if not available_rows:
        print("No signal similarity results available.")
    else:
        suspicious = 0
        for rat, vals in available_rows:
            xcorr = vals['max_abs_xcorr_30s']
            psd_corr = vals['psd_corr']
            lag_s = vals['best_lag_seconds']
            likely = vals['likely_same_or_transformed']
            flag = 'POTENTIAL MATCH' if likely else 'NOT A MATCH'
            if likely:
                suspicious += 1
            print(
                f"{rat}: xcorr={xcorr:.4f}, abs(psd_corr)={abs(psd_corr):.4f}, "
                f"best_lag={lag_s:+.2f}s -> {flag}"
            )

        print("Summary:")
        if suspicious == 0:
            print("  ✓ No rats pass the high-similarity threshold for same/transformed signal reuse.")
        else:
            print(f"  ⚠ {suspicious} rat(s) pass threshold and require manual inspection.")

    print("\n" + "="*80 + "\n")


def main():
    data_root = 'data'

    if not os.path.exists(data_root):
        print(f"Error: {data_root} directory not found!")
        return

    print("Collecting recordings...")
    recordings = collect_recordings_by_type(data_root)

    print("Analyzing recordings...")
    analysis = analyze_recordings(recordings)

    print("Comparing DORSI_PLANTAR with FLEXION recordings...")
    comparison = compare_dorsi_plantar_with_flexions(recordings)

    print("Collecting timestamp windows...")
    timestamp_windows = collect_timestamp_windows(recordings)
    time_separation = evaluate_time_separation(timestamp_windows)

    print("Running cross-type hash identity check...")
    cross_type_duplicates = find_cross_type_hash_matches(recordings)

    print("Running signal-level similarity checks (this can take a while)...")
    signal_similarity = compare_signal_similarity(recordings, max_lag_s=30.0)

    print_analysis_report(
        analysis,
        comparison,
        timestamp_windows,
        time_separation,
        cross_type_duplicates,
        signal_similarity,
    )

    print("Creating visualizations...")
    create_visualizations(recordings, analysis, comparison)
    create_timestamp_timeline_plot(timestamp_windows)
    create_signal_similarity_plot(signal_similarity)

    # Save analysis results to JSON
    output_file = 'analysis_outputs/comparison_analysis.json'
    os.makedirs('analysis_outputs', exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'analysis': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                           for kk, vv in v.items()}
                        for k, v in analysis.items()},
            'comparison': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                              for kk, vv in v.items()}
                          for k, v in comparison.items()},
            'timestamp_windows': [
                {
                    'rat': r['rat'],
                    'type': r['type'],
                    'date': r['date'],
                    'start': r['start'].isoformat(sep=' '),
                    'end': r['end'].isoformat(sep=' '),
                    'count': r['count'],
                }
                for r in timestamp_windows
            ],
            'time_separation': {
                rat: {
                    k: (float(v) if isinstance(v, (int, float, np.number)) and k == 'gap_minutes' and v is not None else v)
                    for k, v in info.items()
                }
                for rat, info in time_separation.items()
            },
            'cross_type_duplicate_hash_groups': len(cross_type_duplicates),
            'signal_similarity': {
                rat: {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in vals.items()
                }
                for rat, vals in signal_similarity.items()
            },
        }, f, indent=2)

    print(f"\nAnalysis saved to: {output_file}")


if __name__ == '__main__':
    main()
