#!/usr/bin/env python3
"""
Script to analyze recording durations for DORSI_PLANTAR_PART1/2 recordings
"""

import os
import glob
import csv
from pathlib import Path
from load_intan_rhd_format import read_data

def get_recording_duration(rhd_file):
    """
    Get recording duration in seconds from an RHD file.
    Returns (duration_seconds, num_samples, sample_rate) or None if error
    """
    try:
        data = read_data(rhd_file)

        if 't_amplifier' in data and len(data['t_amplifier']) > 0:
            duration = data['t_amplifier'][-1] - data['t_amplifier'][0]
            num_samples = len(data['t_amplifier'])
            sample_rate = data.get('amplifier_sample_rate', 30000)  # Intan default
            return duration, num_samples, sample_rate

        return None
    except Exception as e:
        print(f"Error reading {rhd_file}: {e}")
        return None

def analyze_dorsi_plantar_recordings():
    """
    Analyze all DORSI_PLANTAR recordings and save results to CSV
    """
    data_dir = Path("data")
    results = []

    dorsi_dirs = sorted(glob.glob(str(data_dir / "RAT*_DORSI_PLANTAR_PART*")))

    print(f"Found {len(dorsi_dirs)} DORSI_PLANTAR directories")

    for dorsi_dir in dorsi_dirs:
        rat_name = Path(dorsi_dir).name
        print(f"\nProcessing {rat_name}...")

        rhd_files = sorted(glob.glob(os.path.join(dorsi_dir, "*.rhd")))

        for rhd_file in rhd_files:
            filename = os.path.basename(rhd_file)
            print(f"  Reading {filename}...", end=" ", flush=True)

            duration_info = get_recording_duration(rhd_file)

            if duration_info:
                duration, num_samples, sample_rate = duration_info
                results.append({
                    'Rat': rat_name,
                    'File': filename,
                    'Duration_seconds': round(duration, 2),
                    'Duration_minutes': round(duration / 60, 2),
                    'Num_samples': num_samples,
                    'Sample_rate_Hz': sample_rate
                })
                print(f"✓ {duration:.2f}s ({duration/60:.2f}m)")
            else:
                print(f"✗ Error reading file")

    if results:
        output_file = "dorsi_plantar_recording_lengths.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Rat', 'File', 'Duration_seconds', 'Duration_minutes', 'Num_samples', 'Sample_rate_Hz'])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Results saved to {output_file}")

        print("\n=== SUMMARY ===")
        rats = {}
        for r in results:
            rat = r['Rat']
            if rat not in rats:
                rats[rat] = {'total_duration': 0, 'num_files': 0}
            rats[rat]['total_duration'] += r['Duration_seconds']
            rats[rat]['num_files'] += 1

        for rat in sorted(rats.keys()):
            total_dur = rats[rat]['total_duration']
            num_files = rats[rat]['num_files']
            print(f"{rat}: {total_dur:.2f}s ({total_dur/60:.2f}m) across {num_files} files")
    else:
        print("No recordings found!")

if __name__ == "__main__":
    analyze_dorsi_plantar_recordings()
