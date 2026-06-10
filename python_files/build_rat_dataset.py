import gc
import os

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

from utils import read_rhd

data_folder = "./data"
output_folder = "./dataset_rats_50w"
RATS = list(range(4, 11))
WINDOW = int(0.05 * 30_000)  # 1 500 samples per 50 ms window


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def get_files(rat):
    part1 = f"{rat}_DORSI_PLANTAR_PART1"
    part2 = f"{rat}_DORSI_PLANTAR_PART2"
    rhd_files, mat_files = [], []
    for folder in (part1, part2):
        path = f"{data_folder}/{folder}"
        if not os.path.exists(path):
            print(f"  [skip] folder not found: {path}")
            continue
        for file in sorted(os.listdir(path)):
            if file.endswith(".rhd"):
                rhd_files.append(f"{path}/{file}")
            elif file.endswith(".mat"):
                mat_files.append(f"{path}/{file}")
    return rhd_files, mat_files


# ---------------------------------------------------------------------------
# Memory-efficient RMS helpers
# ---------------------------------------------------------------------------

def _rms_windows(data_f32: np.ndarray, n_windows: int, window_size: int) -> np.ndarray:
    """RMS-downsample (n_channels, n_windows*window_size) → (n_channels, n_windows).

    Processes one channel at a time so the only extra allocation is
    ~n_windows * window_size * 4 bytes (one channel's squared windows),
    not n_channels × that amount.
    """
    n_channels = data_f32.shape[0]
    rms = np.empty((n_channels, n_windows), dtype=np.float32)
    for ch in range(n_channels):
        # data_f32[ch] is a contiguous 1-D view → reshape is a zero-copy view.
        wins = data_f32[ch, : n_windows * window_size].reshape(n_windows, window_size)
        rms[ch] = np.sqrt(np.mean(wins ** 2, axis=1))
    return rms


def compute_rms_streaming(
    rhd_files: list,
    t_start: float,
    t_end: float,
    window_size: int,
):
    """Stream RHD files one at a time and accumulate RMS windows.

    Peak memory ≈ one RHD file in float32, instead of all files concatenated.
    Partial windows at file boundaries are carried over to the next file.
    """
    rms_blocks: list[np.ndarray] = []
    t_blocks: list[np.ndarray] = []
    leftover_data: np.ndarray | None = None
    leftover_times: np.ndarray | None = None

    for file in rhd_files:
        print(f"  loading {file}")
        result = read_rhd(file)

        # Cast to float32 immediately (halves memory vs float64; copy=False avoids
        # a redundant allocation when the data is already float32).
        data = result["amplifier_data"].astype(np.float32, copy=False)
        times = result["t_amplifier"]
        del result          # free everything else in the dict
        gc.collect()

        # ── time-range mask ──────────────────────────────────────────────────
        mask = (times >= t_start) & (times <= t_end)
        if not np.any(mask):
            del data, times
            continue

        data = data[:, mask]    # fancy indexing → new contiguous array; old freed below
        times = times[mask]

        # ── stitch leftover from previous file boundary ──────────────────────
        if leftover_data is not None:
            data = np.concatenate([leftover_data, data], axis=1)
            times = np.concatenate([leftover_times, times])
            leftover_data = leftover_times = None

        n_samples = data.shape[1]
        n_complete = n_samples // window_size

        if n_complete > 0:
            rms_blocks.append(_rms_windows(data, n_complete, window_size))
            mid = window_size // 2
            # [:, mid] is a strided view of `times`, so .copy() detaches it
            # before we delete `times` below.
            t_blocks.append(
                times[: n_complete * window_size]
                .reshape(n_complete, window_size)[:, mid]
                .copy()
            )

        # ── save sub-window tail for next iteration ──────────────────────────
        tail_start = n_complete * window_size
        if tail_start < n_samples:
            leftover_data = data[:, tail_start:].copy()
            leftover_times = times[tail_start:].copy()

        del data, times
        gc.collect()

    if not rms_blocks:
        return None, None

    return np.concatenate(rms_blocks, axis=1), np.concatenate(t_blocks)


# ---------------------------------------------------------------------------
# MAT loader (unchanged — angle files are small)
# ---------------------------------------------------------------------------

def load_mat(mat_files: list):
    all_data, all_times = [], []
    offset = 0.0
    for file in mat_files:
        mat = sio.loadmat(file)
        ta = mat["time_angle"]
        t = ta[:, 0]
        all_data.append(ta[:, 1])
        all_times.append(t + offset)
        offset += t[-1] + (t[1] - t[0])
    return np.concatenate(all_data), np.concatenate(all_times)


# ---------------------------------------------------------------------------
# Per-rat processing
# ---------------------------------------------------------------------------

def process_rat(rat: str) -> None:
    print(f"\nProcessing {rat}...")
    rhd_files, mat_files = get_files(rat)

    if not rhd_files:
        print(f"  [skip] no .rhd files found for {rat}")
        return
    if not mat_files:
        print(f"  [skip] no .mat files found for {rat}")
        return

    print(f"  found {len(rhd_files)} .rhd and {len(mat_files)} .mat files")

    t_start, t_end = 20.0, 385.0

    # Stream RHD files — never materialise the full concatenated raw signal.
    rms_data, times_ds = compute_rms_streaming(rhd_files, t_start, t_end, WINDOW)
    if rms_data is None:
        print("  [skip] no RHD data found within the time range")
        return

    mat_data, mat_t = load_mat(mat_files)
    mat_mask = (mat_t >= t_start) & (mat_t <= t_end)

    total_duration = 389.97
    files_p1 = 4
    total_files = 7
    p1_duration = (total_duration / total_files) * files_p1
    effective_p1_duration = p1_duration - t_start
    split_idx = int(effective_p1_duration / 0.05)

    f = interp1d(
        mat_t[mat_mask],
        mat_data[mat_mask],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    angles_ds = f(times_ds)

    os.makedirs(output_folder, exist_ok=True)
    out_path = f"{output_folder}/{rat}.npz"
    np.savez(out_path, t=times_ds, a=angles_ds, rms_data=rms_data, split_idx=split_idx)
    print(f"  saved -> {out_path}  (rms_data shape: {rms_data.shape})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Working directory: {os.path.abspath('.')}")
    print(f"Looking for data in: {os.path.abspath(data_folder)}")
    for r in RATS:
        process_rat(f"RAT{r}")
        gc.collect()        # release between rats
    print("\nDone.")
