import numpy as np
import scipy.io as sio
from load_intan_rhd_format import read_data
import os
from scipy.interpolate import interp1d

data_folder = "./data"

RATS = list(range(4, 11))

WINDOW = int(0.05 * 30_000)


def get_files(rat):
    part1 = f"{rat}_DORSI_PLANTAR_PART1"
    part2 = f"{rat}_DORSI_PLANTAR_PART2"

    rhd_files, mat_files = [], []

    for folder in (part1, part2):
        path = f"{data_folder}/{folder}"

        if not os.path.exists(path):
            continue

        for file in sorted(os.listdir(path)):
            if file.endswith(".rhd"):
                rhd_files.append(f"{path}/{file}")
            elif file.endswith(".mat"):
                mat_files.append(f"{path}/{file}")

    return rhd_files, mat_files


def load_rhd(rhd_files):
    all_data, all_times = [], []

    for file in rhd_files:
        result = read_data(file)
        all_data.append(result['amplifier_data'])
        all_times.append(result['t_amplifier'])

    return np.concatenate(all_data, axis=1), np.concatenate(all_times)


def load_mat(mat_files):
    all_data, all_times = [], []
    offset = 0.0

    for file in mat_files:
        mat = sio.loadmat(file)
        ta = mat['time_angle']

        t = ta[:, 0]
        all_data.append(ta[:, 1])
        all_times.append(t + offset)

        offset += t[-1] + (t[1] - t[0])

    return np.concatenate(all_data), np.concatenate(all_times)


def downsample_rms(data, times, window_size):
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_size

    data = data[:, :n_windows * window_size]
    times = times[:n_windows * window_size]

    data_wins = data.reshape(n_channels, n_windows, window_size)
    rms = np.sqrt(np.mean(data_wins**2, axis=2))

    times_wins = times.reshape(n_windows, window_size)[:, window_size // 2]

    return rms, times_wins


def process_rat(rat):
    rhd_files, mat_files = get_files(rat)

    if len(rhd_files) == 0 or len(mat_files) == 0:
        return

    rhd_data, rhd_t = load_rhd(rhd_files)
    mat_data, mat_t = load_mat(mat_files)

    t_start, t_end = 20.0, 385.0

    rhd_mask = (rhd_t >= t_start) & (rhd_t <= t_end)
    mat_mask = (mat_t >= t_start) & (mat_t <= t_end)

    rms_data, times_ds = downsample_rms(
        rhd_data[:, rhd_mask],
        rhd_t[rhd_mask],
        WINDOW
    )

    total_duration = 389.97
    files_p1 = 4
    total_files = 7
    p1_duration = (total_duration / total_files) * files_p1
    effective_p1_duration = p1_duration - t_start

    split_idx = int(effective_p1_duration / 0.05)

    f = interp1d(
        mat_t[mat_mask],
        mat_data[mat_mask],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    angles_ds = f(times_ds)

    np.savez(
        f"dataset_rats_test/{rat}.npz",
        t=times_ds,
        a=angles_ds,
        rms_data=rms_data,
        split_idx=split_idx
    )


if __name__ == "__main__":
    for r in RATS:
        process_rat(f"RAT{r}")
