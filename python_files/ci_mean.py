import numpy as np
import matplotlib.pyplot as plt
from load_intan_rhd_format import read_data
import os

DATA_DIR = "./data"
RATS = [f"RAT{r}" for r in range(4, 11)]
WINDOW = int(0.05 * 30_000)
N_BOOT = 10_000
CI_LEVEL = 95
ALPHA = (100 - CI_LEVEL) / 2
RNG = np.random.default_rng(42)

def get_files(rat):
    part1 = f"{rat}_DORSI_PLANTAR_PART1"
    part2 = f"{rat}_DORSI_PLANTAR_PART2"
    rhd_files = []
    for folder in (part1, part2):
        path = f"{DATA_DIR}/{folder}"
        if not os.path.exists(path):
            continue
        for file in sorted(os.listdir(path)):
            if file.endswith(".rhd"):
                rhd_files.append(f"{path}/{file}")
    return rhd_files

def load_rhd(rhd_files):
    all_data, all_times = [], []
    for file in rhd_files:
        result = read_data(file)
        all_data.append(result['amplifier_data'])
        all_times.append(result['t_amplifier'])
    return np.concatenate(all_data, axis=1), np.concatenate(all_times)

def downsample_rms(data, times, window_size):
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_size
    data = data[:, :n_windows * window_size]
    data_wins = data.reshape(n_channels, n_windows, window_size)
    return np.sqrt(np.mean(data_wins**2, axis=2))

rat_means = []
loaded = []

for rat in RATS:
    rhd_files = get_files(rat)
    if not rhd_files:
        print(f"[skip] {rat}: no .rhd files found")
        continue

    rhd_data, rhd_t = load_rhd(rhd_files)

    t_start, t_end = 20.0, 385.0
    mask = (rhd_t >= t_start) & (rhd_t <= t_end)
    rms = downsample_rms(rhd_data[:, mask], rhd_t[mask], WINDOW)

    rat_means.append(rms.mean(axis=1))
    loaded.append(rat)
    print(f"{rat}: {rms.shape[0]} channels, {rms.shape[1]} windows")

n_ch = min(m.shape[0] for m in rat_means)
rat_means = np.array([m[:n_ch] for m in rat_means])
n_rats, n_ch = rat_means.shape
print(f"\nLoaded {n_rats} rats, truncated to {n_ch} channels each.")

boot_idx = RNG.integers(0, n_rats, size=(N_BOOT, n_rats))
boot_means = rat_means[boot_idx].mean(axis=1)

grand_mean = rat_means.mean(axis=0)
ci_lo = np.percentile(boot_means, ALPHA, axis=0)
ci_hi = np.percentile(boot_means, 100 - ALPHA, axis=0)

print(f"\n{CI_LEVEL}% bootstrap CI for per-channel RMS mean (n={n_rats} rats, {N_BOOT} resamples)")
print(f"{'Ch':>4}  {'Mean':>10}  {'CI lo':>10}  {'CI hi':>10}  {'Width':>10}")
print("-" * 50)
for ch in range(n_ch):
    print(f"{ch:>4}  {grand_mean[ch]:>10.4f}  {ci_lo[ch]:>10.4f}  "
          f"{ci_hi[ch]:>10.4f}  {ci_hi[ch]-ci_lo[ch]:>10.4f}")

np.savez(
    "ci_rms_results.npz",
    rats=loaded,
    rat_means=rat_means,
    grand_mean=grand_mean,
    ci_lo=ci_lo,
    ci_hi=ci_hi,
)

channels = np.arange(n_ch)
yerr = np.array([grand_mean - ci_lo, ci_hi - grand_mean])

fig, ax = plt.subplots(figsize=(max(8, n_ch * 0.5), 4))

ax.errorbar(
    channels, grand_mean, yerr=yerr,
    fmt="o", color="#2563eb", ecolor="#93c5fd",
    elinewidth=2, capsize=4, capthick=1.5,
    markersize=5, label=f"Mean ± {CI_LEVEL}% bootstrap CI"
)

for i, rat in enumerate(loaded):
    ax.plot(channels, rat_means[i], alpha=0.25, linewidth=0.8, color="gray")

ax.set_xlabel("Channel")
ax.set_ylabel("RMS amplitude")
ax.set_title(f"Per-channel RMS mean across rats ({CI_LEVEL}% bootstrap CI, n={n_rats})")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("ci_rms_plot.png", dpi=150)
