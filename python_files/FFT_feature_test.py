"""
FFT feature evaluation
======================
Tests whether adding FFT power spectrum features improves
leave-one-rat-out accuracy compared to the current 6-feature baseline.

Place this file next to streamer.py and run it directly.
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

try:
    from streamer import load_rat, prepare_dataset
except ImportError:
    raise ImportError("Place this file next to streamer.py")


CLASS_NAMES = ["ingen", "plantar", "dorsi"]
DATA_DIR    = Path(__file__).resolve().parent.parent / "dataset_rats_50w"
RAT_IDS     = list(range(4, 10))


# ── data loading ─────────────────────────────────────────────────────────────

def load_dataset():
    Xs, ys, groups = [], [], []
    for r in RAT_IDS:
        path = DATA_DIR / f"rat{r}.npz"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        rms, ang = load_rat(str(path))
        Xr, yr   = prepare_dataset(rms, ang)
        Xs.append(Xr)
        ys.append(yr)
        groups.extend([r] * len(yr))
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups)


# ── feature sets ─────────────────────────────────────────────────────────────

def features_baseline(X: np.ndarray) -> np.ndarray:
    """
    6 time-domain features per channel (current Arduino setup):
      mean, std, RMS, max-abs, zero-crossing rate, waveform length (log1p)
    Output shape: (n, channels * 6)
    """
    n, ch, w = X.shape
    out = np.zeros((n, ch * 6), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg    = X[i, c, :]
            mean   = seg.mean()
            std    = seg.std()
            rms    = np.sqrt((seg ** 2).mean())
            maxabs = np.abs(seg).max()
            zc     = np.sum(np.diff(np.sign(seg)) != 0) / w
            wl     = np.log1p(np.sum(np.abs(np.diff(seg))))
            out[i, c*6:(c+1)*6] = [mean, std, rms, maxabs, zc, wl]
    return out


def features_fft_only(X: np.ndarray) -> np.ndarray:
    """
    FFT power spectrum per channel.
    Window length w gives w//2 + 1 frequency bins.
    Output shape: (n, channels * (w//2 + 1))
    """
    n, ch, w  = X.shape
    n_bins    = w // 2 + 1
    out       = np.zeros((n, ch * n_bins), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg = X[i, c, :]
            # Apply Hanning window to reduce spectral leakage
            windowed  = seg * np.hanning(w)
            fft_power = np.abs(np.fft.rfft(windowed)) ** 2
            out[i, c*n_bins:(c+1)*n_bins] = fft_power
    return out


def features_combined(X: np.ndarray) -> np.ndarray:
    """
    Time-domain (6) + FFT power (w//2+1) per channel.
    This is the full feature set -- tests whether FFT adds value on top
    of the existing features.
    Output shape: (n, channels * (6 + w//2 + 1))
    """
    n, ch, w  = X.shape
    n_bins    = w // 2 + 1
    n_per_ch  = 6 + n_bins
    out       = np.zeros((n, ch * n_per_ch), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg    = X[i, c, :]
            mean   = seg.mean()
            std    = seg.std()
            rms    = np.sqrt((seg ** 2).mean())
            maxabs = np.abs(seg).max()
            zc     = np.sum(np.diff(np.sign(seg)) != 0) / w
            wl     = np.log1p(np.sum(np.abs(np.diff(seg))))

            windowed  = seg * np.hanning(w)
            fft_power = np.abs(np.fft.rfft(windowed)) ** 2

            base = c * n_per_ch
            out[i, base:base+6]          = [mean, std, rms, maxabs, zc, wl]
            out[i, base+6:base+n_per_ch] = fft_power
    return out


def features_band_power(X: np.ndarray, fs: float = 400.0) -> np.ndarray:
    """
    Band power in 4 physiologically motivated frequency bands per channel.
    Bands chosen for downsampled nerve signal (fs = 20000/50 = 400 Hz):
      low:    0 -  20 Hz  (slow movement artefacts)
      mid:   20 -  80 Hz  (typical EMG/nerve activity)
      high:  80 - 150 Hz  (fast nerve firing)
      ultra:150 - 200 Hz  (high-frequency components)

    Combined with the 6 time-domain features:
    Output shape: (n, channels * 10)

    This is the most Arduino-portable FFT variant -- only 4 extra numbers
    per channel instead of w//2+1, so it could feasibly be ported to
    FeatureExtractor.h with a small FFT library.
    """
    n, ch, w = X.shape
    freqs    = np.fft.rfftfreq(w, d=1.0/fs)

    bands = [
        (  0,  20, "low"),
        ( 20,  80, "mid"),
        ( 80, 150, "high"),
        (150, 200, "ultra"),
    ]
    n_bands  = len(bands)
    n_per_ch = 6 + n_bands
    out      = np.zeros((n, ch * n_per_ch), dtype=np.float32)

    band_masks = [(freqs >= lo) & (freqs < hi) for lo, hi, _ in bands]

    for i in range(n):
        for c in range(ch):
            seg    = X[i, c, :]
            mean   = seg.mean()
            std    = seg.std()
            rms    = np.sqrt((seg ** 2).mean())
            maxabs = np.abs(seg).max()
            zc     = np.sum(np.diff(np.sign(seg)) != 0) / w
            wl     = np.log1p(np.sum(np.abs(np.diff(seg))))

            windowed  = seg * np.hanning(w)
            fft_power = np.abs(np.fft.rfft(windowed)) ** 2

            bp = np.array([fft_power[mask].sum() for mask in band_masks])
            # log1p to compress dynamic range -- same trick as waveform length
            bp = np.log1p(bp)

            base = c * n_per_ch
            out[i, base:base+6]          = [mean, std, rms, maxabs, zc, wl]
            out[i, base+6:base+n_per_ch] = bp
    return out


# ── evaluation ───────────────────────────────────────────────────────────────

def logo_eval(name: str, X_feat: np.ndarray,
              y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    logo   = LeaveOneGroupOut()
    scores = []
    all_true, all_pred = [], []

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    for train_idx, test_idx in logo.split(X_feat, y, groups):
        X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
        y_tr, y_te = y[train_idx],      y[test_idx]

        scaler = RobustScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        scores.append((y_pred == y_te).mean())
        all_true.extend(y_te)
        all_pred.extend(y_pred)

    scores = np.array(scores)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Feature dimensions : {X_feat.shape[1]}")
    print(f"  Per-rat scores     : {np.round(scores, 3).tolist()}")
    print(f"  Mean               : {scores.mean():.4f}")
    print(f"  Std                : {scores.std():.4f}")
    print(classification_report(all_true, all_pred,
                                target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    return scores


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    X, y, groups = load_dataset()
    print(f"  {len(y)} windows, shape {X.shape}")

    print("\nExtracting feature sets (this may take a moment)...")
    feat_sets = {
        "Baseline  (6 time-domain)":        features_baseline(X),
        "FFT only  (power spectrum)":        features_fft_only(X),
        "Combined  (time + FFT)":            features_combined(X),
        "Band power (time + 4 freq bands)":  features_band_power(X),
    }

    results = {}
    for name, X_feat in feat_sets.items():
        results[name] = logo_eval(name, X_feat, y, groups)

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY  (LOGO mean accuracy)")
    print("="*55)
    baseline_mean = results["Baseline  (6 time-domain)"].mean()
    for name, scores in results.items():
        delta = scores.mean() - baseline_mean
        marker = "  ← baseline" if delta == 0 else f"  ({delta:+.4f} vs baseline)"
        print(f"  {scores.mean():.4f}  {name}{marker}")

    print()
    print("  Arduino portability notes:")
    print("  - 'Band power' is the most practical to port to FeatureExtractor.h")
    print("    It adds only 4 numbers per channel and needs a small FFT library.")
    print("  - 'FFT only / Combined' adds w//2+1 bins per channel which may")
    print("    exceed Q8.8 range and needs more SRAM for featureMean storage.")
    print("  - If 'Band power' improves accuracy, consider porting it.")
    print("  - If improvement is <3pp, the gain likely does not justify the")
    print("    added complexity and FFT computation time on Arduino.")
