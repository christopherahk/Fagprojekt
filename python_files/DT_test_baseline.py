from pathlib import Path
import os
import numpy as np
from streamer import prepare_dataset, load_rat
import importlib, streamer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

from streamer import load_rat, prepare_dataset
from sklearn.neural_network import MLPClassifier

def load_baseline_dataset(data_dir, rat_ids):
    Xs, ys, groups = [], [], []
    for r in rat_ids:
        path = Path(data_dir) / f"rat{r}.npz"
        if path.exists():
            rms, ang = load_rat(str(path))
            Xr, yr = prepare_dataset(rms, ang)
            Xs.append(Xr)
            ys.append(yr)
            groups.extend([r] * len(yr))
    if not Xs:
        raise RuntimeError("No data found for given rat_ids")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y, np.array(groups)


def extract_features_extended(X):
    n, ch, w = X.shape
    out = np.zeros((n, ch * 6))
    for i in range(n):
        for c in range(ch):
            seg = X[i, c, :]
            mean = seg.mean()
            std = seg.std()
            rms = np.sqrt((seg ** 2).mean())
            maxabs = np.abs(seg).max()
            zc = np.sum(np.diff(np.sign(seg)) != 0) / w
            wl = np.sum(np.abs(np.diff(seg)))
            base = c * 6
            out[i, base:base + 6] = [mean, std, rms, maxabs, zc, wl]
    return out


def evaluate_stratified(name, clf, X, y, n_splits=5):
    X_flat = X.reshape(len(X), -1)
    pipeline = make_pipeline(StandardScaler(), clf)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_flat, y, cv=skf)
    print(f"{name} Stratified {n_splits}-fold CV mean: {scores.mean():.4f} std: {scores.std():.4f}")

    y_pred = cross_val_predict(pipeline, X_flat, y, cv=skf)
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print("Classification report:")
    print(classification_report(y, y_pred, digits=4))


def evaluate_logo(name, clf, X, y, groups):
    X_flat = X.reshape(len(X), -1)
    pipeline = make_pipeline(StandardScaler(), clf)
    logo = LeaveOneGroupOut()
    scores = cross_val_score(pipeline, X_flat, y, cv=logo, groups=groups)
    print(f"{name} Leave-One-Group-Out scores: {scores} mean: {scores.mean():.4f}")

def evaluate_logo_per_rat_norm(name, clf, X_flat, y, groups):
    logo = LeaveOneGroupOut()
    scores = []

    for train_idx, test_idx in logo.split(X_flat, y, groups):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize using only training rats' statistics
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    scores = np.array(scores)
    print(f"{name} LOGO per-split norm: {scores} mean: {scores.mean():.4f}")

def features_rbi_fft(X, n_bins=8):
    """
    RBI (8 bins) + FFT power spectrum combined.
    56 channels * (8 + 17) = 56 * 25 = 1400 features.
    Memory: 1400 * 3 * 2 = 8400 bytes/leaf -- 20 leaves = 164 KB (tight).
    Use n_bins=4 for safer budget: 56 * (4+17) = 1176 features.
    """
    n, ch, w   = X.shape
    n_fft_bins = w // 2 + 1          # 17
    n_per_ch   = n_bins + n_fft_bins  # 8+17 = 25
    out        = np.zeros((n, ch * n_per_ch), dtype=np.float32)

    for i in range(n):
        for c in range(ch):
            seg      = X[i, c, :]
            bin_size = w // n_bins

            # RBI: rectify then integrate per bin, log-compressed
            rectified = np.abs(seg)
            rbi = np.array([rectified[b*bin_size:(b+1)*bin_size].sum()
                            for b in range(n_bins)])
            rbi = np.log1p(rbi)

            # FFT power spectrum, Hanning windowed, log-compressed
            windowed  = seg * np.hanning(w)
            fft_power = np.log1p(np.abs(np.fft.rfft(windowed)) ** 2)

            base = c * n_per_ch
            out[i, base:base+n_bins]          = rbi
            out[i, base+n_bins:base+n_per_ch] = fft_power
    return out


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "dataset_rats_50w"
    rat_ids = list(range(4, 10))

    X, y, groups = load_baseline_dataset(data_dir, rat_ids)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}, groups={groups.shape}")

    # class distribution
    labels = {0: "dorsi", 1: "plantar", 2: "none"}
    total = len(y)
    for label_idx in sorted(labels):
        count = int((y == label_idx).sum())
        share = (count / total * 100.0) if total else 0.0
        print(f"Klasse {label_idx} ({labels[label_idx]}): {count} ({share:.1f}%)")

    # mask out unlabeled samples (labels < 0)
    unlabeled_count = int((y < 0).sum())
    print(f"Unlabeled samples: {unlabeled_count}")
    labeled_mask = (y >= 0)
    X_l = X[labeled_mask]
    y_l = y[labeled_mask]
    groups_l = groups[labeled_mask]

    # basic flattened evaluation on labeled samples
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=42)

    evaluate_stratified("Decision Tree", dt, X_l, y_l)
    evaluate_stratified("Random Forest", rf, X_l, y_l)

    # extended features evaluation (labeled only)
    X_ext = extract_features_extended(X_l)
    rf_ext = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
    print("Extended features RF (5-fold):", cross_val_score(rf_ext, X_ext, y_l, cv=5).mean())

    # leave-one-rat-out on labeled samples
    evaluate_logo("RF extended (LOCO)", rf_ext, X_ext, y_l, groups_l)

    # per-rat breakdown (labeled only)
    for r in rat_ids:
        mask_r = (groups_l == r)
        counts = np.bincount(y_l[mask_r].astype(int), minlength=3)
        print(f"Rat {r}: {mask_r.sum()} labeled samples, klasse fordeling: {counts}")

        # detailed diagnostics for rat 7 (mask r)
        if r == 7 and mask_r.sum() > 0:
            print("\n--- Detaljeret diagnostik for rat 7 ---")
            X_r = X_l[mask_r]  # shape (n_windows, n_channels, seq_len)
            n_r, ch, w = X_r.shape
            print(f"Rat 7: {n_r} windows, channels={ch}, seq_len={w}")

            # per-channel aggregated stats
            chan_mean = X_r.mean(axis=(0, 2))
            chan_std = X_r.std(axis=(0, 2))
            chan_rms = np.sqrt((X_r ** 2).mean(axis=(0, 2)))
            chan_maxabs = np.max(np.abs(X_r), axis=(0, 2))

            # overall summaries
            print(f"Channel RMS median: {np.median(chan_rms):.4f}, mean: {chan_rms.mean():.4f}")

            # show top-5 channels by RMS
            top5 = np.argsort(chan_rms)[-5:][::-1]
            print("Top-5 kanaler efter RMS (kanal: rms, mean, std, maxabs):")
            for c in top5:
                print(f"  ch{c}: {chan_rms[c]:.4f}, mean={chan_mean[c]:.4f}, std={chan_std[c]:.4f}, maxabs={chan_maxabs[c]:.4f}")

            # class-wise RMS within rat 7
            print("\nKlasse-specifik RMS for rat 7:")
            for cls in sorted(labels.keys()):
                cls_mask = (y_l[mask_r] == cls)
                cls_count = int(cls_mask.sum())
                if cls_count == 0:
                    print(f"  klasse {cls}: ingen samples")
                    continue
                X_cls = X_r[cls_mask]
                cls_rms = np.sqrt((X_cls ** 2).mean(axis=(0, 2)))
                print(f"  klasse {cls}: {cls_count} samples, median RMS {np.median(cls_rms):.4f}, mean RMS {cls_rms.mean():.4f}")

            # quick sanity checks
            extreme_channels = np.where(chan_rms > (np.median(chan_rms) * 3))[0]
            if len(extreme_channels) > 0:
                print(f"Advarsel: {len(extreme_channels)} kanaler har RMS > 3x median: {extreme_channels.tolist()}")
            else:
                print("Ingen kanaler med ekstrem RMS (>3x median)")
                print("--- Slut diagnostik for rat 7 ---\n")



    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)

    evaluate_logo_per_rat_norm("MLP (4 hidden)", mlp, X_ext, y_l, groups_l)
    evaluate_logo_per_rat_norm("RF 50 trees",    rf_ext, X_ext, y_l, groups_l)

    original_seq_len = streamer.SEQ_LEN
    streamer.SEQ_LEN = 32

    X32, y32, groups32 = load_baseline_dataset(data_dir, rat_ids)
    X32_ext = extract_features_extended(X32)
    rf_32 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    evaluate_logo_per_rat_norm("RF SEQ_LEN=32", rf_32, X32_ext, y32, groups32)

    streamer.SEQ_LEN = original_seq_len
    X_rbi_fft = features_rbi_fft(X_l, n_bins=4)  # start med 4 bins
    evaluate_logo_per_rat_norm("RBI(4) + FFT", rf_ext, X_rbi_fft, y_l, groups_l)

    X_rbi_fft8 = features_rbi_fft(X_l, n_bins=8)
    evaluate_logo_per_rat_norm("RBI(8) + FFT", rf_ext, X_rbi_fft8, y_l, groups_l)
