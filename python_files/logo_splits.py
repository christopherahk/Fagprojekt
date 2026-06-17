"""
LORO split generator for rat neural signal dataset.

Generates leave-one-rat-out train/val splits and saves them
to ./splits_logo/rat_{test_rat}/ for use with streamer.py
and bayesian_opt.py.
"""

import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler

try:
    from streamer import load_rat, prepare_dataset, N_CHANNELS, SEQ_LEN
except ImportError:
    raise ImportError("Place this file next to streamer.py")

DATA_DIR = Path("./dataset_rats_50w")
RAT_IDS  = list(range(4, 11))
OUT_DIR  = Path("./splits_logo")


def extract_features_offline(X_seq, window_size=16, n_rbi_bins=8):
    """
    Python equivalent of the C++ extractFeatures() function.
    Transforms raw (samples, channels, window_size) into (samples, 562 features).
    """
    n_samples = X_seq.shape[0]
    # X_seq er allerede (samples, channels, time), f.eks. (N, 56, 16)

    bin_size = window_size // n_rbi_bins

    # RBI Bins (samples, channels, bins)
    reshaped_for_bins = X_seq.reshape(n_samples, N_CHANNELS, n_rbi_bins, bin_size)
    rbi_sums = np.sum(np.abs(reshaped_for_bins), axis=3)
    rbi_features = np.log1p(rbi_sums)

    # local stats (samples, channels, 1)
    ch_means = np.mean(np.abs(X_seq), axis=2, keepdims=True)
    ch_maxs = np.max(np.abs(X_seq), axis=2, keepdims=True)
    local_mean_features = np.log1p(ch_means)
    local_max_features = np.log1p(ch_maxs)

    # concatenate local features per channel: (samples, channels, 10)
    local_features = np.concatenate([rbi_features, local_mean_features, local_max_features], axis=2)
    # Flatten local features: (samples, 560)
    flat_local = local_features.reshape(n_samples, -1)

    # global spatial stats (samples, 1)
    global_mean = np.mean(ch_means.squeeze(axis=2), axis=1, keepdims=True)
    global_max = np.max(ch_means.squeeze(axis=2), axis=1, keepdims=True)
    spatial_variance = global_max - global_mean

    global_mean_features = np.log1p(global_mean)
    spatial_var_features = np.log1p(spatial_variance)


    final_features = np.concatenate([flat_local, global_mean_features, spatial_var_features], axis=1)

    return final_features


def build_logo_splits(data_dir=DATA_DIR, rat_ids=RAT_IDS, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_X, all_y, all_groups = [], [], []
    for r in rat_ids:
        path = Path(data_dir) / f"rat{r}.npz"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        rms, ang = load_rat(str(path))
        X, y     = prepare_dataset(rms, ang)
        all_X.append(X)
        all_y.append(y)
        all_groups.extend([r] * len(y))
        print(f"  Rat {r}: {len(y)} windows")

    all_X      = np.concatenate(all_X)
    all_y      = np.concatenate(all_y)
    all_groups = np.array(all_groups)

    print(f"\nTotal: {len(all_y)} windows across {len(rat_ids)} rats")
    print(f"Class distribution: {np.bincount(all_y.astype(int))}")

    for test_rat in rat_ids:
        fold_dir = out_dir / f"rat_{test_rat}"
        fold_dir.mkdir(exist_ok=True)

        train_mask = all_groups != test_rat
        test_mask  = all_groups == test_rat

        X_train_raw = all_X[train_mask]
        y_train     = all_y[train_mask]
        X_val_raw   = all_X[test_mask]
        y_val       = all_y[test_mask]

        # Scale using ONLY training rats
        n_ch    = X_train_raw.shape[1]
        scaler  = RobustScaler()

        X_tr_2d = X_train_raw.transpose(0, 2, 1).reshape(-1, n_ch)
        X_tr_2d = scaler.fit_transform(X_tr_2d)
        X_train = X_tr_2d.reshape(
            X_train_raw.shape[0], X_train_raw.shape[2], n_ch
        ).transpose(0, 2, 1)

        X_val_2d = X_val_raw.transpose(0, 2, 1).reshape(-1, n_ch)
        X_val_2d = scaler.transform(X_val_2d)
        X_val    = X_val_2d.reshape(
            X_val_raw.shape[0], X_val_raw.shape[2], n_ch
        ).transpose(0, 2, 1)

        np.savez(fold_dir / "train.npz", X=X_train, y=y_train)
        np.savez(fold_dir / "val.npz",   X=X_val,   y=y_val)
        np.savez(fold_dir / "scaler.npz",
                 mean=np.asarray(scaler.center_),
                 scale=np.asarray(scaler.scale_))

        print(f"  Fold rat_{test_rat}: "
              f"train={len(y_train)}, val={len(y_val)}, "
              f"class dist val={np.bincount(y_val.astype(int))}")

    print(f"\nSplits saved to {out_dir}/")
    print("Use load_logo_split(test_rat) to load a specific fold.")


def load_logo_split(test_rat: int, splits_dir=OUT_DIR):
    fold_dir = Path(splits_dir) / f"rat_{test_rat}"
    if not (fold_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"No split found for rat {test_rat}. Run logo_splits.py first."
        )
    train = np.load(fold_dir / "train.npz")
    val   = np.load(fold_dir / "val.npz")
    return train["X"], train["y"], val["X"], val["y"]


def logo_summary(splits_dir=OUT_DIR):
    from sklearn.ensemble import ExtraTreesClassifier

    scores = {}
    for test_rat in RAT_IDS:
        try:
            X_tr, y_tr, X_val, y_val = load_logo_split(test_rat, splits_dir)
        except FileNotFoundError:
            continue


        X_tr_features = extract_features_offline(X_tr, window_size=SEQ_LEN)
        X_val_features = extract_features_offline(X_val, window_size=SEQ_LEN)

        clf = ExtraTreesClassifier(
            n_estimators=15, max_depth=12,
            max_features=0.4, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_features, y_tr)
        acc = (clf.predict(X_val_features) == y_val).mean()
        scores[test_rat] = acc
        print(f"  Rat {test_rat} (test): {acc:.4f}")

    if scores:
        vals = list(scores.values())
        print(f"\n  LOGO mean: {np.mean(vals):.4f}  std: {np.std(vals):.4f}")
        print(f"  This is the fair ceiling for online Mondrian on Arduino.")


if __name__ == "__main__":
    print("Building LOGO splits...")
    build_logo_splits()

    print("\nOffline ExtraTrees baseline per fold (with C++ equivalent features):")
    logo_summary()
