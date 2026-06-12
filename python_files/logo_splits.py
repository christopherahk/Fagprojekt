"""
LOGO split generator for rat neural signal dataset.
====================================================
Generates leave-one-rat-out train/val splits and saves them
to ./splits_logo/rat_{test_rat}/ for use with streamer.py
and bayesian_opt.py.

Place next to streamer.py and run once:
  python logo_splits.py

Then in bayesian_opt.py, load with:
  from logo_splits import load_logo_split
  X_train, y_train, X_val, y_val = load_logo_split(test_rat=9)
"""

import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler

# ── import from your project ──────────────────────────────────────────────────
try:
    from streamer import load_rat, prepare_dataset, N_CHANNELS
except ImportError:
    raise ImportError("Place this file next to streamer.py")

DATA_DIR = Path("./dataset_rats_50w")
RAT_IDS  = list(range(4, 10))
OUT_DIR  = Path("./splits_logo")


def build_logo_splits(data_dir=DATA_DIR, rat_ids=RAT_IDS, out_dir=OUT_DIR):
    """
    For each rat r in rat_ids:
      - Train: all other rats, scaled using only their data
      - Val:   rat r, transformed with the training scaler

    Saves to out_dir/rat_{r}/train.npz and val.npz
    Also saves the scaler for each fold to scaler_rat_{r}.npz
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all rats
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

        # Scale using ONLY training rats' statistics -- no leakage
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
    """Load a pre-built LOGO fold."""
    fold_dir = Path(splits_dir) / f"rat_{test_rat}"
    if not (fold_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"No split found for rat {test_rat}. Run logo_splits.py first."
        )
    train = np.load(fold_dir / "train.npz")
    val   = np.load(fold_dir / "val.npz")
    return train["X"], train["y"], val["X"], val["y"]


def logo_summary(splits_dir=OUT_DIR):
    """Print accuracy summary across all LOGO folds (for offline RF)."""
    from sklearn.ensemble import ExtraTreesClassifier

    scores = {}
    for test_rat in RAT_IDS:
        try:
            X_tr, y_tr, X_val, y_val = load_logo_split(test_rat, splits_dir)
        except FileNotFoundError:
            continue

        clf = ExtraTreesClassifier(
            n_estimators=24, max_depth=5,
            max_features=0.4, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
        acc = (clf.predict(X_val.reshape(len(X_val), -1)) == y_val).mean()
        scores[test_rat] = acc
        print(f"  Rat {test_rat} (test): {acc:.4f}")

    if scores:
        vals = list(scores.values())
        print(f"\n  LOGO mean: {np.mean(vals):.4f}  std: {np.std(vals):.4f}")
        print(f"  This is the fair ceiling for online Mondrian on Arduino.")


if __name__ == "__main__":
    print("Building LOGO splits...")
    build_logo_splits()

    print("\nOffline ExtraTrees baseline per fold:")
    logo_summary()
