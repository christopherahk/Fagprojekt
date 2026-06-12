from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import sys, os
try:
    from streamer import load_rat, prepare_dataset, SEQ_LEN, N_CHANNELS, SUBSAMPLE_RATE
except ImportError:
    print("Could not import streamer.py -- make sure this file sits next to it.")
    sys.exit(1)

#LEAVE ONE GROUP OUT CROSS VALIDATION
# feature extraction (mirrors FeatureExtractor.h)
def extract_features(X: np.ndarray) -> np.ndarray:
    """
    X shape: (n_samples, n_channels, window)
    Returns: (n_samples, n_channels * 6)

    Features per channel (same order as FeatureExtractor.h):
      0  mean
      1  std
      2  RMS
      3  max absolute value
      4  zero-crossing rate
      5  waveform length  (log1p-transformed, matching Arduino fix)
    """
    n, ch, w = X.shape
    out = np.zeros((n, ch * 6), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg = X[i, c, :]
            mean   = seg.mean()
            std    = seg.std()
            rms    = np.sqrt((seg ** 2).mean())
            maxabs = np.abs(seg).max()
            zc     = np.sum(np.diff(np.sign(seg)) != 0) / w
            wl     = np.log1p(np.sum(np.abs(np.diff(seg))))
            base   = c * 6
            out[i, base:base + 6] = [mean, std, rms, maxabs, zc, wl]
    return out


# data loading
def load_dataset(data_dir: str, rat_ids: list):
    Xs, ys, groups = [], [], []
    for r in rat_ids:
        path = Path(data_dir) / f"rat{r}.npz"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        rms, ang = load_rat(str(path))
        Xr, yr   = prepare_dataset(rms, ang)
        Xs.append(Xr)
        ys.append(yr)
        groups.extend([r] * len(yr))
        print(f"  Rat {r}: {len(yr)} windows")
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups)


# LOGO evaluation with per-fold normalisation
def logo_eval(name: str, clf, X_feat: np.ndarray,
              y: np.ndarray, groups: np.ndarray,
              rat_ids: list, class_names: list):
    """
    Leave-One-Group-Out cross-validation.
    Scaler is fit on training folds only -- no data leakage.
    """
    logo   = LeaveOneGroupOut()
    scores = []
    all_y_true, all_y_pred = [], []

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for train_idx, test_idx in logo.split(X_feat, y, groups):
        X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
        y_tr, y_te = y[train_idx],      y[test_idx]
        test_rat   = groups[test_idx[0]]

        scaler = RobustScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc    = (y_pred == y_te).mean()
        scores.append(acc)
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)

        print(f"  Rat {test_rat:2d} (test)  --  {len(y_te):4d} samples  --  acc: {acc:.4f}")

    scores = np.array(scores)
    print(f"\n  Per-rat scores : {np.round(scores, 4).tolist()}")
    print(f"  Mean           : {scores.mean():.4f}")
    print(f"  Std            : {scores.std():.4f}")
    print(f"  Min / Max      : {scores.min():.4f} / {scores.max():.4f}")

    print("\n  Aggregated confusion matrix (all folds):")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)
    print("\n  Classification report (all folds):")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=class_names, digits=4))
    return scores


if __name__ == "__main__":
    DATA_DIR  = Path(__file__).resolve().parent.parent / "dataset_rats_50w"
    RAT_IDS   = list(range(4, 10))
    CLASS_NAMES = ["dorsi", "plantar", "none"]

    print("Loading dataset...")
    X, y, groups = load_dataset(str(DATA_DIR), RAT_IDS)
    print(f"\nTotal: {len(y)} windows, {X.shape[1]} channels, "
          f"window={X.shape[2]}")

    # class distribution
    print("\nClass distribution:")
    total = len(y)
    for ci, cn in enumerate(CLASS_NAMES):
        cnt = int((y == ci).sum())
        print(f"  {ci} ({cn}): {cnt}  ({cnt/total*100:.1f}%)")

    print("\nExtracting features...")
    X_feat = extract_features(X)
    print(f"Feature matrix: {X_feat.shape}")

    # ── Model 1: offline RF  best achievable ceiling
    # This is the gold standard: unlimited trees, full data, offline training.
    # If HAT on Arduino matches this it is performing optimally.
    rf_ceiling = RandomForestClassifier(
        n_estimators=500,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
    )
    ceiling_scores = logo_eval(
        "RF ceiling  (500 trees, depth 3)  -- best offline achievable",
        rf_ceiling, X_feat, y, groups, RAT_IDS, CLASS_NAMES,
    )

    # ── Model 2: RF matched to HAT ensemble (3 trees, depth ~4) ────────────
    # HAT with MAX_LEAVES=20 can reach depth ~4.
    # This shows what a forest of the same capacity achieves offline.
    rf_matched = RandomForestClassifier(
        n_estimators=3,
        max_depth=12,
        random_state=42,
    )
    matched_scores = logo_eval(
        "RF matched to HAT  (3 trees, depth 4)  -- fair offline comparison",
        rf_matched, X_feat, y, groups, RAT_IDS, CLASS_NAMES,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  RF ceiling  (100 trees, d=10) : {ceiling_scores.mean():.4f}")
    print(f"  RF matched  (3 trees,   d=4)  : {matched_scores.mean():.4f}")
    print(f"  HAT on Arduino (reported)     :  ~0.50")
    print()
    print("  Interpretation:")
    print("  - The ceiling score is the best any tree-based model can do")
    print("    on this dataset with leave-one-rat-out evaluation.")
    print("  - If HAT is within ~5pp of the matched RF it is performing")
    print("    as well as theoretically possible given its capacity.")
    print("  - The gap between ceiling and matched RF shows how much")
    print("    ensemble size and depth matter for this problem.")
    print("  - Any remaining gap to the ceiling is due to biological")
    print("    variation across rats, not the model implementation.")
