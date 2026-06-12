"""
Tests FFT features with depth-limited RF (max 12 leaves)
to simulate what HAT with MAX_LEAVES=12 can achieve.
Place next to streamer.py and run.
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

def load_dataset():
    Xs, ys, groups = [], [], []
    for r in RAT_IDS:
        path = DATA_DIR / f"rat{r}.npz"
        if not path.exists(): continue
        rms, ang = load_rat(str(path))
        Xr, yr   = prepare_dataset(rms, ang)
        Xs.append(Xr); ys.append(yr)
        groups.extend([r] * len(yr))
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups)

def features_baseline(X):
    n, ch, w = X.shape
    out = np.zeros((n, ch * 6), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg = X[i, c, :]
            out[i, c*6:(c+1)*6] = [
                seg.mean(), seg.std(),
                np.sqrt((seg**2).mean()),
                np.abs(seg).max(),
                np.sum(np.diff(np.sign(seg)) != 0) / w,
                np.log1p(np.sum(np.abs(np.diff(seg))))
            ]
    return out

def features_fft(X):
    n, ch, w = X.shape
    n_bins = w // 2 + 1
    out = np.zeros((n, ch * n_bins), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            seg = X[i, c, :]
            windowed  = seg * np.hanning(w)
            fft_power = np.log1p(np.abs(np.fft.rfft(windowed)) ** 2)
            out[i, c*n_bins:(c+1)*n_bins] = fft_power
    return out

def logo_eval(name, X_feat, y, groups, max_leaf_nodes=None):
    logo = LeaveOneGroupOut()
    scores = []
    all_true, all_pred = [], []

    rf = RandomForestClassifier(
        n_estimators=3,          # match HAT ensemble size
        max_leaf_nodes=max_leaf_nodes,
        random_state=42,
    )

    for train_idx, test_idx in logo.split(X_feat, y, groups):
        X_tr = RobustScaler().fit(X_feat[train_idx]).transform(X_feat[train_idx])
        X_te = RobustScaler().fit(X_feat[train_idx]).transform(X_feat[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        scores.append((y_pred == y_te).mean())
        all_true.extend(y_te); all_pred.extend(y_pred)

    scores = np.array(scores)
    leaves_str = str(max_leaf_nodes) if max_leaf_nodes else "unlimited"
    print(f"\n{'─'*55}")
    print(f"  {name}  (max_leaves={leaves_str})")
    print(f"{'─'*55}")
    print(f"  Features : {X_feat.shape[1]}")
    print(f"  Scores   : {np.round(scores,3).tolist()}")
    print(f"  Mean     : {scores.mean():.4f}  Std: {scores.std():.4f}")
    print(classification_report(all_true, all_pred,
                                target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    return scores.mean()

if __name__ == "__main__":
    print("Loading dataset...")
    X, y, groups = load_dataset()

    print("Extracting features...")
    X_base = features_baseline(X)
    X_fft  = features_fft(X)

    print("\nComparing within HAT capacity constraints (3 trees):")

    results = {
        "Baseline, 20 leaves" : logo_eval("Baseline", X_base, y, groups, 20),
        "Baseline, 12 leaves" : logo_eval("Baseline", X_base, y, groups, 12),
        "FFT,      12 leaves" : logo_eval("FFT only", X_fft,  y, groups, 12),
        "FFT,      20 leaves" : logo_eval("FFT only", X_fft,  y, groups, 20),
    }

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    for name, score in results.items():
        print(f"  {score:.4f}  {name}")
    print()
    print("  If 'FFT 12 leaves' > 'Baseline 20 leaves':")
    print("  FFT is worth porting despite the leaf constraint.")
    print("  If not: stay with current setup.")
