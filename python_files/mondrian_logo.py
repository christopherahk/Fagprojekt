import numpy as np
import optuna
import os
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

# copied loro.py setup, adjusted for mondrian trees with claude

N_TRIALS   = 50
N_CHANNELS = 56
SEQ_LEN    = 32
SUBSAMPLE  = 10




def extract_features(X: np.ndarray) -> np.ndarray:
    """
    RBI (4 bins) + FFT power spectrum per channel.
    X shape: (n_samples, n_channels, window)
    Returns: (n_samples, n_channels * 21)
    """
    n, ch, w  = X.shape
    n_rbi     = 4
    n_fft     = w // 2 + 1   # 17 at w=32
    n_per_ch  = n_rbi + n_fft
    out       = np.zeros((n, ch * n_per_ch), dtype=np.float32)

    for i in range(n):
        for c in range(ch):
            seg      = X[i, c, :]
            bin_size = w // n_rbi

            # RBI: rectify, bin-integrate, log1p
            for b in range(n_rbi):
                s = np.abs(seg[b*bin_size:(b+1)*bin_size]).sum()
                out[i, c*n_per_ch + b] = np.log1p(s)

            # FFT power spectrum, Hanning windowed, log1p
            windowed  = seg * np.hanning(w)
            fft_power = np.log1p(np.abs(np.fft.rfft(windowed)) ** 2)
            out[i, c*n_per_ch + n_rbi:c*n_per_ch + n_per_ch] = fft_power

    return out



def prepare_dataset(rms_data, angles_ds):
    rms_data = rms_data[:N_CHANNELS, :]
    X        = rms_data.T

    # Label order matches Arduino: 0=dorsi, 1=plantar, 2=none
    y = np.full(len(angles_ds), 2, dtype=np.int64)
    y[angles_ds >  2.0] = 1
    y[angles_ds < -2.0] = 0

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X), SUBSAMPLE):
        window_labels = y[i - SEQ_LEN:i]
        counts   = np.bincount(window_labels, minlength=3)
        majority = int(np.argmax(counts))
        if counts[majority] < SEQ_LEN * 0.6:
            continue
        X_seq.append(X[i - SEQ_LEN:i].T)
        y_seq.append(majority)

    return np.array(X_seq), np.array(y_seq)


def load_rat_file(path):
    data = np.load(path)
    return data["rms_data"], data["a"]


def split_data(X, y, train=0.70, val=0.15, seed=42):
    rng       = np.random.default_rng(seed)
    idx       = rng.permutation(len(X))
    train_end = int(train * len(idx))
    val_end   = int((train + val) * len(idx))
    return (
        X[idx[:train_end]],   y[idx[:train_end]],
        X[idx[train_end:val_end]], y[idx[train_end:val_end]],
        X[idx[val_end:]],     y[idx[val_end:]],
    )



def fits_in_sram(n_trees: int, max_nodes: int,
                 budget_kb: float = 200.0) -> bool:
    node_bytes  = 32
    range_bytes = 1176 * 2 * 2
    tree_bytes  = max_nodes * node_bytes + range_bytes + 2
    total       = (n_trees * tree_bytes
                   + 56 * 32 * 4    # values[]
                   + 1176 * 4       # features[]
                   + 2048)
    return (total / 1024) <= budget_kb



def train_model(X_train, y_train, X_val, y_val, params, verbose=False):
    """
    Offline proxy for Mondrian Forest using ExtraTrees/RF.
    ExtraTrees uses random thresholds like Mondrian -- closest offline analogue.
    Feature extraction and scaling applied here exactly as on Arduino.
    """
    # Extract RBI+FFT features
    X_tr_feat  = extract_features(X_train)
    X_val_feat = extract_features(X_val)

    # Scale with RobustScaler (same as streamer.py)
    scaler     = RobustScaler()
    X_tr_feat  = scaler.fit_transform(X_tr_feat)
    X_val_feat = scaler.transform(X_val_feat)

    depth = int(np.log2(params["max_nodes"] + 1)) - 1
    labels=[0, 1, 2]

    if params["model_type"] == "extra":
        clf = ExtraTreesClassifier(
            n_estimators=params["n_trees"],
            max_depth=depth,
            max_features=params["max_features"],
            random_state=42, n_jobs=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=params["n_trees"],
            max_depth=depth,
            max_features=params["max_features"],
            random_state=42, n_jobs=-1,
        )

    clf.fit(X_tr_feat, y_train)
    y_pred = clf.predict(X_val_feat)
    acc    = (y_pred == y_val).mean()

    if verbose:
        print(f"  Val accuracy: {acc:.4f}")

    return clf, scaler, y_val, y_pred


def run_optuna(X_train, y_train, X_val, y_val, seed=42):
    def objective(trial):
        params = {
            "n_trees":    trial.suggest_int("n_trees", 3, 25),
            "max_nodes":  trial.suggest_categorical("max_nodes", [31, 63, 127]),
            "max_features": trial.suggest_categorical(
                                "max_features", ["sqrt", "log2", 0.2, 0.4]),
            "model_type": trial.suggest_categorical(
                                "model_type", ["extra", "rf"]),
        }

        if not fits_in_sram(params["n_trees"], params["max_nodes"]):
            raise optuna.exceptions.TrialPruned()

        _, _, y_true, y_pred = train_model(
            X_train, y_train, X_val, y_val, params)
        return float((y_pred == y_true).mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n  Best val acc: {study.best_value:.4f}")
    print(f"  Best params:  {study.best_params}")
    return study.best_params



def leave_one_rat_out(data_dir, rat_ids, seed=42):
    results = {}

    for test_rat in rat_ids:
        print(f"\n{'='*50}")
        print(f"  Testing on RAT {test_rat}")
        print(f"{'='*50}")

        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        for rat in rat_ids:
            path     = os.path.join(data_dir, f"rat{rat}.npz")
            rms, ang = load_rat_file(path)
            X, y     = prepare_dataset(rms, ang)

            if rat == test_rat:
                X_test, y_test = X, y
            else:
                X_train_list.append(X)
                y_train_list.append(y)

        X_all = np.concatenate(X_train_list)
        y_all = np.concatenate(y_train_list)

        X_train, y_train, X_val, y_val, X_held, y_held = split_data(
            X_all, y_all, train=0.70, val=0.15, seed=seed)

        print(f"  Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Held: {X_held.shape} | Test rat: {X_test.shape}")
        print("  Running Optuna hyperparameter search...")

        best_params = run_optuna(X_train, y_train, X_val, y_val, seed=seed)

        # Retrain on train+val with best params, evaluate on test rat
        X_all_train = np.concatenate([X_train, X_val])
        y_all_train = np.concatenate([y_train, y_val])
        X_tr_f, y_tr_f, X_va_f, y_va_f, _, _ = split_data(
            X_all_train, y_all_train, train=0.85, val=0.15, seed=seed)

        _, scaler, y_true, y_pred = train_model(
            X_tr_f, y_tr_f, X_va_f, y_va_f,
            best_params, verbose=True)

        # Final evaluation on held-out test rat
        X_test_feat = extract_features(X_test)
        X_test_feat = scaler.transform(X_test_feat)
        depth       = int(np.log2(best_params["max_nodes"] + 1)) - 1

        if best_params["model_type"] == "extra":
            final_clf = ExtraTreesClassifier(
                n_estimators=best_params["n_trees"],
                max_depth=depth,
                max_features=best_params["max_features"],
                random_state=42, n_jobs=-1)
        else:
            final_clf = RandomForestClassifier(
                n_estimators=best_params["n_trees"],
                max_depth=depth,
                max_features=best_params["max_features"],
                random_state=42, n_jobs=-1)

        X_tr_feat   = extract_features(X_all_train)
        X_tr_feat   = scaler.transform(X_tr_feat)
        final_clf.fit(X_tr_feat, y_all_train)
        y_test_pred = final_clf.predict(X_test_feat)

        test_acc = (y_test_pred == y_test).mean()
        print(f"\n  Test rat {test_rat} accuracy: {test_acc:.4f}")
        print(classification_report(
            y_test, y_test_pred,
            labels=[0, 1, 2],
            target_names=["dorsi", "plantar", "none"],
            zero_division=0))

        results[test_rat] = {
    "test_acc":   test_acc,
    "report":     classification_report(
                      y_test, y_test_pred,
                      labels=[0, 1, 2],
                      target_names=["dorsi", "plantar", "none"],
                      output_dict=True, zero_division=0),
    "best_params": best_params,
}

    return results



if __name__ == "__main__":
    data_dir = "./dataset_rats_50w"
    rat_ids  = list(range(4, 10))

    results = leave_one_rat_out(data_dir, rat_ids)

    print("\n\nSummary:")
    accs = []
    for rat, res in results.items():
        acc = res["test_acc"]
        accs.append(acc)
        print(f"  RAT {rat}: test acc={acc:.4f}  "
              f"best params={res['best_params']}")

    print(f"\n  LOGO mean accuracy : {np.mean(accs):.4f}")
    print(f"  LOGO std           : {np.std(accs):.4f}")
    print(f"\n  This is the offline ceiling for Mondrian Forest on Arduino.")
    print(f"  Compare to Arduino online accuracy to measure the online gap.")
