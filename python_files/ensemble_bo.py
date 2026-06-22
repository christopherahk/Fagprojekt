"""
ensemble_bo.py
==============
Bayesian-optimised weighted ensemble of the 1-D CNN and the pre-trained
random forest stored in tree_weights.h.

The blend is:
    final_probs = alpha * cnn_probs + (1 - alpha) * forest_probs

where alpha is the CNN weight (0 → 1). Optuna searches alpha to find the
optimal balance between the models.

Uses pre-made population-level train / val / test splits rather than
leave-one-rat-out cross-validation.

Usage
-----
    python ensemble_bo.py
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import optuna
from dataclasses import dataclass
from typing import List

# ── paths ──────────────────────────────────────────────────────────────────────
TREE_WEIGHTS_H = "./cpp_part/tree_weights.h"

# Folder that contains train.npz / val.npz / test.npz
SPLITS_DIR = r"C:\Users\Matti\Desktop\Code Projects\Fagprojekt2\Fagprojekt\splits_population"

# ── training constants ─────────────────────────────────────────────────────────
EPOCHS   = 50
N_TRIALS = 100  # Increased since searching ONLY alpha is nearly instantaneous

# ── fixed cnn architecture & hyper-parameters ──────────────────────────────────
DEFAULT_CNN_PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "filters1": 32,
    "filters2": 64,
    "fc_units": 32,
    "kernel_size": 5,
}

# ── forest constants (mirror the #defines in tree_weights.h) ───────────────────
TREE_N_FEATURES = 896
TREE_N_CLASSES  = 3
TREE_N_TREES    = 22


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Parse tree_weights.h → Python inference
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    feature:   int
    left:      int
    right:     int
    threshold: float
    probs:     List[float]


def _parse_node_line(line: str) -> TreeNode:
    parts = line.split('{')
    outer = parts[1]
    inner = parts[2].split('}')[0]
    probs = [float(v.strip().replace("f", "")) for v in inner.split(",") if v.strip()]
    outer_parts = [p.strip().replace("f", "") for p in outer.split(",") if p.strip()]
    feature   = int(outer_parts[0])
    left      = int(outer_parts[1])
    right     = int(outer_parts[2])
    threshold = float(outer_parts[3])
    return TreeNode(feature, left, right, threshold, probs)


def load_forest(h_path: str) -> List[List[TreeNode]]:
    """Return a list of TREE_N_TREES trees, each a list of TreeNode objects."""
    with open(h_path, "r") as fh:
        src = fh.read()
    forests: List[List[TreeNode]] = []
    pattern = re.compile(
        r"const TreeNode tree\d+_nodes\[\]\s*=\s*\{(.*?)\};",
        re.DOTALL,
    )
    for match in pattern.finditer(src):
        block = match.group(1)
        nodes = []
        for raw_line in re.findall(r"\{[^}]*\{[^}]*\}[^}]*\}", block):
            nodes.append(_parse_node_line(raw_line))
        forests.append(nodes)
    assert len(forests) == TREE_N_TREES, (
        f"Expected {TREE_N_TREES} trees, parsed {len(forests)}"
    )
    return forests


def _predict_single(nodes: List[TreeNode], x: np.ndarray) -> np.ndarray:
    idx = 0
    while nodes[idx].feature != -2:
        n = nodes[idx]
        idx = n.left if x[n.feature] <= n.threshold else n.right
    return np.array(nodes[idx].probs, dtype=np.float32)


def forest_predict_proba(forest: List[List[TreeNode]], X: np.ndarray) -> np.ndarray:
    """
    X : (N, n_channels, seq_len) — channel-first layout (same as CNN tensors)
    Returns: probs : (N, TREE_N_CLASSES) averaged across all trees
    """
    N = X.shape[0]
    X_flat = X.reshape(N, -1)   # (N, n_channels * seq_len)
    out = np.zeros((N, TREE_N_CLASSES), dtype=np.float32)
    for tree in forest:
        for i in range(N):
            out[i] += _predict_single(tree, X_flat[i])
    out /= TREE_N_TREES
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CNN
# ══════════════════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    def __init__(self, n_channels, filters1, filters2, fc_units, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, filters1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(filters1, filters2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filters2, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, TREE_N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(splits_dir: str):
    def _load(name):
        path = os.path.join(splits_dir, f"{name}.npz")
        d = np.load(path)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)

    X_train, y_train = _load("train")
    X_val,   y_val   = _load("val")
    X_test,  y_test  = _load("test")

    print(f"Loaded splits from: {splits_dir}")
    print(f"  Train : {X_train.shape}  labels {np.bincount(y_train)}")
    print(f"  Val   : {X_val.shape}    labels {np.bincount(y_val)}")
    print(f"  Test  : {X_test.shape}   labels {np.bincount(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Train CNN
# ══════════════════════════════════════════════════════════════════════════════

def _scale(X_train_cf, X_val_cf):
    N_tr, C, T = X_train_cf.shape
    N_vl       = X_val_cf.shape[0]

    scaler = StandardScaler()
    X_tr_2d = scaler.fit_transform(
        X_train_cf.transpose(0, 2, 1).reshape(-1, C)
    )
    X_vl_2d = scaler.transform(
        X_val_cf.transpose(0, 2, 1).reshape(-1, C)
    )

    X_tr_scaled = X_tr_2d.reshape(N_tr, T, C).transpose(0, 2, 1)
    X_vl_scaled = X_vl_2d.reshape(N_vl, T, C).transpose(0, 2, 1)

    return X_tr_scaled, X_vl_scaled, scaler


def train_model(X_train, y_train, X_val, y_val, params, verbose=False):
    X_tr_scaled, X_vl_scaled, scaler = _scale(X_train, X_val)
    n_channels = X_train.shape[1]

    X_train_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
    X_val_t   = torch.tensor(X_vl_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    device = torch.device("cpu")
    model = CNN1D(
        n_channels  = n_channels,
        filters1    = params["filters1"],
        filters2    = params["filters2"],
        fc_units    = params["fc_units"],
        kernel_size = params["kernel_size"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        if verbose:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t.to(device))
                val_loss   = criterion(val_logits, y_val_t.to(device)).item()
                val_preds  = val_logits.argmax(dim=1).cpu()
                val_acc    = (val_preds == y_val_t).float().mean().item()
            print(
                f"Epoch {epoch+1}/{EPOCHS} – "
                f"train_loss: {train_loss/len(y_train):.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
            )

    model.eval()
    with torch.no_grad():
        logits        = model(X_val_t.to(device))
        cnn_val_probs = torch.softmax(logits, dim=1).cpu().numpy()

    return model, scaler, y_val, cnn_val_probs, X_vl_scaled


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Ensemble blend
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(cnn_probs, forest_probs, alpha):
    blended = alpha * cnn_probs + (1.0 - alpha) * forest_probs
    return blended.argmax(axis=1), blended


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Optuna objective (Optimizing ONLY Alpha)
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_alpha_only(X_train, y_train, X_val, y_val, forest, seed=42):
    # 1. Train the CNN exactly once using the baseline architecture configurations
    print("Training the baseline CNN model once to get validation probabilities...")
    _, _, y_true, cnn_probs, X_val_np = train_model(
        X_train, y_train, X_val, y_val, DEFAULT_CNN_PARAMS, verbose=True
    )

    # 2. Get static forest validation probabilities once
    print("Generating Forest baseline validation probabilities...")
    forest_probs = forest_predict_proba(forest, X_val_np)

    # 3. Use Optuna to search ONLY for alpha using the static probabilities
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.5, 1.0)
        y_pred, _ = ensemble_predict(cnn_probs, forest_probs, alpha)
        return (y_pred == y_true).mean()

    print(f"Running Optuna study over {N_TRIALS} trials to optimize alpha...")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest trial val_acc : {study.best_value:.4f}")
    print(f"Best alpha         : {study.best_params['alpha']:.4f}")
    return study.best_params['alpha']


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── load forest ──────────────────────────────────────────────────────────
    print("Loading random-forest weights from", TREE_WEIGHTS_H)
    forest = load_forest(TREE_WEIGHTS_H)
    print(f"  → {len(forest)} trees, {sum(len(t) for t in forest)} nodes total\n")

    # ── load population splits ───────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(SPLITS_DIR)

    # ── Bayesian optimisation on train / val ─────────────────────────────────
    print("\nRunning Optuna alpha optimization search …")
    alpha = run_optuna_alpha_only(X_train, y_train, X_val, y_val, forest)
    print(f"  → Optimised CNN weight (alpha) : {alpha:.3f}\n")

    # ── retrain on train+val, evaluate on test ───────────────────────────────
    print("Retraining CNN on train + val with baseline parameters …")
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    # keep a small monitor split so verbose training can log val_loss
    n_all     = len(X_all)
    rng       = np.random.default_rng(42)
    idx       = rng.permutation(n_all)
    split     = int(0.90 * n_all)
    tr_idx, monitor_idx = idx[:split], idx[split:]

    model, scaler, _, _, _ = train_model(
        X_all[tr_idx],       y_all[tr_idx],
        X_all[monitor_idx],  y_all[monitor_idx],
        DEFAULT_CNN_PARAMS,
        verbose=True,
    )

    # ── evaluate on held-out test set ────────────────────────────────────────
    print("\nEvaluating on test set …")

    # scale test with the fitted scaler
    N_te, C, T = X_test.shape
    X_te_2d    = scaler.transform(
        X_test.transpose(0, 2, 1).reshape(-1, C)
    )
    X_te_scaled = X_te_2d.reshape(N_te, T, C).transpose(0, 2, 1).astype(np.float32)

    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        logits         = model(torch.tensor(X_te_scaled).to(device))
        cnn_test_probs = torch.softmax(logits, dim=1).cpu().numpy()

    forest_test_probs = forest_predict_proba(forest, X_te_scaled)
    y_pred_test, _    = ensemble_predict(cnn_test_probs, forest_test_probs, alpha)

    print(f"\n── Test set report (alpha={alpha:.3f}) ──")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["Neutral", "Dorsi", "Plantar"],
    ))
