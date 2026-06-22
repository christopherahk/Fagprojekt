"""
generate_tree_cpp.py

Trains a CNN and decision tree on RAT4-9 pooled (temporal split, no shuffle),
then runs two sequential Optuna studies:

  Phase 1 — tune DecisionTreeClassifier hyperparams (minimise val log-loss).
  Phase 2 — fix tree, tune ensemble alpha CNN vs tree (minimise val log-loss).

Exports the fitted tree + best alpha to a self-contained C++ header
(tree_weights.h) that compiles straight into the Arduino sketch.

Pipeline match with the on-device CNN:
  - Rats concatenated in order, then temporal split (no shuffle).
  - StandardScaler fit on train windows only; reused for val/test/tree.
  - Tree features are channel-major flat (index = channel * SEQ_LEN + t),
    matching Tensor(N_CHANNELS, SEQ_LEN).data on the Arduino exactly.

Dependencies:
    pip install optuna scikit-learn torch tqdm
"""

import os

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from sklearn.preprocessing import StandardScaler

from streamer import CLASSES, load_or_build_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR  = "./dataset_rats"
SPLIT_DIR = "./splits_6rats"

RAT_IDS = [4, 5, 6, 7, 8, 9]

RANDOM_SEED     = 10
N_OPTUNA_TRIALS = 60
N_ALPHA_TRIALS  = 40
OUTPUT_HEADER   = "./cpp_part/tree_weights.h"

# CNN training hyperparams
BATCH_SIZE = 64
EPOCHS_CNN = 40
LR_CNN     = 0.0005


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN training helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            loss = criterion(model(xb), yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss


def evaluate(model, X_t, y_t, criterion, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_t.to(device))
        loss   = criterion(logits, y_t.to(device)).item()
        preds  = logits.argmax(dim=1).cpu()
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        acc    = (preds == y_t).float().mean().item()
    return loss, acc, preds.numpy(), probs


def to_tensors(X_seq, y, scaler, n_channels):
    """X_seq: (N, SEQ_LEN, N_CHANNELS) → scale → (N, N_CHANNELS, SEQ_LEN) tensor."""
    X_2d = scaler.transform(X_seq.reshape(-1, n_channels))
    X_t  = torch.tensor(X_2d.reshape(X_seq.shape), dtype=torch.float32).permute(0, 2, 1)
    y_t  = torch.tensor(y, dtype=torch.long)
    return X_t, y_t


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """
    Loads RAT4-9 from pre-built splits, concatenates in rat order, then
    applies a temporal split (70/15/15) across the pooled sequence.

    Returns raw (N, SEQ_LEN, N_CHANNELS) windows for the CNN and flat scaled
    features for the tree, both using the same scaler fit on train only.
    """
    splits = load_or_build_splits(DATA_DIR, RAT_IDS, out_dir=SPLIT_DIR)

    # Concatenate all splits in temporal order per rat, then pool
    X_all = np.concatenate([splits["train"]["X"], splits["val"]["X"], splits["test"]["X"]])
    y_all = np.concatenate([splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]])

    n = len(X_all)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_train_raw, y_train = X_all[:train_end],       y_all[:train_end]
    X_val_raw,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test_raw,  y_test  = X_all[val_end:],          y_all[val_end:]

    n_channels = X_train_raw.shape[1]   # X shape: (N, N_CHANNELS, SEQ_LEN)

    # Scaler fit on train only
    scaler = StandardScaler()
    scaler.fit(X_train_raw.transpose(0, 2, 1).reshape(-1, n_channels))

    print(f"Pooled RATs {RAT_IDS}")
    print(f"Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)} windows")

    def flat(X):
        """(N, N_CHANNELS, SEQ_LEN) → scaled flat (N, N_CHANNELS*SEQ_LEN), channel-major."""
        out = []
        for w in X:
            scaled = scaler.transform(w.T).T
            out.append(scaled.astype(np.float32).flatten(order="C"))
        return np.stack(out)

    Xf_train = flat(X_train_raw)
    Xf_val   = flat(X_val_raw)
    Xf_test  = flat(X_test_raw)

    # Transpose to (N, SEQ_LEN, N_CHANNELS) for to_tensors()
    X_train_seq = X_train_raw.transpose(0, 2, 1)
    X_val_seq   = X_val_raw.transpose(0, 2, 1)
    X_test_seq  = X_test_raw.transpose(0, 2, 1)

    return (
        (Xf_train, y_train, X_train_seq),
        (Xf_val,   y_val,   X_val_seq),
        (Xf_test,  y_test,  X_test_seq),
        scaler,
        n_channels,
    )


# ---------------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------------
def train_cnn(X_train_seq, y_train, X_val_seq, y_val, scaler, n_channels):
    print("\n=== CNN training (RATs %s pooled, temporal split) ===" % RAT_IDS)
    device = torch.device("cpu")

    X_train_t, y_train_t = to_tensors(X_train_seq, y_train, scaler, n_channels)
    X_val_t,   y_val_t   = to_tensors(X_val_seq,   y_val,   scaler, n_channels)

    loader    = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    model     = CNN1D(n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_CNN)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(EPOCHS_CNN):
        train_loss = run_epoch(model, loader, optimizer, criterion, device, train=True)
        val_loss, val_acc, _, _ = evaluate(model, X_val_t, y_val_t, criterion, device)
        print(f"  Epoch {epoch+1:3d}/{EPOCHS_CNN} — "
              f"train_loss: {train_loss/len(y_train):.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Restoring best CNN checkpoint (val_loss: {best_val_loss:.4f})")
    model.load_state_dict(best_state)
    return model, device


# ---------------------------------------------------------------------------
# Phase 1: tree Optuna objective
# ---------------------------------------------------------------------------
def make_tree_objective(Xf_train, y_train, Xf_val, y_val):
    def objective(trial):
        params = dict(
            max_depth         = trial.suggest_int(        "max_depth",          2,   14),
            min_samples_split = trial.suggest_int(        "min_samples_split",  2,   50),
            min_samples_leaf  = trial.suggest_int(        "min_samples_leaf",   1,   50),
            max_features      = trial.suggest_categorical("max_features",       ["sqrt", "log2", None]),
            criterion         = trial.suggest_categorical("criterion",          ["gini", "entropy"]),
            ccp_alpha         = trial.suggest_float(      "ccp_alpha",          0.0,  0.01),
            class_weight      = trial.suggest_categorical("class_weight",       ["balanced", None]),
            splitter          = trial.suggest_categorical("splitter",           ["best", "random"]),
            random_state      = RANDOM_SEED,
        )
        m = DecisionTreeClassifier(**params)
        m.fit(Xf_train, y_train)
        return log_loss(y_val, m.predict_proba(Xf_val), labels=list(CLASSES.values()))
    return objective


# ---------------------------------------------------------------------------
# Phase 2: alpha Optuna objective
# ---------------------------------------------------------------------------
def make_alpha_objective(tree_val_probs, cnn_val_probs, y_val):
    labels = list(CLASSES.values())
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        return log_loss(y_val, alpha * cnn_val_probs + (1.0 - alpha) * tree_val_probs, labels=labels)
    return objective


# ---------------------------------------------------------------------------
# C++ export
# ---------------------------------------------------------------------------
def tree_to_cpp_nodes(tree, n_classes):
    nodes = []
    for i in range(tree.node_count):
        if tree.feature[i] == -2:
            counts = tree.value[i][0]
            total  = counts.sum()
            probs  = (counts / total) if total > 0 else np.zeros(n_classes)
            nodes.append(dict(feature=-2, left=-1, right=-1, threshold=0.0, probs=probs))
        else:
            nodes.append(dict(
                feature   = int(tree.feature[i]),
                left      = int(tree.children_left[i]),
                right     = int(tree.children_right[i]),
                threshold = float(tree.threshold[i]),
                probs     = np.zeros(n_classes),
            ))
    return nodes


def export_tree_to_cpp(model, n_features, n_classes, class_names, alpha, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("#ifndef TREE_WEIGHTS_H\n#define TREE_WEIGHTS_H\n\n")
        f.write("// Auto-generated by generate_tree_cpp.py -- do not edit by hand\n")
        f.write("// Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)) + "\n")
        f.write(f"// Ensemble alpha (CNN weight): {alpha:.6f}\n\n")
        f.write(f"#define TREE_N_FEATURES {n_features}\n")
        f.write(f"#define TREE_N_CLASSES  {n_classes}\n")
        f.write( "#define TREE_N_TREES    1\n")
        f.write(f"#define ENSEMBLE_ALPHA  {alpha:.6f}f\n\n")
        f.write(
            "struct TreeNode {\n"
            "  int16_t feature;   // -2 marks a leaf\n"
            "  int16_t left;\n"
            "  int16_t right;\n"
            "  float threshold;\n"
            "  float probs[TREE_N_CLASSES];\n"
            "};\n\n"
        )
        nodes = tree_to_cpp_nodes(model.tree_, n_classes)
        f.write("const TreeNode tree0_nodes[] = {\n")
        for n in nodes:
            probs_str = ", ".join(f"{p:.6f}f" for p in n["probs"])
            f.write(f"  {{ {n['feature']}, {n['left']}, {n['right']}, "
                    f"{n['threshold']:.6f}f, {{{probs_str}}} }},\n")
        f.write("};\n\n")
        f.write("const TreeNode* const TREES[TREE_N_TREES] = {\n  tree0_nodes\n};\n\n")
        f.write(
            "inline void treePredictSingle(const TreeNode* nodes, const float* x, float* outProbs) {\n"
            "  int node = 0;\n"
            "  while (nodes[node].feature != -2) {\n"
            "    if (x[nodes[node].feature] <= nodes[node].threshold) node = nodes[node].left;\n"
            "    else node = nodes[node].right;\n"
            "  }\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] = nodes[node].probs[c];\n"
            "}\n\n"
            "// x must be TREE_N_FEATURES floats, channel-major\n"
            "// (same layout as Tensor(N_CHANNELS, SEQ_LEN).data on the Arduino).\n"
            "inline void treeForestPredict(const float* x, float* outProbs) {\n"
            "  float leaf[TREE_N_CLASSES];\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] = 0.0f;\n"
            "  for (int t = 0; t < TREE_N_TREES; t++) {\n"
            "    treePredictSingle(TREES[t], x, leaf);\n"
            "    for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] += leaf[c];\n"
            "  }\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] /= TREE_N_TREES;\n"
            "}\n\n"
            "// CNN + Tree ensemble: weighted average of the two probability vectors.\n"
            "// ENSEMBLE_ALPHA is the CNN weight; tuned by Optuna on val set.\n"
            "inline void ensembleBlend(const float* cnnProbs, const float* treeProbs,\n"
            "                          float alpha, float* outProbs) {\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++)\n"
            "    outProbs[c] = alpha * cnnProbs[c] + (1.0f - alpha) * treeProbs[c];\n"
            "}\n\n"
            "inline int argmaxProbs(const float* probs, int n) {\n"
            "  int best = 0;\n"
            "  for (int i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;\n"
            "  return best;\n"
            "}\n\n"
        )
        f.write("#endif\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    (
        (Xf_train, y_train, X_train_seq),
        (Xf_val,   y_val,   X_val_seq),
        (Xf_test,  y_test,  X_test_seq),
        scaler,
        n_channels,
    ) = load_data()

    n_features = Xf_train.shape[1]
    n_classes  = len(CLASSES)
    criterion  = nn.CrossEntropyLoss()

    # ── CNN ─────────────────────────────────────────────────────────────────
    cnn_model, device = train_cnn(X_train_seq, y_train, X_val_seq, y_val, scaler, n_channels)

    X_val_t,  y_val_t  = to_tensors(X_val_seq,  y_val,  scaler, n_channels)
    X_test_t, y_test_t = to_tensors(X_test_seq, y_test, scaler, n_channels)

    _, cnn_val_acc,  _,              cnn_val_probs  = evaluate(cnn_model, X_val_t,  y_val_t,  criterion, device)
    _, cnn_test_acc, cnn_test_preds, cnn_test_probs = evaluate(cnn_model, X_test_t, y_test_t, criterion, device)

    print(f"\nCNN val acc: {cnn_val_acc:.4f} | CNN test acc: {cnn_test_acc:.4f}")
    print(classification_report(y_test, cnn_test_preds, target_names=["Neutral", "Dorsi", "Plantar"]))

    # ── Phase 1: tune tree ──────────────────────────────────────────────────
    print("\n=== Phase 1: Tuning decision tree ===")
    tree_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    tree_study.optimize(
        make_tree_objective(Xf_train, y_train, Xf_val, y_val),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    print("Best val log-loss :", tree_study.best_value)
    print("Best params       :", tree_study.best_params)

    best_params = {**tree_study.best_params, "random_state": RANDOM_SEED}

    # Val-only tree for unbiased alpha tuning
    val_tree = DecisionTreeClassifier(**best_params)
    val_tree.fit(Xf_train, y_train)
    tree_val_probs = val_tree.predict_proba(Xf_val)

    # Final tree refit on train + val
    final_tree = DecisionTreeClassifier(**best_params)
    final_tree.fit(
        np.concatenate([Xf_train, Xf_val]),
        np.concatenate([y_train,  y_val]),
    )
    tree_test_probs = final_tree.predict_proba(Xf_test)

    # ── Phase 2: tune alpha ─────────────────────────────────────────────────
    print("\n=== Phase 2: Tuning ensemble alpha ===")
    alpha_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    alpha_study.optimize(
        make_alpha_objective(tree_val_probs, cnn_val_probs, y_val),
        n_trials=N_ALPHA_TRIALS,
        show_progress_bar=True,
    )
    best_alpha = alpha_study.best_params["alpha"]
    print(f"Best alpha (CNN weight) : {best_alpha:.4f}  (val log-loss {alpha_study.best_value:.4f})")

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\n=== Test set evaluation ===")
    tree_test_pred = final_tree.predict(Xf_test)
    ensemble_probs = best_alpha * cnn_test_probs + (1.0 - best_alpha) * tree_test_probs
    ensemble_pred  = np.argmax(ensemble_probs, axis=1)

    print(f"Tree     accuracy : {accuracy_score(y_test, tree_test_pred):.4f}")
    print(f"CNN      accuracy : {cnn_test_acc:.4f}")
    print(f"Ensemble accuracy : {accuracy_score(y_test, ensemble_pred):.4f}  (alpha={best_alpha:.4f})")
    print("Confusion matrix (ensemble):\n", confusion_matrix(y_test, ensemble_pred))
    print(f"Tree node count   : {final_tree.tree_.node_count}")

    export_tree_to_cpp(
        final_tree,
        n_features  = n_features,
        n_classes   = n_classes,
        class_names = list(CLASSES.keys()),
        alpha       = best_alpha,
        path        = OUTPUT_HEADER,
    )
    print(f"\nWrote {OUTPUT_HEADER}  (ENSEMBLE_ALPHA = {best_alpha:.6f})")


if __name__ == "__main__":
    main()
