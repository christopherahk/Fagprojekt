"""
generate_tree_cpp.py

Trains a tree ensemble (sklearn RandomForestClassifier, where n_estimators is
itself tuned -- so Optuna can collapse it down to a single tree if that's what
the data wants) on the exact same windowed/scaled EMG features the on-device
CNN sees, tunes it with Bayesian optimization (Optuna), and exports the fitted
trees to a self-contained C++ header (tree_weights.h) that can be dropped next
to trained_weights.h and compiled straight into the Arduino sketch.

Pipeline match with the CNN (see cpp_part.ino / streamer.py):
  - Same train/val/test split loader (load_or_build_splits) and same StandardScaler
    fit procedure as streamer.py, so the tree sees identical inputs to the CNN.
  - Each window is scaled then flattened in channel-major order
    (index = channel * SEQ_LEN + t), which is exactly the byte layout
    streamer.py streams over serial and the layout of Tensor(N_CHANNELS, SEQ_LEN)
    ("input") on the Arduino. That means on-device you can call
    treeForestPredict(input.data, treeProbs) directly -- no re-ordering needed.

Training data composition (the part that matters for the drift story):
  - "Population" rats (POPULATION_RAT_IDS) contribute ALL of their data
    (train+val+test) to the tree's training pool. None of them is the
    deployment target, so there's nothing to reserve a test split for.
  - The "target" rat (TARGET_RAT_ID) is the deployment patient. Its val and
    test folds are NEVER used for training under any setting -- they're the
    only thing the tree is ever scored on, using the same split directory
    the CNN finetuning run already uses, so numbers are directly comparable.
  - Whether the target rat's TRAIN fold also joins the tree's training pool
    is controlled by INCLUDE_TARGET_TRAIN_IN_TREE. True = tree gets a
    population prior plus the same calibration-session data the CNN gets.
    False = tree is a strict zero-shot generalist that never sees the target
    rat at all during training. Flip the flag and re-run to compare both;
    nothing else in the pipeline needs to change.

Dependencies (on top of what streamer.py already needs):
    pip install optuna scikit-learn

Usage:
    python generate_tree_cpp.py
"""

import os

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from streamer import CLASSES, load_or_build_splits

# Quiet down Optuna's per-trial [I ...] logging -- we use a single tqdm
# progress bar instead (set via show_progress_bar=True in study.optimize).
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "./dataset_rats"

# Rats the tree is allowed to see in full -- the "general population" prior.
POPULATION_RAT_IDS = [4, 5, 6, 7, 8, 9]
POPULATION_SPLIT_DIR = "./splits_population"

# The held-out deployment subject. Reuses the same split dir as the CNN
# finetuning run (streamer.py) so train/val/test windows match exactly --
# don't point this at a different dir or you'll silently re-shuffle and
# desync from what the CNN was evaluated on.
TARGET_RAT_ID = 10
TARGET_SPLIT_DIR = "./splits_finetune"

# See module docstring. Default True per current design decision; flip to
# False to test the strict zero-shot-generalist variant instead.
INCLUDE_TARGET_TRAIN_IN_TREE = True

RANDOM_SEED = 10
N_OPTUNA_TRIALS = 60
MAX_TREES = 25  # upper bound on n_estimators -- keeps the exported header small
OUTPUT_HEADER = "../cpp_part/tree_weights.h"


# ---------------------------------------------------------------------------
# Data loading -- identical scaling/flattening to what the Arduino receives
# ---------------------------------------------------------------------------
def flatten_window(window, scaler):
    """window: (N_CHANNELS, SEQ_LEN) raw.

    Scales per-channel exactly like streamer.py, then flattens channel-major
    (the same order as Tensor(N_CHANNELS, SEQ_LEN) storage on the Arduino).
    """
    scaled = scaler.transform(window.T).T
    return scaled.astype(np.float32).flatten(order="C")


def build_flat_features(X, scaler, desc="Flattening windows"):
    return np.stack([flatten_window(w, scaler) for w in tqdm(X, desc=desc)])


def load_data():
    pop = load_or_build_splits(DATA_DIR, POPULATION_RAT_IDS, out_dir=POPULATION_SPLIT_DIR)
    target = load_or_build_splits(DATA_DIR, [TARGET_RAT_ID], out_dir=TARGET_SPLIT_DIR)

    # Training pool: all population data (no test split needed -- none of
    # these rats is ever the evaluation subject) plus, optionally, the
    # target rat's own train fold.
    X_train_parts = [pop["train"]["X"], pop["val"]["X"], pop["test"]["X"]]
    y_train_parts = [pop["train"]["y"], pop["val"]["y"], pop["test"]["y"]]

    if INCLUDE_TARGET_TRAIN_IN_TREE:
        X_train_parts.append(target["train"]["X"])
        y_train_parts.append(target["train"]["y"])

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    # Val/test always come exclusively from the target rat's reserved folds.
    # This is the only place "how does the tree do on the actual deployment
    # subject" gets measured, regardless of the flag above.
    X_val, y_val = target["val"]["X"], target["val"]["y"]
    X_test, y_test = target["test"]["X"], target["test"]["y"]

    n_channels = X_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(X_train.transpose(0, 2, 1).reshape(-1, n_channels))

    Xf_train = build_flat_features(X_train, scaler, desc="Flattening train windows")
    Xf_val = build_flat_features(X_val, scaler, desc="Flattening val windows")
    Xf_test = build_flat_features(X_test, scaler, desc="Flattening test windows")

    mode = "population + target-train" if INCLUDE_TARGET_TRAIN_IN_TREE else "population only"
    print(f"Tree training pool: {len(X_train)} windows ({mode})")
    print(f"Val/test: {len(X_val)} / {len(X_test)} windows, both from rat {TARGET_RAT_ID} exclusively")

    return (Xf_train, y_train), (Xf_val, y_val), (Xf_test, y_test)


# ---------------------------------------------------------------------------
# Optuna objective -- minimizes validation log-loss
# ---------------------------------------------------------------------------
def make_objective(Xf_train, y_train, Xf_val, y_val):
    def objective(trial):
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 1, MAX_TREES),
            max_depth=trial.suggest_int("max_depth", 2, 14),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            ccp_alpha=trial.suggest_float("ccp_alpha", 0.0, 0.01),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
            bootstrap=bootstrap,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        if bootstrap:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

        model = RandomForestClassifier(**params)
        model.fit(Xf_train, y_train)
        val_probs = model.predict_proba(Xf_val)
        return log_loss(y_val, val_probs, labels=list(CLASSES.values()))

    return objective


# ---------------------------------------------------------------------------
# C++ export -- flat node-array representation, works for 1 tree or many
# ---------------------------------------------------------------------------
def tree_to_cpp_nodes(tree, n_classes):
    """Walk an sklearn tree_ into a flat list of node dicts.

    sklearn marks leaves with feature == -2 (TREE_UNDEFINED); we keep that
    convention so the C++ traversal can use the same sentinel.
    """
    nodes = []
    for i in range(tree.node_count):
        if tree.feature[i] == -2:
            counts = tree.value[i][0]
            total = counts.sum()
            probs = (counts / total) if total > 0 else np.zeros(n_classes)
            nodes.append(dict(feature=-2, left=-1, right=-1, threshold=0.0, probs=probs))
        else:
            nodes.append(
                dict(
                    feature=int(tree.feature[i]),
                    left=int(tree.children_left[i]),
                    right=int(tree.children_right[i]),
                    threshold=float(tree.threshold[i]),
                    probs=np.zeros(n_classes),
                )
            )
    return nodes


def export_forest_to_cpp(model, n_features, n_classes, class_names, path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    estimators = model.estimators_

    with open(path, "w") as f:
        f.write("#ifndef TREE_WEIGHTS_H\n#define TREE_WEIGHTS_H\n\n")
        f.write("// Auto-generated by generate_tree_cpp.py -- do not edit by hand\n")
        f.write("// Classes: " + ", ".join(f"{i}={name}" for i, name in enumerate(class_names)) + "\n\n")
        f.write(f"#define TREE_N_FEATURES {n_features}\n")
        f.write(f"#define TREE_N_CLASSES {n_classes}\n")
        f.write(f"#define TREE_N_TREES {len(estimators)}\n\n")

        f.write(
            "struct TreeNode {\n"
            "  int16_t feature;   // -2 marks a leaf\n"
            "  int16_t left;\n"
            "  int16_t right;\n"
            "  float threshold;\n"
            "  float probs[TREE_N_CLASSES];\n"
            "};\n\n"
        )

        for t_idx, est in enumerate(estimators):
            nodes = tree_to_cpp_nodes(est.tree_, n_classes)
            f.write(f"const TreeNode tree{t_idx}_nodes[] = {{\n")
            for n in nodes:
                probs_str = ", ".join(f"{p:.6f}f" for p in n["probs"])
                f.write(
                    f"  {{ {n['feature']}, {n['left']}, {n['right']}, "
                    f"{n['threshold']:.6f}f, {{{probs_str}}} }},\n"
                )
            f.write("};\n\n")

        f.write("const TreeNode* const TREES[TREE_N_TREES] = {\n")
        f.write(",\n".join(f"  tree{i}_nodes" for i in range(len(estimators))))
        f.write("\n};\n\n")

        f.write(
            "inline void treePredictSingle(const TreeNode* nodes, const float* x, float* outProbs) {\n"
            "  int node = 0;\n"
            "  while (nodes[node].feature != -2) {\n"
            "    if (x[nodes[node].feature] <= nodes[node].threshold) node = nodes[node].left;\n"
            "    else node = nodes[node].right;\n"
            "  }\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] = nodes[node].probs[c];\n"
            "}\n\n"
        )

        f.write(
            "// x must be TREE_N_FEATURES floats, channel-major (same layout as\n"
            "// Tensor(N_CHANNELS, SEQ_LEN).data on the Arduino) -- pass input.data directly.\n"
            "inline void treeForestPredict(const float* x, float* outProbs) {\n"
            "  float leaf[TREE_N_CLASSES];\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] = 0.0f;\n"
            "  for (int t = 0; t < TREE_N_TREES; t++) {\n"
            "    treePredictSingle(TREES[t], x, leaf);\n"
            "    for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] += leaf[c];\n"
            "  }\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] /= TREE_N_TREES;\n"
            "}\n\n"
        )

        f.write(
            "// CNN + Tree ensemble: weighted average of the two probability vectors.\n"
            "// alpha is the weight on the CNN (0..1); tune it on a held-out set.\n"
            "inline void ensembleBlend(const float* cnnProbs, const float* treeProbs, float alpha, float* outProbs) {\n"
            "  for (int c = 0; c < TREE_N_CLASSES; c++) outProbs[c] = alpha * cnnProbs[c] + (1.0f - alpha) * treeProbs[c];\n"
            "}\n\n"
            "inline int argmaxProbs(const float* probs, int n) {\n"
            "  int best = 0;\n"
            "  for (int i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;\n"
            "  return best;\n"
            "}\n\n"
        )

        f.write("#endif\n")


# ---------------------------------------------------------------------------
def main():
    (Xf_train, y_train), (Xf_val, y_val), (Xf_test, y_test) = load_data()
    n_features = Xf_train.shape[1]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        make_objective(Xf_train, y_train, Xf_val, y_val),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    print("Best val log-loss:", study.best_value)
    print("Best params:", study.best_params)

    best_params = dict(study.best_params)
    best_params["random_state"] = RANDOM_SEED
    best_params["n_jobs"] = -1

    final_model = RandomForestClassifier(**best_params)
    # Refit on train+val for the deployed model; test (rat 10's reserved
    # test fold) stays untouched for reporting.
    X_fit = np.concatenate([Xf_train, Xf_val])
    y_fit = np.concatenate([y_train, y_val])
    final_model.fit(X_fit, y_fit)

    test_probs = final_model.predict_proba(Xf_test)
    test_pred = np.argmax(test_probs, axis=1)
    print("Test accuracy:", accuracy_score(y_test, test_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    n_nodes_total = sum(e.tree_.node_count for e in final_model.estimators_)
    print(f"Exporting {len(final_model.estimators_)} tree(s), {n_nodes_total} total nodes")

    export_forest_to_cpp(
        final_model,
        n_features=n_features,
        n_classes=len(CLASSES),
        class_names=list(CLASSES.keys()),
        path=OUTPUT_HEADER,
    )
    print(f"Wrote {OUTPUT_HEADER}")


if __name__ == "__main__":
    main()
