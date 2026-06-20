"""
generate_tree_cpp.py

Trains a SINGLE decision tree (sklearn DecisionTreeClassifier) on
statistical features (Mean, Min, Max, Std, RMS) extracted from the EMG windows,
tunes it with Bayesian optimisation (Optuna), and exports the fitted tree to a
self-contained C++ header (tree_weights.h).

WARNING: The Arduino C++ code MUST extract these exact same features
in the exact same order before passing the array to treeForestPredict!
"""

import os

import numpy as np
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from streamer import CLASSES, load_or_build_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "./dataset_rats_50w"

POPULATION_RAT_IDS   = [4, 5, 6, 7, 8, 9]
POPULATION_SPLIT_DIR = "./splits_population"

TARGET_RAT_ID    = 10
TARGET_SPLIT_DIR = "./splits_finetune"

INCLUDE_TARGET_TRAIN_IN_TREE = True

RANDOM_SEED      = 10
N_OPTUNA_TRIALS  = 60
OUTPUT_HEADER    = "./cpp_part/tree_weights.h"


# ---------------------------------------------------------------------------
# Feature Extraction (Replaces raw flattening)
# ---------------------------------------------------------------------------
def extract_stats(window):
    """
    window: (N_CHANNELS, SEQ_LEN) raw data.
    Returns a 1D array of statistical features per channel.
    """
    # Calculate stats along the time axis (axis=1) for each channel
    means = np.mean(window, axis=1)
    mins  = np.min(window, axis=1)
    maxs  = np.max(window, axis=1)
    stds  = np.std(window, axis=1)
    rms   = np.sqrt(np.mean(window**2, axis=1))

    # Concatenate all stats into a single 1D array
    # Shape will be: (5 * N_CHANNELS,)
    features = np.concatenate([means, mins, maxs, stds, rms])
    return features.astype(np.float32)


def build_stat_features(X, desc="Extracting stats"):
    # X shape: (N_SAMPLES, N_CHANNELS, SEQ_LEN)
    return np.stack([extract_stats(w) for w in tqdm(X, desc=desc)])


def load_data():
    pop    = load_or_build_splits(DATA_DIR, POPULATION_RAT_IDS, out_dir=POPULATION_SPLIT_DIR)
    target = load_or_build_splits(DATA_DIR, [TARGET_RAT_ID],    out_dir=TARGET_SPLIT_DIR)

    X_train_parts = [pop["train"]["X"], pop["val"]["X"], pop["test"]["X"]]
    y_train_parts = [pop["train"]["y"], pop["val"]["y"], pop["test"]["y"]]

    if INCLUDE_TARGET_TRAIN_IN_TREE:
        X_train_parts.append(target["train"]["X"])
        y_train_parts.append(target["train"]["y"])

    X_train_raw = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    X_val_raw,  y_val  = target["val"]["X"],  target["val"]["y"]
    X_test_raw, y_test = target["test"]["X"], target["test"]["y"]

    # 1. Extract statistical features FIRST
    X_feat_train = build_stat_features(X_train_raw, desc="Extracting train stats")
    X_feat_val   = build_stat_features(X_val_raw,   desc="Extracting val stats")
    X_feat_test  = build_stat_features(X_test_raw,  desc="Extracting test stats")

    # 2. Scale the extracted features, NOT the raw windows
    scaler = StandardScaler()
    Xf_train = scaler.fit_transform(X_feat_train)
    Xf_val   = scaler.transform(X_feat_val)
    Xf_test  = scaler.transform(X_feat_test)

    mode = "population + target-train" if INCLUDE_TARGET_TRAIN_IN_TREE else "population only"
    print(f"Tree training pool : {len(y_train)} windows ({mode})")
    print(f"Val / test         : {len(y_val)} / {len(y_test)} windows  (rat {TARGET_RAT_ID} only)")
    print(f"Features per window: {Xf_train.shape[1]}")

    return (Xf_train, y_train), (Xf_val, y_val), (Xf_test, y_test)


# ---------------------------------------------------------------------------
# Optuna objective — single DecisionTreeClassifier, minimise val log-loss
# ---------------------------------------------------------------------------
def make_objective(Xf_train, y_train, Xf_val, y_val):
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

        model = DecisionTreeClassifier(**params)
        model.fit(Xf_train, y_train)
        val_probs = model.predict_proba(Xf_val)
        return log_loss(y_val, val_probs, labels=list(CLASSES.values()))

    return objective


# ---------------------------------------------------------------------------
# C++ export — same format as before; TREE_N_TREES is always 1
# ---------------------------------------------------------------------------
def tree_to_cpp_nodes(tree, n_classes):
    nodes = []
    for i in range(tree.node_count):
        if tree.feature[i] == -2:          # leaf
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


def export_tree_to_cpp(model, n_features, n_classes, class_names, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w") as f:
        f.write("#ifndef TREE_WEIGHTS_H\n#define TREE_WEIGHTS_H\n\n")
        f.write("// Auto-generated by generate_tree_cpp.py -- do not edit by hand\n")
        f.write("// Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)) + "\n\n")
        f.write(f"#define TREE_N_FEATURES {n_features}\n")
        f.write(f"#define TREE_N_CLASSES  {n_classes}\n")
        f.write( "#define TREE_N_TREES    1\n\n")

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
            f.write(
                f"  {{ {n['feature']}, {n['left']}, {n['right']}, "
                f"{n['threshold']:.6f}f, {{{probs_str}}} }},\n"
            )
        f.write("};\n\n")

        f.write("const TreeNode* const TREES[TREE_N_TREES] = {\n  tree0_nodes\n};\n\n")

        # ── inference helpers ──────────────
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
            "// WARNING: x must be an array of TREE_N_FEATURES extracted statistical features\n"
            "// Order: [all means, all mins, all maxs, all stds, all rms]\n"
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

    print("Best val log-loss :", study.best_value)
    print("Best params       :", study.best_params)

    best_params = dict(study.best_params)
    best_params["random_state"] = RANDOM_SEED

    # Refit on train + val; test stays untouched for final reporting.
    final_model = DecisionTreeClassifier(**best_params)
    final_model.fit(
        np.concatenate([Xf_train, Xf_val]),
        np.concatenate([y_train,  y_val]),
    )

    test_pred = final_model.predict(Xf_test)
    print("Test accuracy     :", accuracy_score(y_test, test_pred))
    print("Confusion matrix  :\n", confusion_matrix(y_test, test_pred))
    print(f"Tree node count   : {final_model.tree_.node_count}")

    export_tree_to_cpp(
        final_model,
        n_features  = n_features,
        n_classes   = len(CLASSES),
        class_names = list(CLASSES.keys()),
        path        = OUTPUT_HEADER,
    )
    print(f"Wrote {OUTPUT_HEADER}")


if __name__ == "__main__":
    main()
