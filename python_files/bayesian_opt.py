from pathlib import Path
import numpy as np
import optuna
import os
optuna.logging.set_verbosity(optuna.logging.WARNING)
from logo_splits import load_logo_split, RAT_IDS
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler

CLASSES = {
    "DORSIFLEXION": 0,
    "PLANTARFLEXION": 1,
    "NONE": 2,
}
N_CLASSES = len(CLASSES)

DATA_DIR = Path("./splits")

def load_splits():
    """Henter de færdige, skreddersyede splits fra jeres main branch pipeline."""
    if not (DATA_DIR / "train.npz").exists():
        raise FileNotFoundError("Kør streamer.py først for at generere ./splits mappen!")

    train = np.load(DATA_DIR / "train.npz")
    val = np.load(DATA_DIR / "val.npz")


    return train["X"], train["y"], val["X"], val["y"]

def fits_in_sram(n_trees: int, max_nodes: int, budget_kb: float = 80.0) -> bool:
    """
    Hukommelsesberegning skræddersyet til jeres MondrianForest.h struktur.
    Hver node indeholder: splitDim(2) + threshold(4) + tau(4) + left(2) + right(2)
    + parent(2) + nSamples(4) + classCounts(3 * 4 = 12) = 34 bytes pr. node.
    """
    node_size_bytes = 34


    mem_bytes = n_trees * max_nodes * node_size_bytes


    mem_bytes += (56 * 32 * 4) + (1176 * 4) + 4096

    return (mem_bytes / 1024) <= budget_kb

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial):

        n_trees = trial.suggest_int("n_trees", 3, 25)
        max_nodes = trial.suggest_categorical("max_nodes", [31, 63, 127])
        max_feat = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4])
        model_type = trial.suggest_categorical("model_type", ["rf", "extra"])

        if not fits_in_sram(n_trees, max_nodes, budget_kb=80.0):
            raise optuna.exceptions.TrialPruned()


        calculated_depth = int(np.log2(max_nodes + 1)) - 1

        if model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=calculated_depth,
                max_features=max_feat,
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = ExtraTreesClassifier(
                n_estimators=n_trees,
                max_depth=calculated_depth,
                max_features=max_feat,
                random_state=42,
                n_jobs=-1,
            )

        clf.fit(X_train.reshape(len(X_train), -1), y_train)
        score = (clf.predict(X_val.reshape(len(X_val), -1)) == y_val).mean()

        return float(score)

    return objective

def logo_score_objective(clf, X_feat_builder=None):
    scores = []
    for test_rat in RAT_IDS:
        X_tr, y_tr, X_val, y_val = load_logo_split(test_rat)
        clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
        scores.append((clf.predict(X_val.reshape(len(X_val), -1)) == y_val).mean())
    return float(np.mean(scores))

if __name__ == "__main__":
    print("Loading pre-scaled data from main branch splits...")
    try:
        X_train, y_train, X_val, y_val = load_splits()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print(f"  Train: {X_train.shape[0]} windows | Val: {X_val.shape[0]} windows")
    print(f"  Features per window: {X_train.shape[1]} channels x {X_train.shape[2]} samples")

    print("\nRunning Bayesian Optimization under SRAM Constraints (100 trials)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def print_progress(study, trial):
        if trial.number % 10 == 0:
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                print(f"  Trial {trial.number:3d} | All trials pruned so far (Too heavy for SRAM)")
                return
            print(f"  Trial {trial.number:3d} | Best Val Acc: {study.best_value:.4f} | Params: {study.best_params}")

    study.optimize(
        make_objective(X_train, y_train, X_val, y_val),
        n_trials=100,
        callbacks=[print_progress],
    )

    best = study.best_params
    best_score = study.best_value

    print("\n" + "="*55)
    print("  OPTIMAL ARDUINO CONFIGURATION FOUND")
    print("="*55)
    print(f"  Best Validation Accuracy : {best_score:.4f}")
    print(f"  #define MF_N_TREES        {best['n_trees']}")
    print(f"  #define MF_MAX_NODES      {best['max_nodes']}")
    print(f"  Model Type               : {best['model_type']}")
    print(f"  Max Features Chosen      : {best['max_features']}")

    node_size_bytes = 34
    est_mem = (best['n_trees'] * best['max_nodes'] * node_size_bytes) / 1024
    print(f"  Estimated Tree SRAM      : {est_mem:.2f} KB (Fits comfortably in 256KB)")

    try:
        import matplotlib.pyplot as plt
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Antal træer vs Valideringsnøjagtighed
        trees = [t.params['n_trees'] for t in completed]
        scores = [t.value for t in completed]
        nodes = [t.params['max_nodes'] for t in completed]

        sc = axes[0].scatter(trees, scores, c=nodes, cmap='plasma', alpha=0.7, edgecolors='k')
        cbar = plt.colorbar(sc, ax=axes[0])
        cbar.set_label('MAX_NODES')
        axes[0].set_xlabel('MF_N_TREES')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].set_title('Ensemble Size vs Performance')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Fordeling af de bedste noder i top 20
        top20_nodes = [t.params['max_nodes'] for t in sorted(completed, key=lambda t: t.value, reverse=True)[:20]]
        unique, counts = np.unique(top20_nodes, return_counts=True)
        axes[1].bar([str(u) for u in unique], counts, color='teal', edgecolor='k', alpha=0.8)
        axes[1].set_xlabel('MAX_NODES Capacity')
        axes[1].set_ylabel('Count in Top 20 Trials')
        axes[1].set_title('Optimal Node Budget Distribution')
        axes[1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig('bayesian_opt_results.png', dpi=150)
        print("\n[INFO] Optimization plots saved to 'bayesian_opt_results.png' - klar til rapporten!")
        plt.show()

    except ImportError:
        print("\n  (matplotlib ikke tilgængelig, springer plots over)")
