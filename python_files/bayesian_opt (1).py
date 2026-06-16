
from pathlib import Path
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from logo_splits import load_logo_split, RAT_IDS
except ImportError:
    raise ImportError("Place this file next to logo_splits.py and run logo_splits.py first.")

N_CLASSES = 3


def fits_in_sram(n_trees: int, max_nodes: int,
                 budget_kb: float = 200.0) -> bool:
  # estiamted memory usage based on MondrianForest.h structure: probs to claude
    node_bytes  = 32
    range_bytes = 1176 * 2 * 2   # fMin + fMax, int16
    tree_bytes  = max_nodes * node_bytes + range_bytes + 2
    total_bytes = (n_trees * tree_bytes
                   + 56 * 32 * 4    # values[]
                   + 1176 * 4       # features[]
                   + 2048)          # overhead
    return (total_bytes / 1024) <= budget_kb


# LORO objective

def loro_score(clf, flatten=True) -> float:
    """
    Evaluate clf with leave-one-rat-out cross-validation.
    Uses pre-built splits from logo_splits.py (correct per-fold scaling).
    """
    scores = []
    for test_rat in RAT_IDS:
        X_tr, y_tr, X_val, y_val = load_logo_split(test_rat)
        if flatten:
            X_tr  = X_tr.reshape(len(X_tr),   -1)
            X_val = X_val.reshape(len(X_val), -1)
        clf.fit(X_tr, y_tr)
        scores.append((clf.predict(X_val) == y_val).mean())
    return float(np.mean(scores))


def make_objective():
    def objective(trial):
        n_trees    = trial.suggest_int("n_trees",   3, 25)
        max_nodes  = trial.suggest_categorical("max_nodes", [31, 63, 127])
        max_feat   = trial.suggest_categorical(
                         "max_features", ["sqrt", "log2", 0.2, 0.4])
        model_type = trial.suggest_categorical("model_type", ["rf", "extra"])

        if not fits_in_sram(n_trees, max_nodes):
            raise optuna.exceptions.TrialPruned()

        depth = int(np.log2(max_nodes + 1)) - 1

        if model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=depth,
                max_features=max_feat, random_state=42, n_jobs=-1)
        else:
            clf = ExtraTreesClassifier(
                n_estimators=n_trees, max_depth=depth,
                max_features=max_feat, random_state=42, n_jobs=-1)


        return loro_score(clf)

    return objective



if __name__ == "__main__":
    print(f"Running Bayesian optimisation over {len(RAT_IDS)} LORO folds "
          f"(rats {RAT_IDS})...")
    print("Each trial trains on 5 rats and tests on 1,  repeated for all 7.\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def print_progress(study, trial):
        if trial.number % 5 == 0:
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                print(f"  Trial {trial.number:3d} | all pruned so far")
                return
            print(f"  Trial {trial.number:3d} | "
                  f"best LORO acc: {study.best_value:.4f} | "
                  f"params: {study.best_params}")

    study.optimize(
        make_objective(),
        n_trials=100,
        callbacks=[print_progress],
    )

    best       = study.best_params
    best_score = study.best_value

    print("\n" + "="*55)
    print("  BEST RESULT  (LORO cross-validation)")
    print("="*55)
    print(f"  LORO accuracy  : {best_score:.4f}")
    print(f"  n_trees        : {best['n_trees']}")
    print(f"  max_nodes      : {best['max_nodes']}")
    print(f"  model_type     : {best['model_type']}")
    print(f"  max_features   : {best['max_features']}")

    depth = int(np.log2(best['max_nodes'] + 1)) - 1
    node_bytes = 32
    range_bytes = 1176 * 2 * 2
    tree_kb = (best['n_trees'] *
               (best['max_nodes'] * node_bytes + range_bytes + 2)) / 1024
    print(f"\n  Estimated tree SRAM : {tree_kb:.1f} KB")
    print(f"  Fits in 256 KB      : {fits_in_sram(best['n_trees'], best['max_nodes'])}")
    print(f"\n  Add to Fagprojekt.ino:")
    print(f"    #define MF_N_TREES   {best['n_trees']}")
    print(f"    #define MF_MAX_NODES {best['max_nodes']}")

# 10 best
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    top10 = sorted(completed, key=lambda t: t.value, reverse=True)[:10]

    print(f"\n  Top 10 trials:")
    print(f"  {'rank':>4}  {'LORO':>6}  {'trees':>6}  "
          f"{'nodes':>6}  {'type':>6}  {'feat':>6}  {'KB':>6}")
    print("  " + "-"*50)
    for rank, t in enumerate(top10, 1):
        p  = t.params
        kb = (p['n_trees'] * (p['max_nodes'] * node_bytes
              + range_bytes + 2)) / 1024
        print(f"  {rank:4d}  {t.value:.4f}  {p['n_trees']:6d}  "
              f"{p['max_nodes']:6d}  {p['model_type']:>6}  "
              f"{str(p['max_features']):>6}  {kb:6.1f}")

# plotting
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        trees  = [t.params['n_trees']   for t in completed]
        nodes  = [t.params['max_nodes'] for t in completed]
        scores = [t.value               for t in completed]

        sc = axes[0].scatter(trees, scores, c=nodes,
                             cmap='plasma', alpha=0.7, edgecolors='k')
        plt.colorbar(sc, ax=axes[0], label='max_nodes')
        axes[0].set_xlabel('n_trees')
        axes[0].set_ylabel('LOGO accuracy')
        axes[0].set_title('Ensemble size vs LOGO accuracy')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        top20_nodes = [t.params['max_nodes']
                       for t in sorted(completed,
                                       key=lambda t: t.value,
                                       reverse=True)[:20]]
        unique, counts = np.unique(top20_nodes, return_counts=True)
        axes[1].bar([str(u) for u in unique], counts,
                    color='teal', edgecolor='k', alpha=0.8)
        axes[1].set_xlabel('max_nodes')
        axes[1].set_ylabel('count in top 20')
        axes[1].set_title('Node budget in top 20 trials')
        axes[1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig('bayesian_opt_logo_results.png', dpi=150)
        print("\n  Plot saved to bayesian_opt_logo_results.png")
        plt.show()

    except ImportError:
        print("\n  (matplotlib not available, skipping plot)")
