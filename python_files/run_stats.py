import numpy as np
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q

MODEL_FILES = {
    "CNN": "./outputs/model_CNN_correctness.txt",
    "Focal": "./outputs/model_focal_correctness.txt",
    "Tree": "./outputs/model_tree_correctness.txt",
    "MF": "./outputs/model_MF_correctness.txt",
}

def load_correctness(path):
    with open(path) as f:
        return np.array([int(line.strip()) for line in f if line.strip()])

def run_mcnemar(name_a, preds_a, name_b, preds_b):
    min_len = min(len(preds_a), len(preds_b))
    preds_a = preds_a[:min_len]
    preds_b = preds_b[:min_len]

    both_correct = np.sum((preds_a == 1) & (preds_b == 1))
    a_only = np.sum((preds_a == 1) & (preds_b == 0))
    b_only = np.sum((preds_a == 0) & (preds_b == 1))
    both_wrong = np.sum((preds_a == 0) & (preds_b == 0))

    print(f"\n--- McNemar: {name_a} vs {name_b} ---")
    print(f"  Accuracy  {name_a}: {preds_a.mean()*100:.2f}%  |  {name_b}: {preds_b.mean()*100:.2f}%")
    print(f"  Discordant  {name_a} unique: {a_only}  |  {name_b} unique: {b_only}")

    if a_only + b_only == 0:
        print("  No discordant pairs -> models are identical on this subset, skipping test")
        return

    table = np.array([[both_correct, a_only], [b_only, both_wrong]])
    result = mcnemar(table, exact=False, correction=True)
    sig = "SIGNIFICANT" if result.pvalue < 0.05 else "not significant"
    print(f"  chi2={result.statistic:.4f}   p={result.pvalue:.6f}   -> {sig}")

def run_cochrans_q(model_preds):
    names = list(model_preds.keys())
    min_len = min(len(v) for v in model_preds.values())
    matrix = np.column_stack([model_preds[n][:min_len] for n in names])

    result = cochrans_q(matrix)
    sig = "SIGNIFICANT" if result.pvalue < 0.05 else "not significant"

    print("\n=== Cochran's Q (all models) ===")
    print(f"  Models : {names}")
    print(f"  Samples used: {min_len}")
    print(f"  Q={result.statistic:.4f}   df={len(names)-1}   p={result.pvalue:.6f}   -> {sig}")

def analyze_saved_results():
    model_preds = {name: load_correctness(path) for name, path in MODEL_FILES.items()}

    lengths = {name: len(v) for name, v in model_preds.items()}
    print("Sample counts:", lengths)

    for (name_a, preds_a), (name_b, preds_b) in combinations(model_preds.items(), 2):
        run_mcnemar(name_a, preds_a, name_b, preds_b)

    run_cochrans_q(model_preds)

if __name__ == "__main__":
    analyze_saved_results()
