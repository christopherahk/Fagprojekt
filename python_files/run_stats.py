import numpy as np
import os
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q

def analyze_saved_results():
    data_dir = "./outputs"
    
    file_a = os.path.join(data_dir, "model_CNN_correctness.txt")
    file_b = os.path.join(data_dir, "model_forrest_correctness.txt")
    file_c = os.path.join(data_dir, "model_gamma_correctness.txt")
    
    if not (os.path.exists(file_a) and os.path.exists(file_b)):
        print("Ensure at least model_alpha and model_beta text data files exist before executing analysis.")
        return

    correct_A = np.loadtxt(file_a, dtype=int)
    correct_B = np.loadtxt(file_b, dtype=int)
    
    min_len = min(len(correct_A), len(correct_B))
    correct_A = correct_A[:min_len]
    correct_B = correct_B[:min_len]

    print("MCNEMAR BIPARTITE TEST RESULTS")
    
    both_correct = np.sum((correct_A == 1) & (correct_B == 1))
    a_only = np.sum((correct_A == 1) & (correct_B == 0))
    b_only = np.sum((correct_A == 0) & (correct_B == 1))
    both_incorrect = np.sum((correct_A == 0) & (correct_B == 0))
    
    contingency_table = [
        [both_correct, a_only],
        [b_only, both_incorrect]
    ]
    
    mc_result = mcnemar(contingency_table, exact=True)
    print(f"Model A Accuracy: {np.mean(correct_A)*100:.2f}%")
    print(f"Model B Accuracy: {np.mean(correct_B)*100:.2f}%")
    print(f"Discordant cells count -> A unique: {a_only} | B unique: {b_only}")
    print(f"McNemar Computed p-value: {mc_result.pvalue:.6f}")
    
    if mc_result.pvalue < 0.05:
        print("Conclusion: Statistically significant performance variation detected.")
    else:
        print("Conclusion: Variation is not statistically significant.")

    if os.path.exists(file_c):
        print("COCHRAN Q COHORT TEST RESULTS")
        
        correct_C = np.loadtxt(file_c, dtype=int)
        min_len = min(min_len, len(correct_C))
        
        cochran_matrix = np.column_stack([
            correct_A[:min_len], 
            correct_B[:min_len], 
            correct_C[:min_len]
        ])
        
        q_stat, q_pvalue = cochrans_q(cochran_matrix)
        print(f"Model C Accuracy: {np.mean(correct_C[:min_len])*100:.2f}%")
        print(f"Cochran's Q Evaluation Metric: {q_stat:.4f}")
        print(f"Cochran's Q Computed p-value:   {q_pvalue:.6f}")
        
        if q_pvalue < 0.05:
            print("Conclusion: Overall distinct operational deviations observed inside this system.")
        else:
            print("Conclusion: Distribution of results matches general homogeneity expectations.")

if __name__ == "__main__":
    analyze_saved_results()