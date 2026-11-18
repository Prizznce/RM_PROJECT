import glob
import os
import pandas as pd
from scipy import stats

def run_statistical_tests():
    print("=== Running Statistical Significance Tests on Prequential Steps ===")
    
    # Load step files
    cgt_df = pd.read_csv("results/raw/steps_cgt_proposed_s1.csv")
    arf_df = pd.read_csv("results/raw/steps_always-arf.csv")
    ht_df = pd.read_csv("results/raw/steps_always-ht.csv")
    ablation_df = pd.read_csv("results/raw/steps_cgt_ablation_fixed_schedule.csv")
    a2_df = pd.read_csv("results/raw/steps_a2_cost-aware_ucb.csv")
    
    cgt_plus_path = "results/raw/steps_cgt-plus_optimized_hybrid.csv"
    has_plus = os.path.exists(cgt_plus_path)
    cgt_plus_df = pd.read_csv(cgt_plus_path) if has_plus else None

    pairs = [
        ("CGT (S1) vs Always-ARF", cgt_df['sliding_acc'], arf_df['sliding_acc']),
        ("CGT (S1) vs Always-HT", cgt_df['sliding_acc'], ht_df['sliding_acc']),
        ("CGT (S1) vs Fixed Ablation", cgt_df['sliding_acc'], ablation_df['sliding_acc']),
        ("CGT (S1) vs A2 Cost-UCB", cgt_df['sliding_acc'], a2_df['sliding_acc'])
    ]

    if has_plus:
        pairs.extend([
            ("CGT-Plus vs Always-ARF", cgt_plus_df['sliding_acc'], arf_df['sliding_acc']),
            ("CGT-Plus vs Always-HT", cgt_plus_df['sliding_acc'], ht_df['sliding_acc']),
            ("CGT-Plus vs CGT (S1)", cgt_plus_df['sliding_acc'], cgt_df['sliding_acc']),
        ])
    
    results = []
    for name, s1, s2 in pairs:
        # Paired Wilcoxon signed-rank test
        diff = s1 - s2
        if (diff != 0).any():
            stat, p_val = stats.wilcoxon(s1, s2)
            t_stat, t_pval = stats.ttest_rel(s1, s2)
        else:
            stat, p_val, t_stat, t_pval = 0.0, 1.0, 0.0, 1.0
            
        results.append({
            'Comparison': name,
            'Mean_Diff_Acc_pct': round(float(diff.mean()), 2),
            'Wilcoxon_W': round(float(stat), 2),
            'Wilcoxon_p_value': f"{p_val:.4e}",
            'Paired_t_stat': round(float(t_stat), 2),
            'Paired_t_p_value': f"{t_pval:.4e}",
            'Significant_at_0.05': bool(p_val < 0.05)
        })
        print(f"{name}: Mean Diff = {diff.mean():+.2f}%, Wilcoxon p = {p_val:.4e} (Sig: {p_val < 0.05})")

    res_df = pd.DataFrame(results)
    out_path = "results/tables/statistical_tests.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nStatistical test results saved to {out_path}")
    return res_df

if __name__ == '__main__':
    run_statistical_tests()
