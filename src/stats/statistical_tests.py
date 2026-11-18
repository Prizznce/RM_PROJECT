import glob
import os
import pandas as pd
from scipy import stats

import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

def run_block_aggregated_tests():
    """
    Computes rigorous non-overlapping block-aggregated statistical significance tests
    (Daily N=1,022 blocks and Weekly N=146 blocks) to control for hourly serial autocorrelation.
    """
    pred_dir = "results/raw"
    cgt_v3_path = os.path.join(pred_dir, "step_predictions_cgt-v3_learned_meta-gate.csv")
    ht_path = os.path.join(pred_dir, "step_predictions_always-ht.csv")
    
    if not (os.path.exists(cgt_v3_path) and os.path.exists(ht_path)):
        return None

    ht_df = pd.read_csv(ht_path)
    y_true = ht_df["y_true"].values
    ht_corr = (ht_df["y_pred"].values == y_true).astype(float)
    n_total = len(y_true)

    target_models = {
        "CGT-v3 (Learned Meta-Gate)": "step_predictions_cgt-v3_learned_meta-gate.csv",
        "CGT-v2 (Adaptive Dual-Gate)": "step_predictions_cgt-v2_adaptive_dual-gate.csv",
        "CGT-v1 (Pure ADWIN)": "step_predictions_cgt_proposed_s1.csv",
        "ADWIN-HT (Adaptive Tree)": "step_predictions_adaptive_hoeffding_tree_hat.csv",
        "ARF-3 (3 Trees)": "step_predictions_arf-3_3_trees.csv",
        "A3: Optimized ARF (5 trees)": "step_predictions_a3_optimized_arf.csv",
        "Always-ARF (10 trees)": "step_predictions_always-arf.csv",
        "OMS-MAB (Wilson et al., 2024)": "step_predictions_oms-mab_wilson_et_al._2024.csv",
    }

    block_results = []

    # 1. Models vs Always-HT Baseline
    for block_name, block_size in [("Daily (24h)", 24), ("Weekly (168h)", 168)]:
        n_blocks = n_total // block_size
        ht_blocks = ht_corr[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1) * 100.0

        for m_name, f_name in target_models.items():
            f_path = os.path.join(pred_dir, f_name)
            if not os.path.exists(f_path):
                continue
            m_df = pd.read_csv(f_path)
            m_corr = (m_df["y_pred"].values == y_true).astype(float)
            m_blocks = m_corr[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1) * 100.0
            
            diff = m_blocks - ht_blocks
            mean_gain = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            cohen_d = float(mean_gain / std_diff) if std_diff > 0 else 0.0

            non_zero = diff[diff != 0]
            if len(non_zero) > 0:
                w_stat, w_pval = stats.wilcoxon(non_zero)
                t_stat, t_pval = stats.ttest_rel(m_blocks, ht_blocks)
            else:
                w_stat, w_pval, t_stat, t_pval = 0.0, 1.0, 0.0, 1.0

            block_results.append({
                "Test_Type": "Model vs Always-HT",
                "Aggregation_Window": block_name,
                "Block_Count_N": n_blocks,
                "Comparison": f"{m_name} vs Always-HT",
                "Mean_Accuracy_Gain_pct": round(mean_gain, 3),
                "Paired_Cohen_d": round(cohen_d, 3),
                "Wilcoxon_W": round(float(w_stat), 2),
                "Wilcoxon_p_value": f"{w_pval:.4e}",
                "Paired_t_stat": round(float(t_stat), 2),
                "Paired_t_p_value": f"{t_pval:.4e}",
                "Significant_at_0.05": bool(w_pval < 0.05)
            })

    # 2. CGT-v3 Head-to-Head vs Key Baselines (Weekly Blocks N=146)
    cgt_df = pd.read_csv(cgt_v3_path)
    cgt_corr = (cgt_df["y_pred"].values == y_true).astype(float)
    n_w = n_total // 168
    cgt_w_blocks = cgt_corr[:n_w * 168].reshape(n_w, 168).mean(axis=1) * 100.0

    direct_comparisons = [
        ("CGT-v3 vs ADWIN-HT Alone", "step_predictions_adaptive_hoeffding_tree_hat.csv"),
        ("CGT-v3 vs ARF-3 (3 Trees)", "step_predictions_arf-3_3_trees.csv"),
        ("CGT-v3 vs Always-HT", "step_predictions_always-ht.csv"),
        ("CGT-v3 vs Always-ARF (10 trees)", "step_predictions_always-arf.csv"),
        ("CGT-v3 vs OMS-MAB (Wilson et al.)", "step_predictions_oms-mab_wilson_et_al._2024.csv")
    ]

    for comp_name, comp_file in direct_comparisons:
        comp_path = os.path.join(pred_dir, comp_file)
        if not os.path.exists(comp_path):
            continue
        comp_df = pd.read_csv(comp_path)
        comp_corr = (comp_df["y_pred"].values == y_true).astype(float)
        comp_w_blocks = comp_corr[:n_w * 168].reshape(n_w, 168).mean(axis=1) * 100.0

        diff = cgt_w_blocks - comp_w_blocks
        mean_gain = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        cohen_d = float(mean_gain / std_diff) if std_diff > 0 else 0.0

        non_zero = diff[diff != 0]
        if len(non_zero) > 0:
            w_stat, w_pval = stats.wilcoxon(non_zero)
            t_stat, t_pval = stats.ttest_rel(cgt_w_blocks, comp_w_blocks)
        else:
            w_stat, w_pval, t_stat, t_pval = 0.0, 1.0, 0.0, 1.0

        block_results.append({
            "Test_Type": "CGT-v3 Direct Head-to-Head",
            "Aggregation_Window": "Weekly (168h)",
            "Block_Count_N": n_w,
            "Comparison": comp_name,
            "Mean_Accuracy_Gain_pct": round(mean_gain, 3),
            "Paired_Cohen_d": round(cohen_d, 3),
            "Wilcoxon_W": round(float(w_stat), 2),
            "Wilcoxon_p_value": f"{w_pval:.4e}",
            "Paired_t_stat": round(float(t_stat), 2),
            "Paired_t_p_value": f"{t_pval:.4e}",
            "Significant_at_0.05": bool(w_pval < 0.05)
        })

    out_df = pd.DataFrame(block_results)
    out_path = "results/tables/statistical_tests_block_aggregated.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Block-aggregated statistical tests saved to {out_path}")
    return out_df

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

    # Also run block-aggregated tests
    run_block_aggregated_tests()
    return res_df

if __name__ == '__main__':
    run_statistical_tests()
