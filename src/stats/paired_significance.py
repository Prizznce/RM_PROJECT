"""
Day 8 - Task 1: Paired Statistical Significance on Per-Instance Correctness.
Computes:
  1. Contingency tables (both right, discordant pairs b and c, both wrong)
  2. McNemar test with continuity correction and exact binomial p-values
  3. Odds Ratio (OR = b / c) with 95% confidence intervals
  4. Moving-Block Bootstrap for accuracy differences across block lengths L in {24, 48, 72, 168} hours
     (Accounting for hourly autocorrelation in grid telemetry)
  5. Holm-Bonferroni Family-Wise Error Rate (FWER) correction across the 8 test pairs
"""

import os
import math
import numpy as np
import pandas as pd
from scipy import stats

def compute_mcnemar(b, c):
    """Computes McNemar test statistic and p-value."""
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, 1.0, 1.0, 1.0
    
    # Exact binomial two-sided test
    # Under H0, b ~ Binomial(b + c, 0.5)
    binom_p = stats.binomtest(b, n_discordant, 0.5, alternative='two-sided').pvalue

    # Continuity-corrected chi-square statistic
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n_discordant
    chi2_p = stats.chi2.sf(chi2_stat, df=1)

    # Odds ratio OR = b / c with 95% log-normal CI
    if c == 0:
        odds_ratio = float('inf')
        ci_lower = float('inf')
        ci_upper = float('inf')
    elif b == 0:
        odds_ratio = 0.0
        ci_lower = 0.0
        ci_upper = 0.0
    else:
        odds_ratio = b / c
        se_log_or = math.sqrt(1.0 / b + 1.0 / c)
        ci_lower = math.exp(math.log(odds_ratio) - 1.96 * se_log_or)
        ci_upper = math.exp(math.log(odds_ratio) + 1.96 * se_log_or)

    return chi2_stat, binom_p, odds_ratio, ci_lower, ci_upper

def moving_block_bootstrap(diff_series, block_length=24, n_bootstraps=2000, seed=42):
    """
    Moving-Block Bootstrap (Kunsch, 1989; Politis & Romano, 1994)
    Preserves time-series autocorrelation within blocks of length L.
    Returns bootstrap mean, 95% CI (percentile), and two-sided bootstrap p-value.
    """
    rng = np.random.RandomState(seed)
    N = len(diff_series)
    num_blocks = N - block_length + 1
    blocks = [diff_series[i:i + block_length] for i in range(num_blocks)]
    
    k_blocks = int(math.ceil(N / block_length))
    boot_means = np.zeros(n_bootstraps)

    for b in range(n_bootstraps):
        sampled_block_indices = rng.randint(0, num_blocks, size=k_blocks)
        sampled_stream = np.concatenate([blocks[idx] for idx in sampled_block_indices])[:N]
        boot_means[b] = np.mean(sampled_stream)

    ci_lower = np.percentile(boot_means, 2.5) * 100.0
    ci_upper = np.percentile(boot_means, 97.5) * 100.0
    boot_mean_pct = np.mean(boot_means) * 100.0

    # Empirical p-value for H0: mean == 0
    p_le = np.mean(boot_means <= 0.0)
    p_ge = np.mean(boot_means >= 0.0)
    boot_p = 2.0 * min(p_le, p_ge)
    boot_p = min(max(boot_p, 1.0 / n_bootstraps), 1.0)

    return boot_mean_pct, ci_lower, ci_upper, boot_p

def run_paired_significance_suite():
    print("=================================================================", flush=True)
    print("DAY 8 - TASK 1: PAIRED STATISTICAL SIGNIFICANCE & BLOCK BOOTSTRAP", flush=True)
    print("=================================================================\n", flush=True)

    raw_dir = "results/raw"
    cgt_v3_path = os.path.join(raw_dir, "step_predictions_cgt-v3_learned_meta-gate.csv")
    if not os.path.exists(cgt_v3_path):
        raise FileNotFoundError(f"Missing {cgt_v3_path}")

    df_v3 = pd.read_csv(cgt_v3_path)
    y_true = df_v3["y_true"].values
    corr_v3 = (df_v3["y_pred"].values == y_true).astype(int)
    N = len(y_true)

    comparator_files = [
        ("Always-HT", os.path.join(raw_dir, "step_predictions_always-ht.csv")),
        ("CGT-v1 (S1)", os.path.join(raw_dir, "step_predictions_cgt_proposed_s1.csv")),
        ("CGT-v2", os.path.join(raw_dir, "step_predictions_cgt-v2_adaptive_dual-gate.csv")),
        ("Always-ARF", os.path.join(raw_dir, "step_predictions_always-arf.csv")),
        ("OMS-MAB", os.path.join(raw_dir, "step_predictions_oms-mab_wilson_et_al._2024.csv")),
        ("A3: Optimized ARF", os.path.join(raw_dir, "step_predictions_a3_optimized_arf.csv")),
        ("Online SGD", os.path.join(raw_dir, "step_predictions_online_sgd_logistic_regression.csv")),
        ("Persistence 1-Hr", os.path.join(raw_dir, "step_predictions_persistence_1-hr_momentum.csv"))
    ]

    results = []

    for name, fpath in comparator_files:
        if not os.path.exists(fpath):
            print(f"Warning: Missing comparator {fpath}", flush=True)
            continue
        df_comp = pd.read_csv(fpath)
        corr_comp = (df_comp["y_pred"].values == y_true).astype(int)

        # Contingency counts
        # b: CGT-v3 right (1), Comparator wrong (0)
        # c: CGT-v3 wrong (0), Comparator right (1)
        b = int(np.sum((corr_v3 == 1) & (corr_comp == 0)))
        c = int(np.sum((corr_v3 == 0) & (corr_comp == 1)))
        both_right = int(np.sum((corr_v3 == 1) & (corr_comp == 1)))
        both_wrong = int(np.sum((corr_v3 == 0) & (corr_comp == 0)))

        acc_v3 = np.mean(corr_v3) * 100.0
        acc_comp = np.mean(corr_comp) * 100.0
        acc_diff_points = acc_v3 - acc_comp

        # McNemar test
        chi2_stat, mcnemar_p, odds_ratio, or_ci_lo, or_ci_hi = compute_mcnemar(b, c)

        # Moving-Block Bootstrap across L in {24, 48, 72, 168}
        diff_series = corr_v3 - corr_comp
        boot_stats = {}
        for L in [24, 48, 72, 168]:
            b_mean, b_lo, b_hi, b_p = moving_block_bootstrap(diff_series, block_length=L, n_bootstraps=2000, seed=42)
            boot_stats[L] = (b_lo, b_hi, b_p)

        results.append({
            "pair": f"CGT-v3 vs {name}",
            "comparator": name,
            "cgt_v3_acc_pct": round(acc_v3, 2),
            "comparator_acc_pct": round(acc_comp, 2),
            "acc_diff_pct": round(acc_diff_points, 2),
            "discordant_b_v3_only": b,
            "discordant_c_comp_only": c,
            "odds_ratio": round(odds_ratio, 3),
            "odds_ratio_95ci": f"[{or_ci_lo:.3f}, {or_ci_hi:.3f}]",
            "mcnemar_raw_p": mcnemar_p,
            "boot_ci_24h": f"[{boot_stats[24][0]:+.2f}, {boot_stats[24][1]:+.2f}]",
            "boot_p_24h": boot_stats[24][2],
            "boot_ci_48h": f"[{boot_stats[48][0]:+.2f}, {boot_stats[48][1]:+.2f}]",
            "boot_ci_72h": f"[{boot_stats[72][0]:+.2f}, {boot_stats[72][1]:+.2f}]",
            "boot_ci_168h": f"[{boot_stats[168][0]:+.2f}, {boot_stats[168][1]:+.2f}]",
            "boot_p_168h": boot_stats[168][2]
        })

    res_df = pd.DataFrame(results)

    # Apply Holm-Bonferroni correction across the family of tests
    # We apply it to both McNemar p-values and 168h Block Bootstrap p-values
    m = len(res_df)
    
    # 1. Holm on McNemar p
    order_mcn = np.argsort(res_df["mcnemar_raw_p"].values)
    adj_p_mcn = np.zeros(m)
    running_max = 0.0
    for rank, idx in enumerate(order_mcn):
        pval = res_df.loc[idx, "mcnemar_raw_p"]
        adjusted = min(pval * (m - rank), 1.0)
        running_max = max(running_max, adjusted)
        adj_p_mcn[idx] = min(running_max, 1.0)
    res_df["mcnemar_holm_adj_p"] = adj_p_mcn

    # 2. Holm on 168h Block Bootstrap p
    order_boot = np.argsort(res_df["boot_p_168h"].values)
    adj_p_boot = np.zeros(m)
    running_max_boot = 0.0
    for rank, idx in enumerate(order_boot):
        pval = res_df.loc[idx, "boot_p_168h"]
        adjusted = min(pval * (m - rank), 1.0)
        running_max_boot = max(running_max_boot, adjusted)
        adj_p_boot[idx] = min(running_max_boot, 1.0)
    res_df["boot_168h_holm_adj_p"] = adj_p_boot

    # Formatted console output
    print("--- PAIRED HYPOTHESIS TESTS (CGT-v3 vs ALL COMPARATORS) ---")
    display_cols = [
        "pair", "acc_diff_pct", "discordant_b_v3_only", "discordant_c_comp_only",
        "odds_ratio", "mcnemar_holm_adj_p", "boot_ci_168h", "boot_168h_holm_adj_p"
    ]
    print(res_df[display_cols].to_string(index=False))

    os.makedirs("results/tables", exist_ok=True)
    out_csv = "results/tables/day8_paired_significance.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved paired statistical results to {out_csv}", flush=True)

    # Sensitivity across block sizes table
    sensitivity_rows = []
    for _, row in res_df.iterrows():
        sensitivity_rows.append({
            "pair": row["pair"],
            "diff_pct": row["acc_diff_pct"],
            "iid_mcnemar_p": f"{row['mcnemar_raw_p']:.2e}",
            "ci_24h_diurnal": row["boot_ci_24h"],
            "ci_48h_2day": row["boot_ci_48h"],
            "ci_72h_3day": row["boot_ci_72h"],
            "ci_168h_weekly": row["boot_ci_168h"],
            "holm_adj_p_weekly": f"{row['boot_168h_holm_adj_p']:.2e}" if row['boot_168h_holm_adj_p'] >= 0.001 else "<0.001"
        })
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_csv = "results/tables/day8_block_bootstrap_sensitivity.csv"
    sens_df.to_csv(sens_csv, index=False)
    print(f"Saved block bootstrap sensitivity analysis to {sens_csv}\n", flush=True)

if __name__ == "__main__":
    run_paired_significance_suite()

