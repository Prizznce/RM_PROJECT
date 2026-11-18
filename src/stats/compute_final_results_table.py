import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 101, 7, 2024, 999, 123, 777, 314, 555, 888]

ALL_POLICIES_ORDER = [
    # (key, display_name, family, is_stochastic, is_deployable)
    ("majority", "Zero-Skill Majority (Always 0)", "Naive Baseline", False, True),
    ("persistence", "Persistence (1-Hr Momentum)", "Naive Baseline", False, True),
    ("sgd", "Online SGD (Logistic Regression)", "Online Linear", False, True),
    ("knn", "Streaming kNN (k=5, W=100)", "Instance-Based", False, True),
    ("always_ht", "Always-HT", "Single Tree", False, True),
    ("hat", "Adaptive Hoeffding Tree (HAT)", "Single Tree", False, True),
    ("cgt_v1", "CGT (Proposed, S1)", "CGT Baseline (v1)", False, True),
    ("cgt_v2", "CGT-v2 (Adaptive Dual-Gate)", "CGT Dual-Gate (v2)", False, True),
    ("cgt_v3", "CGT-v3 (Learned Meta-Gate)", "CGT Learned-Gate (v3)", True, True),
    ("a1", "A1: Threshold Switch", "Switching Policy", False, True),
    ("a2_ucb", "A2: Cost-aware UCB", "Switching Policy", True, True),
    ("oms_mab", "OMS-MAB (Wilson et al., 2024)", "Bandit Baseline", True, True),
    ("a4_fixed", "A4: Learned Router (Calibrated)", "Learned Router", True, True),
    ("srp", "Streaming Random Patches (SRP, 3 trees)", "Ensemble", True, True),
    ("levbag", "Leveraging Bagging (3 trees)", "Ensemble", True, True),
    ("a3_optimized_arf", "A3: Optimized ARF (5 trees)", "Ensemble", True, True),
    ("always_arf", "Always-ARF (Full, 10 trees)", "Ensemble Reference", True, True),
    ("oracle", "Oracle Switch (Upper Bound)**", "Omniscient Ceiling", False, False)
]

FILE_MAPPINGS = {
    "majority": "step_predictions_zero-skill_majority_always_0.csv",
    "persistence": "step_predictions_persistence_1-hr_momentum.csv",
    "sgd": "step_predictions_online_sgd_logistic_regression.csv",
    "knn": "step_predictions_streaming_knn_k=5_w=100.csv",
    "always_ht": "step_predictions_always-ht.csv",
    "hat": "step_predictions_adaptive_hoeffding_tree_hat.csv",
    "cgt_v1": "step_predictions_cgt_proposed_s1.csv",
    "cgt_v2": "step_predictions_cgt-v2_adaptive_dual-gate.csv",
    "cgt_v3": "step_predictions_cgt-v3_learned_meta-gate.csv",
    "a1": "step_predictions_a1_threshold_switch.csv",
    "a2_ucb": "step_predictions_a2_cost-aware_ucb.csv",
    "oms_mab": "step_predictions_oms-mab_wilson_et_al._2024.csv",
    "a4_fixed": "step_predictions_a4_learned_router_calibrated.csv",
    "srp": "step_predictions_streaming_random_patches_srp_3_trees.csv",
    "levbag": "step_predictions_leveraging_bagging_3_trees.csv",
    "a3_optimized_arf": "step_predictions_a3_optimized_arf.csv",
    "always_arf": "step_predictions_always-arf.csv",
    "oracle": "step_predictions_oracle_switch_upper_bound.csv"
}

HYGIENE_KEY_MAP = {
    "majority": "Zero-Skill Majority (Always 0)",
    "persistence": "Persistence (1-Hr Momentum)",
    "sgd": "Online SGD (Logistic Regression)",
    "knn": "Streaming kNN (k=5, W=100)",
    "always_ht": "Always-HT",
    "hat": "Adaptive Hoeffding Tree (HAT)",
    "a1": "A1: Threshold Switch",
    "a2_ucb": "A2: Cost-aware UCB",
    "a3_optimized_arf": "A3: Optimized ARF",
    "cgt_v1": "CGT (Proposed, S1)",
    "cgt_v2": "CGT-v2 (Adaptive Dual-Gate)",
    "cgt_v3": "CGT-v3 (Learned Meta-Gate)",
    "oms_mab": "OMS-MAB (Wilson et al., 2024)",
    "a4_fixed": "A4: Learned Router (Calibrated)",
    "srp": "Streaming Random Patches (SRP, 3 trees)",
    "levbag": "Leveraging Bagging (3 trees)",
    "always_arf": "Always-ARF (10 trees)",
    "oracle": "Oracle Switch (Upper Bound)**"
}

def compile_multiseed_summary():
    """Aggregates all seed runs in results/raw/seeds/ into day8_multiseed_stability.csv."""
    stochastic_keys = ["always_arf", "a3_optimized_arf", "srp", "levbag", "a2_ucb", "a4_fixed", "oms_mab", "cgt_v3"]
    summary_rows = []
    
    for key in stochastic_keys:
        seed_files = glob.glob(f"results/raw/seeds/{key}_seed_*.csv")
        if not seed_files:
            continue
            
        accs = []
        rhos = []
        cpus = []
        
        for sf in seed_files:
            df_seed = pd.read_csv(sf)
            acc = (df_seed['y_true'] == df_seed['y_pred']).mean() * 100.0
            rho = df_seed['gate_open'].mean()
            mean_cpu = (df_seed['cpu_time_ns'].mean() / 1000.0) if 'cpu_time_ns' in df_seed else 0.0
            accs.append(acc)
            rhos.append(rho)
            cpus.append(mean_cpu)
            
        n = len(accs)
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if n > 1 else 0.0
        t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
        se = std_acc / np.sqrt(n)
        ci_low = mean_acc - t_crit * se
        ci_high = mean_acc + t_crit * se
        
        # Display name
        d_name = [item[1] for item in ALL_POLICIES_ORDER if item[0] == key][0]
        
        summary_rows.append({
            "policy_key": key,
            "policy_name": d_name,
            "seeds_tested": n,
            "mean_acc_pct": round(mean_acc, 2),
            "std_acc_pct": round(std_acc, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "ci_95_str": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "min_acc_pct": round(float(np.min(accs)), 2),
            "max_acc_pct": round(float(np.max(accs)), 2),
            "mean_rho": round(float(np.mean(rhos)), 4),
            "mean_cpu_us": round(float(np.mean(cpus)), 2)
        })
        
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("results/tables/day8_multiseed_stability.csv", index=False)
    return df_sum

def generate_master_final_results_table():
    print("=================================================================", flush=True)
    print("GENERATING MASTER FINAL RESULTS TABLE (CSV & LATEX)", flush=True)
    print("=================================================================\n", flush=True)
    
    # 1. Compile seeds
    df_seeds = compile_multiseed_summary()
    seed_stats = {r['policy_key']: r for _, r in df_seeds.iterrows()}
    
    # 2. Load Day 7 measurement hygiene table for latency & memory
    hygiene_path = "results/tables/day7_measurement_hygiene_benchmark.csv"
    if not os.path.exists(hygiene_path):
        raise FileNotFoundError(f"Missing {hygiene_path}")
    df_hygiene = pd.read_csv(hygiene_path)
    hygiene_map = {r['model_name']: r for _, r in df_hygiene.iterrows()}
    
    # Baseline HT accuracy & Oracle accuracy for gap closed
    df_ht = pd.read_csv(os.path.join("results/raw", FILE_MAPPINGS["always_ht"]))
    acc_ht_baseline = (df_ht['y_true'] == df_ht['y_pred']).mean() * 100.0
    
    df_oracle = pd.read_csv(os.path.join("results/raw", FILE_MAPPINGS["oracle"]))
    acc_oracle_ceiling = (df_oracle['y_true'] == df_oracle['y_pred']).mean() * 100.0
    oracle_gap_denominator = acc_oracle_ceiling - acc_ht_baseline # 93.09 - 85.07 = 8.02%
    
    # ARF latency for CPU saving baseline
    arf_latency_us = hygiene_map["Always-ARF (10 trees)"]["mean_latency_us"]
    
    # 3. Compute per-model predictions, balanced acc, and McNemar vs HT
    y_true_ht = df_ht['y_true'].values
    y_pred_ht = df_ht['y_pred'].values
    
    table_rows = []
    
    for key, name, family, is_stoch, is_deploy in ALL_POLICIES_ORDER:
        raw_file = os.path.join("results/raw", FILE_MAPPINGS[key])
        df_pred = pd.read_csv(raw_file)
        y_true = df_pred['y_true'].values
        y_pred = df_pred['y_pred'].values
        n_inst = len(y_true)
        
        # Single-run accuracy
        single_acc = (y_true == y_pred).mean() * 100.0
        
        # Multi-seed accuracy if stochastic
        if is_stoch and key in seed_stats and seed_stats[key]['seeds_tested'] >= 5:
            m_acc = seed_stats[key]['mean_acc_pct']
            s_acc = seed_stats[key]['std_acc_pct']
            acc_str = f"{m_acc:.2f} ± {s_acc:.2f}"
            eval_acc = m_acc
        else:
            acc_str = f"{single_acc:.2f} ± 0.00"
            eval_acc = single_acc
            
        # Balanced accuracy: 0.5 * (TPR + TNR)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = 0.5 * (tpr + tnr) * 100.0
        
        # Hygiene metrics lookup via exact mapping
        hyg_target_name = HYGIENE_KEY_MAP[key]
        hyg_row = hygiene_map[hyg_target_name]
                    
        mean_lat_us = hyg_row['mean_latency_us']
        std_lat_us = hyg_row['std_latency_us']
        lat_str = f"{mean_lat_us:.1f} ± {std_lat_us:.1f}"
        
        total_cpu_sec = (mean_lat_us * n_inst) / 1e6
        cpu_saving_pct = (1.0 - mean_lat_us / arf_latency_us) * 100.0
        peak_heap_mb = hyg_row['model_peak_tracemalloc_mb']
        duty_cycle_rho = hyg_row['arf_active_duty_cycle_rho']
        
        acc_per_cpu_sec = eval_acc / total_cpu_sec if total_cpu_sec > 0 else 0.0
        gap_closed_pct = ((eval_acc - acc_ht_baseline) / oracle_gap_denominator) * 100.0
        
        # McNemar test vs HT
        if key == "always_ht":
            p_val_vs_ht = 1.0
            discordant_b = 0
            discordant_c = 0
        else:
            b = np.sum((y_pred == y_true) & (y_pred_ht != y_true))
            c = np.sum((y_pred != y_true) & (y_pred_ht == y_true))
            discordant_b = int(b)
            discordant_c = int(c)
            # 2-sided binomial test on discordant pairs
            p_val_vs_ht = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
            
        table_rows.append({
            "key": key,
            "policy_name": name,
            "model_family": family,
            "is_deployable": is_deploy,
            "eval_acc": eval_acc,
            "acc_mean_std_str": acc_str,
            "balanced_acc_pct": round(balanced_acc, 2),
            "duty_cycle_rho_pct": duty_cycle_rho if isinstance(duty_cycle_rho, str) else f"{float(duty_cycle_rho)*100:.2f}%",
            "mean_latency_us": mean_lat_us,
            "lat_mean_std_str": lat_str,
            "total_cpu_sec": round(total_cpu_sec, 2),
            "cpu_saving_vs_arf_pct": round(cpu_saving_pct, 1),
            "peak_heap_ram_mb": round(peak_heap_mb, 2),
            "acc_per_cpu_sec": round(acc_per_cpu_sec, 2),
            "gap_closed_pct": round(gap_closed_pct, 2),
            "mcnemar_raw_p": p_val_vs_ht,
            "discordant_b": discordant_b,
            "discordant_c": discordant_c
        })
        
    df_table = pd.DataFrame(table_rows)
    
    # 4. Holm-Bonferroni correction on p-values vs HT
    # Sort non-HT models by raw p-value
    non_ht_indices = [i for i, r in df_table.iterrows() if r['key'] != "always_ht"]
    p_vals = [df_table.loc[i, 'mcnemar_raw_p'] for i in non_ht_indices]
    
    sorted_pairs = sorted(enumerate(p_vals), key=lambda x: x[1])
    m_tests = len(sorted_pairs)
    adj_p_dict = {}
    
    for rank_idx, (orig_pos, p_val) in enumerate(sorted_pairs):
        adj_p = min(1.0, p_val * (m_tests - rank_idx))
        adj_p_dict[non_ht_indices[orig_pos]] = adj_p
        
    # Enforce monotonicity
    curr_max = 0.0
    for rank_idx, (orig_pos, _) in enumerate(sorted_pairs):
        idx = non_ht_indices[orig_pos]
        curr_max = max(curr_max, adj_p_dict[idx])
        adj_p_dict[idx] = min(1.0, curr_max)
        
    df_table['holm_adj_p_vs_ht'] = 1.0
    for idx, adj_p in adj_p_dict.items():
        df_table.loc[idx, 'holm_adj_p_vs_ht'] = adj_p
        
    # Format Holm p string
    def fmt_p(p):
        if p == 1.0:
            return "ref"
        elif p < 0.001:
            return f"{p:.2e}***"
        elif p < 0.01:
            return f"{p:.4f}**"
        elif p < 0.05:
            return f"{p:.4f}*"
        else:
            return f"{p:.4f} (ns)"
    df_table['p_str_vs_ht'] = df_table['holm_adj_p_vs_ht'].apply(fmt_p)
    
    # 5. Pareto Dominance Analysis
    # Objectives: Maximize eval_acc, Minimize mean_latency_us
    # Deployable learning models only (excludes non-learning dummy majority baseline)
    deployable = df_table[df_table['is_deployable'] & (df_table['key'] != 'majority')].copy()
    
    non_dominated = []
    for i, row_a in deployable.iterrows():
        dominated = False
        for j, row_b in deployable.iterrows():
            if i == j:
                continue
            # B dominates A if B has >= acc AND <= lat, and at least one strict
            if (row_b['eval_acc'] >= row_a['eval_acc'] and row_b['mean_latency_us'] <= row_a['mean_latency_us']) and \
               (row_b['eval_acc'] > row_a['eval_acc'] or row_b['mean_latency_us'] < row_a['mean_latency_us']):
                dominated = True
                break
        if not dominated:
            non_dominated.append(row_a['key'])
            
    df_table['pareto_optimal_acc_lat'] = df_table['key'].apply(lambda k: k in non_dominated)
    
    # Save final CSV
    out_csv = "results/tables/final_results_table.csv"
    df_table.to_csv(out_csv, index=False)
    print(f"Saved Master Final Results CSV to: {out_csv}")
    
    # 6. Generate LaTeX Table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Benchmark across 18 Streaming Policies on the Frozen Test Stream (24,528 instances). Accuracy reflects multi-seed evaluation (mean $\pm$ std over 10 seeds) for stochastic models and deterministic evaluation for single-run models. Non-dominated Pareto optimal policies in the (Accuracy vs.\ Latency) objective space are highlighted in \textbf{bold}. Holm-adjusted McNemar significance against Always-HT is denoted by *** ($p < 0.001$), ** ($p < 0.01$), * ($p < 0.05$), and ns (not significant).}",
        r"\label{tab:final_results}",
        r"\small",
        r"\begin{tabular}{llcccccccr}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Family} & \textbf{Acc (\%)} & \textbf{Bal Acc} & \textbf{$\rho_{\text{ARF}}$ (\%)} & \textbf{Latency ($\mu$s)} & \textbf{CPU Sav} & \textbf{RAM} & \textbf{Gap Cl} & \textbf{Adj $p$ vs HT} \\",
        r"\midrule"
    ]
    
    current_family = ""
    for _, row in df_table.iterrows():
        fam = row['model_family']
        name = row['policy_name']
        if row['pareto_optimal_acc_lat']:
            name = rf"\textbf{{{name}}}"
            
        acc_s = row['acc_mean_std_str'].replace("±", r"$\pm$")
        bal_s = f"{row['balanced_acc_pct']:.2f}"
        rho_s = str(row['duty_cycle_rho_pct']).replace("%", r"\%")
        lat_s = row['lat_mean_std_str'].replace("±", r"$\pm$")
        cpu_sav = f"{row['cpu_saving_vs_arf_pct']:.1f}\\%"
        ram_s = f"{row['peak_heap_ram_mb']:.2f}M"
        gap_s = f"{row['gap_closed_pct']:+.1f}\\%"
        p_s = row['p_str_vs_ht']
        if "***" in p_s:
            p_s = p_s.replace("***", r"$^{***}$")
        elif "**" in p_s:
            p_s = p_s.replace("**", r"$^{**}$")
        elif "*" in p_s:
            p_s = p_s.replace("*", r"$^*$")
        
        line = f"{name} & {fam} & {acc_s} & {bal_s} & {rho_s} & {lat_s} & {cpu_sav} & {ram_s} & {gap_s} & {p_s} \\\\"
        tex_lines.append(line)
        
    tex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{10}{l}{\footnotesize $^*$Continuous ensembles (A3, SRP, LevBag, Always-ARF) evaluate all ensemble trees per instance; not a dynamic switcher.} \\",
        r"\multicolumn{10}{l}{\footnotesize $^{**}$Oracle Gated Switch utilizes counterfactual ground truth; non-deployable theoretical ceiling.}",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    out_tex = "results/tables/final_results_table.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"Saved LaTeX-ready table to: {out_tex}")
    
    # Display preview in console
    preview_cols = ['policy_name', 'acc_mean_std_str', 'duty_cycle_rho_pct', 'lat_mean_std_str', 'peak_heap_ram_mb', 'gap_closed_pct', 'p_str_vs_ht', 'pareto_optimal_acc_lat']
    print("\n" + df_table[preview_cols].to_string(index=False))

if __name__ == "__main__":
    generate_master_final_results_table()
