import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_friedman_nemenyi():
    print("=================================================================", flush=True)
    print("DAY 8 TASK 3: MULTI-POLICY FRIEDMAN TEST & NEMENYI POST-HOC", flush=True)
    print("=================================================================\n", flush=True)

    # 1. Load dataset timestamps
    data_df = pd.read_csv("data/energy_dataset_clean.csv")
    data_df['Parsed_Timestamp'] = pd.to_datetime(data_df['Timestamp'], format='mixed')
    data_df = data_df.sort_values('Parsed_Timestamp').reset_index(drop=True)

    val_len = 10512
    test_timestamps = data_df['Parsed_Timestamp'].values[24 + val_len:]
    n_test = len(test_timestamps)

    # Extract Year-Month
    ts_series = pd.Series(test_timestamps)
    year_months = ts_series.dt.to_period('M').astype(str).values

    # Select the 24 balanced full calendar months (2017-01 to 2018-12)
    selected_months = [
        f"{year}-{month:02d}"
        for year in [2017, 2018]
        for month in range(1, 13)
    ]
    print(f"Evaluation Blocks: N = {len(selected_months)} balanced calendar months (2017-01 through 2018-12)\n", flush=True)

    # Core competitive deployable models
    model_files = {
        "Persistence (1-Hr)": "results/raw/step_predictions_persistence_1-hr_momentum.csv",
        "Streaming kNN": "results/raw/step_predictions_streaming_knn_k=5_w=100.csv",
        "Online SGD": "results/raw/step_predictions_online_sgd_logistic_regression.csv",
        "Always-HT": "results/raw/step_predictions_always-ht.csv",
        "Adaptive HT (HAT)": "results/raw/step_predictions_adaptive_hoeffding_tree_hat.csv",
        "CGT-v1 (S1)": "results/raw/step_predictions_cgt_proposed_s1.csv",
        "CGT-v2": "results/raw/step_predictions_cgt-v2_adaptive_dual-gate.csv",
        "CGT-v3 (Learned)": "results/raw/step_predictions_cgt-v3_learned_meta-gate.csv",
        "A2: Cost UCB": "results/raw/step_predictions_a2_cost-aware_ucb.csv",
        "OMS-MAB": "results/raw/step_predictions_oms-mab_wilson_et_al._2024.csv",
        "A4: Router (Fixed)": "results/raw/step_predictions_a4_learned_router_calibrated.csv",
        "SRP (3 trees)": "results/raw/step_predictions_streaming_random_patches_srp_3_trees.csv",
        "LevBag (3 trees)": "results/raw/step_predictions_leveraging_bagging_3_trees.csv",
        "A3: Optimized ARF": "results/raw/step_predictions_a3_optimized_arf.csv",
        "Always-ARF (10 trees)": "results/raw/step_predictions_always-arf.csv"
    }

    # Load predictions
    preds_dict = {}
    for name, fpath in model_files.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing raw prediction file: {fpath}")
        df = pd.read_csv(fpath)
        if len(df) != n_test:
            raise ValueError(f"Length mismatch in {fpath}: {len(df)} vs {n_test}")
        preds_dict[name] = {
            'y_true': df['y_true'].values,
            'y_pred': df['y_pred'].values
        }

    # 2. Build Monthly Accuracy Matrix (N x k)
    matrix_rows = []
    for ym in selected_months:
        mask = (year_months == ym)
        row = {'month': ym, 'n_instances': int(np.sum(mask))}
        for name in model_files.keys():
            y_t = preds_dict[name]['y_true'][mask]
            y_p = preds_dict[name]['y_pred'][mask]
            acc = float(np.mean(y_t == y_p) * 100.0)
            row[name] = acc
        matrix_rows.append(row)

    df_monthly = pd.DataFrame(matrix_rows)
    os.makedirs("results/tables", exist_ok=True)
    df_monthly.to_csv("results/tables/day8_monthly_accuracy_matrix.csv", index=False)
    print("Saved monthly accuracy matrix to: results/tables/day8_monthly_accuracy_matrix.csv")

    # 3. Compute Ranks (1 = best, k = worst)
    model_names = list(model_files.keys())
    k = len(model_names)
    N = len(selected_months)

    acc_matrix = df_monthly[model_names].values # shape (N, k)
    # scipy.stats.rankdata ranks ascending, so rank -acc to give rank 1 to highest accuracy
    rank_matrix = np.zeros_like(acc_matrix)
    for i in range(N):
        rank_matrix[i, :] = stats.rankdata(-acc_matrix[i, :], method='average')

    avg_ranks = np.mean(rank_matrix, axis=0) # shape (k,)

    df_ranks = pd.DataFrame({
        'model': model_names,
        'avg_rank': avg_ranks
    }).sort_values('avg_rank').reset_index(drop=True)

    print("\n--- Average Ranks across 24 Monthly Blocks (1 = Best, Lower is Better) ---")
    print(df_ranks.to_string(index=False))

    # 4. Friedman Test
    # Chi-square Friedman statistic
    sum_r2 = np.sum(avg_ranks ** 2)
    chi2_f = (12.0 * N / (k * (k + 1.0))) * (sum_r2 - (k * (k + 1.0) ** 2) / 4.0)
    p_chi2 = 1.0 - stats.chi2.cdf(chi2_f, df=k - 1)

    # Iman-Davenport adjusted F-statistic
    f_f = ((N - 1.0) * chi2_f) / (N * (k - 1.0) - chi2_f)
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    p_f = 1.0 - stats.f.cdf(f_f, df1, df2)

    print(f"\nFriedman Chi2: {chi2_f:.4f} (df={k-1}, p={p_chi2:.4e})")
    print(f"Iman-Davenport F: {f_f:.4f} (df1={df1}, df2={df2}, p={p_f:.4e})")

    # 5. Nemenyi Post-Hoc Test (alpha = 0.05)
    # Critical value from Studentized range distribution
    # q_alpha = studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)
    q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
    cd = q_alpha * np.sqrt((k * (k + 1.0)) / (6.0 * N))
    print(f"\nNemenyi Critical Difference (CD) at alpha=0.05: {cd:.4f} (q_alpha={q_alpha:.4f}, N={N}, k={k})")

    # Pairwise rank differences vs CGT-v3
    cgt_v3_rank = df_ranks.loc[df_ranks['model'] == 'CGT-v3 (Learned)', 'avg_rank'].values[0]
    pairwise_rows = []
    for idx, row in df_ranks.iterrows():
        m = row['model']
        r = row['avg_rank']
        diff = abs(r - cgt_v3_rank)
        sig = diff > cd
        pairwise_rows.append({
            'model': m,
            'avg_rank': round(r, 3),
            'diff_vs_cgt_v3': round(diff, 3),
            'critical_difference': round(cd, 3),
            'statistically_significant': sig
        })

    df_pairwise = pd.DataFrame(pairwise_rows)
    df_pairwise.to_csv("results/tables/day8_friedman_nemenyi_results.csv", index=False)
    print("\nPairwise Rank Differences vs CGT-v3:")
    print(df_pairwise.to_string(index=False))

    # 6. Plot Critical Difference Diagram
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    
    # Sort models by average rank
    sorted_models = df_ranks['model'].values
    sorted_ranks = df_ranks['avg_rank'].values

    y_pos = np.arange(k)
    colors = []
    for m in sorted_models:
        if "CGT-v3" in m:
            colors.append('#0284C7') # Sky blue for CGT-v3
        elif "CGT" in m:
            colors.append('#38BDF8')
        elif "ARF" in m or "SRP" in m or "LevBag" in m:
            colors.append('#10B981') # Emerald for ensembles
        elif "HT" in m:
            colors.append('#F59E0B') # Amber for trees
        elif "MAB" in m or "Router" in m or "UCB" in m:
            colors.append('#8B5CF6') # Purple for bandits/routers
        else:
            colors.append('#94A3B8') # Gray for naive baselines

    bars = ax.barh(y_pos, sorted_ranks, height=0.65, color=colors, edgecolor='none', alpha=0.85)

    # Annotate rank values
    for i, (bar, r) in enumerate(zip(bars, sorted_ranks)):
        ax.text(r + 0.15, bar.get_y() + bar.get_height()/2.0, f"{r:.2f}",
                va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1E293B')

    # Add CD reference bracket at top
    cd_start = 1.0
    cd_end = 1.0 + cd
    y_cd = k - 0.2
    ax.plot([cd_start, cd_end], [y_cd, y_cd], color='#DC2626', lw=2.5)
    ax.plot([cd_start, cd_start], [y_cd - 0.25, y_cd + 0.25], color='#DC2626', lw=2.0)
    ax.plot([cd_end, cd_end], [y_cd - 0.25, y_cd + 0.25], color='#DC2626', lw=2.0)
    ax.text((cd_start + cd_end)/2.0, y_cd + 0.35, f"Critical Difference (CD = {cd:.2f}, α = 0.05)",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#DC2626')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=10.5, fontweight='semibold')
    ax.invert_yaxis()  # Best rank at top
    ax.set_xlabel("Average Rank across N = 24 Monthly Blocks (Lower Rank = Higher Accuracy)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xlim(0, k + 1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    title_str = f"Multi-Model Critical Difference (CD) Ranking on 24 Monthly Test Blocks\n" \
                f"Iman-Davenport F = {f_f:.2f} (p = {p_f:.2e}) | N = 24 Months, k = {k} Models"
    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=18)

    # Highlight CGT-v3 rank with vertical dashed line
    ax.axvline(cgt_v3_rank, color='#0284C7', linestyle=':', lw=1.5, alpha=0.7)

    plt.tight_layout()
    fig_path = "results/figures/fig_day8_critical_difference.png"
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved Critical Difference diagram to: {fig_path}")

if __name__ == "__main__":
    compute_friedman_nemenyi()

