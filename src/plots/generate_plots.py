import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

def generate_all_plots():
    os.makedirs("results/figures", exist_ok=True)
    summary_df = pd.read_csv("results/tables/benchmark_summary.csv")
    
    # 1. Figure 1: Prequential Sliding Accuracy over Time
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    models_to_plot = [
        ("Always-ARF", "results/raw/steps_always-arf.csv", "#1f77b4", "-"),
        ("Always-HT", "results/raw/steps_always-ht.csv", "#7f7f7f", "--"),
        ("A2: Cost-aware UCB", "results/raw/steps_a2_cost-aware_ucb.csv", "#ff7f0e", ":"),
        ("CGT (Proposed, S1)", "results/raw/steps_cgt_proposed_s1.csv", "#2ca02c", "-"),
        ("CGT-Plus (Optimized)", "results/raw/steps_cgt-plus_optimized_hybrid.csv", "#9467bd", "-"),
        ("CGT Ablation", "results/raw/steps_cgt_ablation_fixed_schedule.csv", "#d62728", "-.")
    ]
    
    for label, path, color, ls in models_to_plot:
        if os.path.exists(path):
            df = pd.read_csv(path)
            ax.plot(df['step'], df['sliding_acc'], label=label, color=color, linestyle=ls, linewidth=1.8, alpha=0.9)
            
    ax.set_title("Prequential Sliding Window Accuracy Across Test Stream (Window = 500)")
    ax.set_xlabel("Stream Instance (t)")
    ax.set_ylabel("Sliding Accuracy (%)")
    ax.set_ylim(65, 95)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig1_path = "results/figures/fig1_prequential_accuracy.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved {fig1_path}")

    # 2. Figure 2: Cumulative CPU Execution Time
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    bar_df = summary_df.sort_values('total_cpu_sec', ascending=True)
    colors = ['#9467bd' if 'CGT-Plus' in name else '#2ca02c' if 'CGT' in name and 'S1' in name else '#1f77b4' if 'ARF' in name else '#aec7e8' for name in bar_df['model_name']]
    
    y_pos = np.arange(len(bar_df))
    ax.barh(y_pos, bar_df['total_cpu_sec'], color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_df['model_name'])
    ax.set_xlabel("Total Test Stream CPU Execution Time (Seconds)")
    ax.set_title("Computational Cost Across Stream Models (Lower is Better)")
    
    # Annotate CPU savings
    for i, (_, row) in enumerate(bar_df.iterrows()):
        saving = row['cpu_saving_pct']
        ax.text(row['total_cpu_sec'] + 0.3, i, f"{row['total_cpu_sec']:.2f}s (-{saving}%)", va='center', fontsize=8.5)
        
    ax.set_xlim(0, max(bar_df['total_cpu_sec']) * 1.3)
    plt.tight_layout()
    fig2_path = "results/figures/fig2_cumulative_cpu_cost.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved {fig2_path}")

    # 3. Figure 3: Pareto Frontier (Accuracy vs Per-instance CPU Latency)
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    
    for _, row in summary_df.iterrows():
        name = row['model_name']
        acc = row['accuracy_pct']
        cpu = row['mean_cpu_us_per_instance']
        
        is_plus = 'CGT-Plus' in name
        is_cgt = 'CGT' in name and 'S1' in name and not is_plus
        is_arf = name == 'Always-ARF'
        is_ht = name == 'Always-HT'
        
        marker = 'P' if is_plus else ('*' if is_cgt else ('s' if is_arf else ('^' if is_ht else 'o')))
        size = 200 if (is_plus or is_cgt) else 110
        color = '#9467bd' if is_plus else ('#2ca02c' if is_cgt else ('#1f77b4' if is_arf else ('#7f7f7f' if is_ht else '#ff7f0e')))
        
        ax.scatter(cpu, acc, s=size, color=color, marker=marker, edgecolors='black', zorder=5)
        ax.annotate(name, (cpu, acc), textcoords="offset points", xytext=(8, -4), fontsize=8.5, weight='bold' if (is_plus or is_cgt) else 'normal')

    ax.set_title("Pareto Efficiency: Prequential Accuracy vs Computational Cost")
    ax.set_xlabel("Mean CPU Latency (μs / instance) [Log Scale]")
    ax.set_ylabel("Overall Prequential Accuracy (%)")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = "results/figures/fig3_pareto_accuracy_cost.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved {fig3_path}")

    # 4. Figure 4: CGT Gating State & ADWIN Interrupts over Time
    cgt_step_path = "results/raw/steps_cgt-plus_optimized_hybrid.csv" if os.path.exists("results/raw/steps_cgt-plus_optimized_hybrid.csv") else "results/raw/steps_cgt_proposed_s1.csv"
    if os.path.exists(cgt_step_path):
        df_cgt = pd.read_csv(cgt_step_path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300, gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(df_cgt['step'], df_cgt['sliding_acc'], color='#2ca02c', label="CGT Sliding Accuracy")
        ax1.set_ylabel("Sliding Acc (%)")
        ax1.set_title("CGT Dynamic Behavior: Sliding Accuracy and ARF Gate State (g_t)")
        ax1.legend(loc="lower right")
        
        ax2.step(df_cgt['step'], df_cgt['gate_open'], color='#d62728', where='post', label="Gate State (1=ARF Active, 0=ARF Dormant)")
        ax2.set_ylabel("Gate State g_t")
        ax2.set_xlabel("Stream Instance (t)")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["0 (Dormant)", "1 (Active)"])
        ax2.legend(loc="upper right")
        
        plt.tight_layout()
        fig4_path = "results/figures/fig4_cgt_gating_behavior.png"
        plt.savefig(fig4_path)
        plt.close()
        print(f"Saved {fig4_path}")

if __name__ == '__main__':
    generate_all_plots()
