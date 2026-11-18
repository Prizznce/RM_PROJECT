"""
Generate publication-quality Day 6 Pareto Frontier figure comparing all 18 evaluated
models on the 24,528 frozen test stream instances.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_day6_pareto():
    csv_path = "results/tables/day6_clean_test_benchmark.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    family_styles = {
        "Naive Baseline": {"color": "#8c564b", "marker": "X", "s": 90},
        "Online Linear": {"color": "#e377c2", "marker": "s", "s": 90},
        "Instance-Based": {"color": "#7f7f7f", "marker": "v", "s": 90},
        "Single Tree": {"color": "#1f77b4", "marker": "o", "s": 100},
        "Ensemble": {"color": "#ff7f0e", "marker": "^", "s": 110},
        "Ensemble Reference": {"color": "#d62728", "marker": "*", "s": 180},
        "Switching Policy": {"color": "#9467bd", "marker": "D", "s": 90},
        "Learned Router": {"color": "#bcbd22", "marker": "P", "s": 110},
        "Bandit Baseline (OMS-MAB)": {"color": "#17becf", "marker": "h", "s": 130},
        "CGT Baseline (v1)": {"color": "#2ca02c", "marker": "o", "s": 110},
        "CGT Dual-Gate (v2)": {"color": "#2ca02c", "marker": "p", "s": 130},
        "CGT Learned-Gate (v3)": {"color": "#006400", "marker": "H", "s": 170},
        "Omniscient Ceiling": {"color": "#000000", "marker": "d", "s": 160},
    }

    plt.figure(figsize=(12, 7))

    plotted_families = set()
    for _, row in df.iterrows():
        fam = row["model_family"]
        name = row["model_name"]
        acc = row["accuracy_pct"]
        cpu = row["mean_cpu_us_per_instance"]
        style = family_styles.get(fam, {"color": "black", "marker": "o", "s": 80})

        label = fam if fam not in plotted_families else None
        plotted_families.add(fam)

        plt.scatter(cpu, acc, color=style["color"], marker=style["marker"], s=style["s"],
                    label=label, edgecolors="black", linewidths=1.0, zorder=5)

        # Custom labeling to prevent overlap
        offset_x = 1.08
        offset_y = 0.0
        ha = "left"

        if "Majority" in name:
            label_text = "Majority (55.5%)"
            offset_y = -0.5
        elif "Persistence" in name:
            label_text = "Persistence 1-Hr (81.0%)"
            offset_y = 0.4
        elif "Online SGD" in name:
            label_text = "Online SGD (84.9%)"
            offset_y = -0.6
        elif "kNN" in name:
            label_text = "kNN (78.0%)"
        elif name == "Always-HT":
            label_text = "Always-HT (85.1%)"
            offset_y = -0.6
        elif "HAT" in name:
            label_text = "HAT (85.1%)"
            offset_y = 0.3
        elif "Leveraging Bagging" in name:
            label_text = "LevBag (89.3%)"
            offset_y = 0.3
        elif "SRP" in name:
            label_text = "SRP (88.6%)"
            offset_y = -0.6
        elif "A1" in name:
            label_text = "A1: Thresh (85.6%)"
            offset_y = 0.4
        elif "A2" in name:
            label_text = "A2: UCB (86.4%)"
            offset_y = -0.6
        elif "A3" in name:
            label_text = "A3: Opt-ARF (89.3%)"
            offset_y = 0.3
        elif name == "Always-ARF":
            label_text = "Always-ARF (89.6%)"
            offset_y = 0.4
        elif "A4" in name:
            label_text = "A4: Learned (88.7%)"
            offset_y = -0.7
        elif "OMS-MAB" in name:
            label_text = "OMS-MAB (86.9%)"
            offset_y = 0.4
        elif "v1" in name:
            label_text = "CGT-v1 (85.1%)"
            offset_y = -0.7
        elif "v2" in name:
            label_text = "CGT-v2 (85.5%)"
            offset_y = -0.6
        elif "v3" in name:
            label_text = "CGT-v3 [Best Gate] (85.8%)"
            offset_y = 0.5
        elif "Oracle" in name:
            label_text = "Oracle Ceiling (93.1%)"
            offset_y = 0.3
        else:
            label_text = name

        plt.annotate(
            label_text,
            (cpu, acc),
            xytext=(cpu * offset_x, acc + offset_y),
            fontsize=8.5,
            fontweight="bold" if "CGT" in name or "Oracle" in name or "ARF" in name else "normal",
            ha=ha,
            va="center",
            zorder=6
        )

    plt.xscale("log")
    plt.xlabel("Mean Execution Latency per Instance (μs, Log Scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Prequential Accuracy on Frozen Test Stream (%)", fontsize=11, fontweight="bold")
    plt.title("Empirical Resource-Accuracy Pareto Frontier on 24,528 Test Instances (Day 6 Clean Benchmark)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.ylim(52, 95)
    plt.xlim(1.5, 8000)

    # Highlight CGT Evolution path
    cgt_pts = df[df["model_name"].str.contains("CGT") | (df["model_name"] == "Always-HT")].sort_values("accuracy_pct")
    plt.plot(cgt_pts["mean_cpu_us_per_instance"], cgt_pts["accuracy_pct"],
             color="#2ca02c", linestyle=":", linewidth=2, alpha=0.7, label="CGT Controller Evolution")

    plt.legend(loc="lower right", frameon=True, fontsize=8.5, ncol=2)
    plt.tight_layout()

    out_fig = "results/figures/fig_day6_clean_pareto_frontier.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated Day 6 Clean Pareto Figure: {out_fig}", flush=True)

if __name__ == "__main__":
    plot_day6_pareto()

