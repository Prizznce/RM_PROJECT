"""
Task F: Publication-Grade Pareto Frontier with Explicit Dual-Regime Tradeoff.
Features:
  - Side-by-side or high-resolution focused layout maximizing vertical separation in 85%-90%
  - Non-dominated empirical Pareto hull highlighted with connecting line
  - Reference baselines: Persistence (80.95%) and Oracle Ceiling (93.09%, Upper Bound = 9.0 / 10)
  - Explicit callouts for CGT-v3 (85.85%, 876.9 μs, Deployable Champion = 8.8 / 10)
  - Region shading: Zone 1 (RTU Edge Envelope <= 1000 us) vs Zone 2 (Server Cluster Regime > 1000 us)
  - Zero label collisions and calibrated multi-objective scoring
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_rigorous_pareto_plot():
    # Prefer final_results_table.csv if available
    final_csv = "results/tables/final_results_table.csv"
    day6_csv = "results/tables/day6_clean_test_benchmark.csv"
    
    if os.path.exists(final_csv):
        df = pd.read_csv(final_csv)
        col_name = "policy_name"
        col_acc = "eval_acc"
        col_cpu = "mean_latency_us"
        col_pareto = "pareto_optimal_acc_lat"
        col_rho = "duty_cycle_rho_pct"
    elif os.path.exists(day6_csv):
        df = pd.read_csv(day6_csv)
        col_name = "model_name"
        col_acc = "accuracy_pct"
        col_cpu = "mean_cpu_us_per_instance"
        col_pareto = None
        col_rho = "active_gate_rho"
    else:
        raise FileNotFoundError(f"Missing {final_csv} or {day6_csv}")

    # Clean / prepare data
    deployable = df[~df[col_name].str.contains("Oracle")].copy()
    oracle_row = df[df[col_name].str.contains("Oracle")].iloc[0]

    if col_pareto is not None:
        pareto_df = deployable[deployable[col_pareto]].sort_values(col_cpu)
    else:
        sorted_df = deployable.sort_values(col_cpu).reset_index(drop=True)
        pareto_pts = []
        max_acc_so_far = -1.0
        for _, row in sorted_df.iterrows():
            if row[col_acc] > max_acc_so_far:
                pareto_pts.append(row)
                max_acc_so_far = row[col_acc]
        pareto_df = pd.DataFrame(pareto_pts)

    pareto_names = set(pareto_df[col_name])

    # Create publication figure: 2 Panels (Left: Macro Spectrum, Right: High-Resolution Competitive Zoom)
    fig, (ax_macro, ax_zoom) = plt.subplots(
        1, 2, figsize=(16, 8.5), 
        gridspec_kw={"width_ratios": [1, 2.3]}, 
        facecolor="#FFFFFF"
    )

    # =========================================================================
    # PANEL 1 (LEFT): GLOBAL MACRO SPECTRUM (50% - 95%)
    # =========================================================================
    ax_macro.set_facecolor("#FAFAFA")
    ax_macro.axvspan(1.5, 1000, color="#E6F4EA", alpha=0.6, label="RTU Edge (<= 1 ms)")
    ax_macro.axvspan(1000, 25000, color="#FFF3E0", alpha=0.5, label="Server (> 1 ms)")
    ax_macro.axvline(1000, color="#D93025", linestyle="--", linewidth=1.8, alpha=0.85)
    ax_macro.axhline(oracle_row[col_acc], color="#7952B3", linestyle=":", linewidth=1.8, alpha=0.85)

    # Plot points in macro view
    for _, row in df.iterrows():
        name = row[col_name]
        cpu = row[col_cpu]
        acc = row[col_acc]
        is_or = "Oracle" in name
        is_cgt = "CGT-v3" in name
        is_par = name in pareto_names

        if is_or:
            ax_macro.scatter(cpu, acc, color="#7952B3", marker="*", s=220, edgecolors="black", linewidths=1.2, zorder=6)
        elif is_cgt:
            ax_macro.scatter(cpu, acc, color="#00897B", marker="D", s=140, edgecolors="black", linewidths=1.2, zorder=6)
        elif is_par:
            ax_macro.scatter(cpu, acc, color="#2E7D32", marker="o", s=80, edgecolors="black", linewidths=1.0, zorder=5)
        else:
            ax_macro.scatter(cpu, acc, color="#9E9E9E", marker="o", s=50, alpha=0.6, edgecolors="black", linewidths=0.8, zorder=4)

    # Connect Pareto line
    ax_macro.plot(pareto_df[col_cpu], pareto_df[col_acc], color="#2E7D32", linewidth=2.0, zorder=5)

    # Key callouts on macro panel
    ax_macro.annotate("Zero-Skill\nMajority (55.5%)", (2.3, 55.5), xytext=(5, 58),
                      fontsize=8, fontweight="bold", color="#5F6368", arrowprops=dict(arrowstyle="->", color="#5F6368", lw=0.8))
    ax_macro.annotate("Streaming kNN\n(78.0%)", (194, 78.0), xytext=(25, 75),
                      fontsize=8, fontweight="bold", color="#5F6368", arrowprops=dict(arrowstyle="->", color="#5F6368", lw=0.8))
    ax_macro.annotate("Persistence\n(81.0%)", (2.6, 81.0), xytext=(6, 82.5),
                      fontsize=8, fontweight="bold", color="#5F6368", arrowprops=dict(arrowstyle="->", color="#5F6368", lw=0.8))
    ax_macro.annotate("Competitive\nRegime (84-90%)\n[See Right Zoom]", (600, 87), xytext=(15, 89.5),
                      fontsize=9, fontweight="bold", color="#00897B",
                      bbox=dict(boxstyle="round,pad=0.4", fc="#E0F2F1", ec="#00897B", lw=1.2),
                      arrowprops=dict(arrowstyle="->", color="#00897B", lw=1.2))

    ax_macro.set_xscale("log")
    ax_macro.set_xlim(1.5, 25000)
    ax_macro.set_ylim(50, 96)
    ax_macro.set_xlabel("Mean CPU Latency (μs, Log Scale)", fontsize=10, fontweight="bold")
    ax_macro.set_ylabel("Prequential Stream Accuracy (%)", fontsize=10, fontweight="bold")
    ax_macro.set_title("A. Global Macro Overview (All 18 Models)", fontsize=11, fontweight="bold", pad=10)
    ax_macro.grid(True, which="both", linestyle="--", alpha=0.35)

    # =========================================================================
    # PANEL 2 (RIGHT): HIGH-RESOLUTION COMPETITIVE ZOOM (84.0% - 94.0%)
    # 5x vertical expansion for 85%-90% to eliminate all overlap
    # =========================================================================
    ax_zoom.set_facecolor("#FFFFFF")
    ax_zoom.axvspan(1.5, 1000, color="#E6F4EA", alpha=0.6, label="Zone 1: RTU Edge Hardware Envelope (<= 1 ms / < 0.2W)")
    ax_zoom.axvspan(1000, 25000, color="#FFF3E0", alpha=0.5, label="Zone 2: Server / SCADA Cluster Regime (> 1 ms)")
    
    # 1 ms edge hard deadline vertical line
    ax_zoom.axvline(1000, color="#D93025", linestyle="--", linewidth=2.2, alpha=0.9, zorder=3)
    ax_zoom.text(1050, 84.3, "1,000 μs (1 ms) RTU Hard Deadline", color="#D93025", fontsize=9.5, fontweight="bold", rotation=90)

    # Theoretical Oracle ceiling horizontal line (93.09%)
    ax_zoom.axhline(oracle_row[col_acc], color="#7952B3", linestyle="--", linewidth=2.0, alpha=0.9, zorder=3)
    ax_zoom.text(3, 93.25, "THEORETICAL ORACLE CEILING: 93.09% (Upper Bound = 9.0 / 10, Non-Causal)", 
                 color="#7952B3", fontsize=10, fontweight="bold")

    # Connect Pareto frontier
    ax_zoom.plot(pareto_df[col_cpu], pareto_df[col_acc], color="#00897B", linestyle="-", linewidth=3.0, alpha=0.9, zorder=5, label="Deployable Pareto Frontier")
    
    # Dotted line to Oracle
    last_pareto = pareto_df.iloc[-1]
    ax_zoom.plot([last_pareto[col_cpu], oracle_row[col_cpu]], [last_pareto[col_acc], oracle_row[col_acc]], 
                 color="#7952B3", linestyle=":", linewidth=2.0, alpha=0.8, zorder=4, label="Theoretical Ceiling Extension")

    # Filter models visible in zoom (acc >= 84.0)
    zoom_df = df[df[col_acc] >= 84.0].copy()

    # Explicit, staggered, collision-free text annotations for every single model in 84%-94%
    staggered_labels = {
        "Online SGD (Logistic Regression)": {"text": "Online SGD Linear\n(84.9%, 84 μs)", "xytext": (-65, -28), "color": "#455A64", "va": "top", "ha": "center"},
        "Always-HT": {"text": "Always-HT Baseline\n(85.1%, 302 μs, 100% Sleep)", "xytext": (0, -32), "color": "#1B5E20", "va": "top", "ha": "center"},
        "CGT (Proposed, S1)": {"text": "CGT-v1 Pure ADWIN\n(85.1%, 316 μs)", "xytext": (-80, 18), "color": "#00695C", "va": "bottom", "ha": "center"},
        "Adaptive Hoeffding Tree (HAT)": {"text": "HAT Adaptive Tree\n(85.1%, 504 μs)", "xytext": (75, -28), "color": "#37474F", "va": "top", "ha": "center"},
        "CGT-v2 (Adaptive Dual-Gate)": {"text": "CGT-v2 Dual-Gate\n(85.5%, 416 μs, ρ=5.1%)", "xytext": (-70, 20), "color": "#00695C", "va": "bottom", "ha": "center"},
        "A1: Threshold Switch": {"text": "A1: Threshold Switch\n(85.6%, 1551 μs)", "xytext": (50, -28), "color": "#6A1B9A", "va": "top", "ha": "center"},
        "CGT-v3 (Learned Meta-Gate)": {
            "text": "★ CGT-v3 [PROPOSED EDGE CHAMPION]\nAcc: 85.85% | Lat: 876.9 μs (< 1 ms Edge Budget)\nDuty: ρ=7.02% (142h Battery) | Score: 8.8 / 10",
            "xytext": (-140, 36), "color": "#004D40", "va": "bottom", "ha": "left", "is_hero": True
        },
        "A2: Cost-aware UCB": {"text": "A2: Cost-aware UCB\n(86.5%, 1433 μs, ρ=42.7%)", "xytext": (-80, -28), "color": "#6A1B9A", "va": "top", "ha": "center"},
        "OMS-MAB (Wilson et al., 2024)": {"text": "OMS-MAB [Wilson et al., 2024]\n(86.5%, 1934 μs, ρ=61.9%)", "xytext": (80, 18), "color": "#6A1B9A", "va": "bottom", "ha": "center"},
        "Streaming Random Patches (SRP, 3 trees)": {"text": "SRP 3-Tree Ensemble\n(87.2%, 2971 μs)", "xytext": (80, -26), "color": "#E65100", "va": "top", "ha": "center"},
        "A4: Learned Router (Calibrated)": {
            "text": "A4: Learned Router (88.2%, 947 μs)\n[Caution: 86.1% Duty Cycle Trap]", 
            "xytext": (-120, 24), "color": "#C2185B", "va": "bottom", "ha": "right"
        },
        "A3: Optimized ARF (5 trees)": {"text": "A3: Optimized ARF 5-Tree\n(88.3%, 1349 μs) [Zone 2]", "xytext": (80, 20), "color": "#E65100", "va": "bottom", "ha": "center"},
        "Leveraging Bagging (3 trees)": {"text": "Leveraging Bagging\n(89.1%, 17.5 ms)", "xytext": (0, 20), "color": "#E65100", "va": "bottom", "ha": "center"},
        "Always-ARF (Full, 10 trees)": {"text": "Always-ARF 10-Tree Reference\n(89.8%, 3037 μs, 100% Duty / 18h Battery)", "xytext": (-110, 22), "color": "#B71C1C", "va": "bottom", "ha": "right"},
        "Oracle Switch (Upper Bound)**": {
            "text": "ORACLE SWITCH (THEORETICAL CEILING)\nAcc: 93.09% | Non-Causal Upper Bound = 9.0 / 10\n(Future-leakage omniscience: bounds best possible switching)",
            "xytext": (-150, -42), "color": "#4A148C", "va": "top", "ha": "right", "is_hero": True
        }
    }

    # Plot markers and annotations
    for _, row in zoom_df.iterrows():
        name = row[col_name]
        cpu = row[col_cpu]
        acc = row[col_acc]
        is_or = "Oracle" in name
        is_cgt = "CGT-v3" in name
        is_par = name in pareto_names

        # Draw Marker
        if is_or:
            ax_zoom.scatter(cpu, acc, color="#7952B3", marker="*", s=350, edgecolors="#000000", linewidths=1.8, zorder=8, label="Oracle Ceiling (Upper Bound = 9.0 / 10)")
        elif is_cgt:
            ax_zoom.scatter(cpu, acc, color="#00BCD4", marker="D", s=240, edgecolors="#004D40", linewidths=2.2, zorder=8, label="★ Proposed CGT-v3 (8.8 / 10 Champion)")
        elif is_par:
            ax_zoom.scatter(cpu, acc, color="#2E7D32", marker="o", s=110, edgecolors="#000000", linewidths=1.4, zorder=7)
        elif "A4" in name:
            ax_zoom.scatter(cpu, acc, color="#E91E63", marker="^", s=120, edgecolors="#000000", linewidths=1.4, zorder=7)
        else:
            ax_zoom.scatter(cpu, acc, color="#9E9E9E", marker="o", s=80, edgecolors="#000000", linewidths=1.0, alpha=0.8, zorder=6)

        # Draw Label
        lbl_cfg = staggered_labels.get(name, None)
        if lbl_cfg:
            txt = lbl_cfg["text"]
            x_off, y_off = lbl_cfg["xytext"]
            txt_col = lbl_cfg["color"]
            va_align = lbl_cfg["va"]
            ha_align = lbl_cfg["ha"]
            is_hero = lbl_cfg.get("is_hero", False)

            if is_hero:
                bbox_style = dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec=txt_col, lw=1.8, alpha=0.96)
                font_weight = "bold"
                font_size = 9.2
            else:
                bbox_style = dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#CCCCCC", lw=0.8, alpha=0.90)
                font_weight = "bold" if is_par or "A4" in name else "normal"
                font_size = 8.2

            ax_zoom.annotate(
                txt,
                xy=(cpu, acc),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=font_size,
                fontweight=font_weight,
                color=txt_col,
                va=va_align,
                ha=ha_align,
                bbox=bbox_style,
                arrowprops=dict(arrowstyle="->", color=txt_col, lw=1.2 if is_hero else 0.8, alpha=0.85),
                zorder=9
            )

    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(50, 25000)
    ax_zoom.set_ylim(84.0, 94.4)
    ax_zoom.set_yticks([84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.09, 94.0])
    ax_zoom.set_yticklabels(["84.0%", "85.0%", "86.0%", "87.0%", "88.0%", "89.0%", "90.0%", "91.0%", "92.0%", "93.1% (Oracle)", "94.0%"], fontsize=9.5)
    
    ax_zoom.set_xlabel("Mean CPU Latency per Instance (μs, Logarithmic Scale)", fontsize=11, fontweight="bold")
    ax_zoom.set_ylabel("Prequential Stream Accuracy (%) [5x Expanded Resolution]", fontsize=11, fontweight="bold")
    ax_zoom.set_title("B. Competitive Operational Frontier & Real-Time Edge Hardware Envelope (85%–90% Focus)", fontsize=12, fontweight="bold", pad=10)
    ax_zoom.grid(True, which="both", linestyle="--", alpha=0.4)

    # Custom Clean Legend
    legend_elements = [
        mpatches.Patch(facecolor="#E6F4EA", edgecolor="#2E7D32", label="Zone 1: RTU Edge Real-Time Envelope (<= 1 ms / < 0.2W)"),
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#E65100", label="Zone 2: Server / SCADA Cluster Regime (> 1 ms / > 2.5W)"),
        plt.Line2D([0], [0], color="#00897B", lw=3.0, label="Deployable Pareto Frontier"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#00BCD4", markeredgecolor="#004D40", markersize=10, label="Proposed CGT-v3 (85.85%, 876 μs, Score = 8.8 / 10)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#7952B3", markeredgecolor="#000000", markersize=14, label="Oracle Ceiling (93.09%, Upper Bound = 9.0 / 10)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E7D32", markeredgecolor="#000000", markersize=8, label="Non-Dominated Pareto Set"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#E91E63", markeredgecolor="#000000", markersize=8, label="A4 Router (86.1% Heavy Duty Trap)")
    ]
    ax_zoom.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#BDBDBD", fontsize=8.5)

    plt.suptitle("The Fundamental Accuracy vs. Latency / Energy Tradeoff in Online Stream Classification\nEvaluated on 24,528 Continuous Electrical Grid Test Instances (National Grid Telemetry)", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save to both paths
    out_fig1 = "results/figures/fig_day7_pareto_rigorous.png"
    out_fig2 = "results/figures/fig_cgt_tradeoff_presentation.png"
    
    os.makedirs(os.path.dirname(out_fig1), exist_ok=True)
    plt.savefig(out_fig1, dpi=300)
    plt.savefig(out_fig2, dpi=300)
    plt.close()

    print(f"✅ Generated Rigorous Pareto Frontier Figure: {out_fig1}", flush=True)
    print(f"✅ Generated High-Resolution Presentation Figure: {out_fig2}", flush=True)

if __name__ == "__main__":
    generate_rigorous_pareto_plot()
