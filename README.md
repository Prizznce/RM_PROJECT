# Cascaded Compute-Gated Trees (CGT)
### Resource-Efficient Continuous Data Stream Mining Under Concept Drift for Cyber-Physical Edge Infrastructure

[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-40%2F40%20Passing-brightgreen)](file:///d:/SEM5/ResearchMgmt-2/tests)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.17-blue)](file:///d:/SEM5/ResearchMgmt-2/pyproject.toml)
[![IEEE Standard](https://img.shields.io/badge/Standard-IEEE%20C37.118%20PMU%20(50%20Hz)-orange)](file:///d:/SEM5/ResearchMgmt-2/src/physics/grid_stability.py)
[![Reproducibility](https://img.shields.io/badge/Configs-SHA--256%20Locked-purple)](file:///d:/SEM5/ResearchMgmt-2/configs)

---

## 1. Executive Summary & The Edge ML Trilemma

In mission-critical cyber-physical systems—such as national electrical grids, digital substations, and microgrid inverter controllers—machine learning algorithms must continuously process high-frequency sensor streams under non-stationary distribution shifts (concept drift). However, deploying modern streaming machine learning at the edge introduces an uncompromising **three-way physical trilemma**:

```
                                  THE EDGE ML TRILEMMA
                                Predictive Accuracy
                                       /    \
                                      /      \
                             ARF-3   /        \   Always-ARF (10T)
                         (Continuous)          \ (133% CPU Overload)
                                    /   CGT     \
                                   /  (Balanced) \
                                  /                \
        Always-HT / Linear Models ------------------ Micro-Watt Battery Autonomy
      (Sub-100 µs, Low Memory,                    (Dormancy, Long Lifespan,
       Degrades Under Grid Drift)                  Sub-1ms Interrupt Budget)
```

1. **Lightweight Edge Models (Always-HT, Streaming Linear SGD)**:
   Execute rapidly in microseconds ($\approx 13$ to $94\ \mu\text{s}$) with negligible memory ($\le 74\text{ KB}$ SRAM), but suffer from underfitting and representational degradation during sudden cyber-physical shocks.
2. **Continuous Heavy Ensembles (Adaptive Random Forests - 10 Trees)**:
   Accurately track volatile distribution shifts ($89.60\%$ accuracy), but incur severe computational latency ($1,334\ \mu\text{s}$ mean, spiking to $> 100\text{ ms}$ during tree splits). At 1 kHz RTU sampling, this produces a **133.4% CPU overload**, violating non-preemptible interrupt service routine (ISR) deadlines ($\le 1.0\text{ ms}$) and draining emergency backup batteries.
3. **Prior-Art Unconstrained Bandits (OMS-MAB by Wilson et al.)**:
   Because unconstrained multi-armed bandits optimize 0-1 accuracy without hardware cost awareness, they pull the heavy ensemble 63.3% of the time, exceeding thermal and battery budgets.

### The Cascaded Compute-Gated Trees (CGT) Solution
**CGT** is a causal online meta-controller that decouples continuous inference into a hierarchical cascade:
- **Fast Tier (Always-On)**: A lightweight Hoeffding Tree processes incoming telemetry frames in $\approx 94\ \mu\text{s}$.
- **Meta-Gating Tier (Microsecond Causal Router)**: An online meta-classifier continuously evaluates stream uncertainty (margin gap, base model sliding error, diurnal harmonics, and ADWIN cooldown).
- **Heavy Tier (Selectively Escalated)**: If and only if severe uncertainty or regime drift is detected, the gate activates a dormant Adaptive Random Forest (ARF-3) via an $S_1$ warm-start replay buffer ($B = 200$) to provide immediate corrective capacity.
- **Dormancy Profile**: The heavy ensemble remains powered-down or dormant **$\sim 95.4\%$ of the time**, preserving micro-watt battery autonomy and guaranteeing low average CPU load.

---

## 2. The Real-Time Pareto Frontier: Three Non-Dominated Regimes

On the sub-1ms real-time deployable frontier, our comprehensive multi-dataset evaluation reveals **three distinct, non-dominated operational regimes**:

```
Accuracy (%)
  87.5% |                                          [ARF-3 (100% Duty)]
        |                                          (86.19% Acc, 281 µs)
  86.0% |                     [CGT (4.6% Duty)]
        |                     (85.33% Acc, 243 µs)
  85.0% |   [Always-HT (0% Duty)]
        |   (85.07% Acc, 94 µs)
  84.0% |              [Cost-Aware MAB (4.6% Duty)]
        |              (84.30% Acc, 191 µs)
        +-------------------------------------------------------------- Heavy Duty (%)
        0%                     5%                                100%
```

1. **Ultra-Low-Power Regime (Always-HT)**:
   - *Test Accuracy*: $85.07\%$ | *Mean Latency*: $94.26\ \mu\text{s}$ | *Heavy Duty*: $0.00\%$ | *100 Wh Battery*: **1.20 years**.
   - *Operating Niche*: Lowest CPU load and latency; zero gating overhead.
2. **Cost-Adjusted Balanced Regime (CGT - Ours)**:
   - *Test Accuracy*: $85.33\%$ | *Mean Latency*: $243.37\ \mu\text{s}$ | *Heavy Duty*: **$4.59\%$** | *100 Wh Battery*: **0.99 years**.
   - *Operating Niche*: Outperforms Random-Gate (**+0.58 pp**, $p = 2.09 \times 10^{-8}$) and Cost-Aware OMS-MAB (**+1.03 pp**, $p = 1.13 \times 10^{-7}$) at matched duty, while reducing heavy ensemble compute by **~95.4%**.
3. **High-Accuracy Continuous Regime (ARF-3)**:
   - *Test Accuracy*: $86.19\%$ | *Mean Latency*: $281.35\ \mu\text{s}$ | *Heavy Duty*: **$100.00\%$** | *100 Wh Battery*: **0.95 years**.
   - *Operating Niche*: Maximum raw accuracy (+0.86 pp over CGT) for grid nodes with continuous mains power.

---

## 3. Comprehensive Master Benchmark Suite

Evaluated under the **prequential test-then-train protocol** across two massive real-world streams:
- **Stream 1: Primary SCADA Benchmark Stream** (24,528 continuous test hours, 146 independent calendar weeks).
- **Stream 2: Real-World Indian National Power Grid Stream (POSOCO / NLDC)** (16,800 continuous test hours, 100 independent calendar weeks, compiled from 28 monthly SCADA workbooks, Sept 2021 – Dec 2023).

### Master Comparative Results Table

| Model Architecture | Ensemble Size | Heavy Duty ($\rho$) | Benchmark Acc (24k h) | POSOCO Acc (16.8k h) | Mean Latency (host) | Worst-Case Latency | SRAM Footprint | Modeled 50 Hz CPU Duty | 100 Wh Battery Autonomy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Streaming Linear (SGD)** | 1 | 0.00% | 85.062% | **49.060%** *(Underfits)* | **13.92 $\mu$s** | **320.4 $\mu$s** | **~1.2 KB** | **0.070%** | **11,240 h (1.28 yr)** |
| **Always-HT (Single Tree)** | 1 | 0.00% | 85.066% | 84.244% | 94.26 $\mu$s | 2,404.0 $\mu$s | ~74 KB | 0.471% | 10,547 h (1.20 yr) |
| **Random-Gate (20 seeds)** | 1 / 3 | 4.58% | 84.756% | 83.904% | 102.82 $\mu$s | 4,309.3 $\mu$s | ~74 KB | 0.514% | 10,402 h (1.19 yr) |
| **Cost-Aware OMS-MAB** | 1 / 3 | 4.60% | 84.300% | 85.286% | 191.05 $\mu$s | 3,671.7 $\mu$s | ~110 KB | 0.955% | 9,262 h (1.06 yr) |
| **CGT (Learned Meta-Gate)** | 1 / 3 | **4.59%** | **85.331%** | **84.923%** | **243.37 $\mu$s** | **38,002.8 $\mu$s** | **~74 KB** | **1.217%** | **8,690 h (0.99 yr)** |
| **ARF-3 (Fair Protocol)** | 3 | **100.00%** | **86.191%** | **87.018%** | **281.35 $\mu$s** | **2,231.1 $\mu$s** | **~110 KB** | **1.407%** | **8,317 h (0.95 yr)** |
| **Always-ARF (10 Trees Ref)** | 10 | 100.00% | 89.596% | 90.125% | 1,333.95 $\mu$s | 101,442.2 $\mu$s | ~246 KB | 6.670% | 3,798 h (0.43 yr) |

---

## 4. Deep-Dive: Streaming Linear Regression & Multi-Dimensional Trade-Offs

We benchmarked an online streaming logistic regression classifier (`StandardScaler() | LogisticRegression(SGD)`):

### The POSOCO Failure Case (49.06% Accuracy)
On the real Indian Power Grid (POSOCO / NLDC, 16,800 test instances):
- **Class 1 (Stress Events)**: 5,078 hours (30.2%) | **Class 0 (Normal State)**: 11,722 hours (69.8%).
- **Linear Model Predictions**: Over-predicted stress **9,600 times (57.1% of stream)**.
- **Confusion Breakdown**: $\text{TP} = 3,060$, $\text{FP} = 6,540$ (**55.8% False Positive Rate**), $\text{TN} = 5,182$, $\text{FN} = 2,018$.
- **Precision**: $31.88\%$ | **Recall**: $60.26\%$ | **F1-Score**: **$41.70\%$**.

### Why Did Linear Regression Collapse?
1. **Non-Linear Physical Telemetry**: Real AC power flow follows trigonometric power-angle equations ($P \propto \sin(\delta)$), solar generation is non-linear ($P \propto \sin^{1.8}(\omega t)$), wind output is cubic ($P \propto v^3$), and thermal plants follow piecewise ramp-rate limits.
2. **Linear Inseparability**: Power grid stress boundaries are non-linearly intertwined. A single hyperplane ($\mathbf{w}^\top \mathbf{x} + b = 0$) cannot capture these interactions, leading to massive false alarms.
3. **The Architectural Insight of CGT**:
   - While linear models fail as end-to-end classifiers for raw grid state, **they are optimal as microsecond meta-gates**.
   - In CGT-v3, the linear model operates on high-level engineered meta-features (margin gap, sliding error, diurnal harmonics), making routing decisions in **$< 35\ \mu\text{s}$**, while non-linear decision trees handle the physical grid classification.

---

## 5. Statistical Rigor: 146-Week & 100-Week Block Hypothesis Testing

To avoid prequential temporal autocorrelation, predictions are aggregated into non-overlapping calendar weeks ($N = 146$ weeks on Benchmark, $N = 100$ weeks on POSOCO):

### Primary Benchmark Stream (146 Weekly Blocks, Holm-Bonferroni Corrected)
- **CGT vs. Cost-Aware OMS-MAB**: Mean Delta $= \mathbf{+1.031\%}$, $W = 2,007.5$, **Holm $p = \mathbf{1.13 \times 10^{-7}}$** ($p < 0.001$, Significant), Cohen's $d = 0.503$ (Medium Effect).
- **CGT vs. Random-Gate (20 seeds)**: Mean Delta $= \mathbf{+0.575\%}$, $W = 2,376.5$, **Holm $p = \mathbf{2.09 \times 10^{-8}}$** ($p < 0.001$, Significant), Cohen's $d = 0.488$.
- **CGT vs. Always-HT**: Mean Delta $= \mathbf{+0.265\%}$, $W = 1,645.0$, **Holm $p = \mathbf{5.53 \times 10^{-3}}$** ($p < 0.01$, Significant), Cohen's $d = 0.220$.
- **ARF-3 vs. CGT**: Mean Delta $= \mathbf{+0.860\%}$, $W = 3,192.5$, **Holm $p = \mathbf{2.89 \times 10^{-3}}$** ($p < 0.01$, Significant), Cohen's $d = 0.264$.

### Real Indian National Grid Stream (100 Weekly Blocks, Holm Corrected)
- **CGT vs. Random-Gate**: Mean Delta $= \mathbf{+0.679\%}$, **Holm $p = \mathbf{6.65 \times 10^{-3}}$** (Significant), Cohen's $d = 0.321$.
- **CGT vs. Always-HT**: Mean Delta $= \mathbf{+0.679\%}$, **Holm $p = \mathbf{4.99 \times 10^{-3}}$** (Significant), Cohen's $d = 0.321$.
- **ARF-3 vs. CGT**: Mean Delta $= \mathbf{+2.095\%}$, **Holm $p = \mathbf{7.56 \times 10^{-3}}$** (Significant), Cohen's $d = 0.337$.
- **CGT vs. Cost-Aware OMS-MAB**: Delta $= -0.363\%$, $p = 0.750$ (Statistical Tie).

---

## 6. Embedded Hardware Feasibility & Cyber-Physical Grid Simulation

### Multi-Rate Sampling Profiling (IEEE C37.118 PMU Standard)
Modeled for STM32H743ZI (ARM Cortex-M7, 480 MHz, 280 mW active) and Raspberry Pi CM4 (Quad Cortex-A72, 3,000 mW):
- **50 Hz Standard PMU ($T_{\text{sample}} = 20.0\text{ ms}$)**:
  - Always-HT, Random-Gate, Cost-Aware OMS-MAB, and ARF-3 meet the hard interrupt deadline ($t_{\max} \le 20\text{ ms}$).
  - CGT incurs an `OVERRUN` label on worst-case latency due to infrequent $38\text{ ms}$ buffer replays during drift recovery, though its mean latency consumes just **$1.22\%$ CPU duty**.
- **1 kHz High-Frequency RTU ($T_{\text{sample}} = 1.0\text{ ms}$)**:
  - 10-Tree ARF experiences **$133.4\%$ CPU overload** (catastrophic failure).
  - CGT operates at **$24.3\%$ average CPU duty**, leaving headroom for critical IEC 61850-9-2 Sampled Values and GOOSE protection loops.

### Cyber-Physical Swing Equation Simulation
Coupled into a synchronous grid stability simulation under a 4,500 MW generation contingency ($\text{RoCoF} = -0.500\text{ Hz/s}$):
- **Uncontrolled**: Grid trips Stage-1 Under-Frequency Load Shedding (UFLS: 49.50 Hz) in **1.00 s**, plunging to Nadir $= 49.151\text{ Hz}$ (2,500 MW feeder blackout).
- **Fast Edge Streaming (Always-HT, CGT, ARF-3)**: Actuate battery storage in $< 50\text{ ms}$, stabilizing frequency nadir at **$49.499\text{ Hz}$** and preventing blackouts.
- **Deep Sequence Baselines (LSTM / MLP, 12.5 ms latency)**: Miss real-time interrupt cycles, incur a 1.6 s communication queue deferral, plunging nadir to **$49.357\text{ Hz}$** (severe blackout).

---

## 7. Repository Architecture

```
ResearchMgmt-2/
├── app/
│   └── dashboard.py                  # Interactive Streamlit 6-tab control room dashboard
├── configs/
│   ├── frozen.yaml                   # Frozen CGT-v1 configuration (SHA-256 locked)
│   ├── frozen_v2.yaml                # Frozen CGT-v2 dual-gate configuration
│   ├── frozen_v3.yaml                # Frozen CGT-v3 learned-gate configuration
│   ├── fair_comparison_frozen_configs.json # Locked ARF-3 / CGT fair comparison parameters
│   └── cost_aware_oms_mab_frozen_config.json # Locked Cost-Aware OMS-MAB config
├── data/
│   ├── energy_dataset_clean.csv      # Primary 35,064-hour multi-fuel SCADA stream
│   └── india_electricity_real_hourly.csv # 20,448-hour Real POSOCO/NLDC Indian Grid telemetry
├── src/
│   ├── models/                       # Streaming models (HT, ARF, SGD, kNN, LevBag, SRP)
│   ├── controllers/                  # Meta-controllers (CGT-v1, CGT-v2, CGT-v3, OMS-MAB, Router)
│   ├── eval/                         # Causal prequential test engines and benchmark runners
│   ├── physics/                      # Swing Equation cyber-physical grid stability simulator
│   └── stats/                        # Holm-Bonferroni tests, moving-block bootstrap, ROI models
├── scripts/
│   ├── compile_real_posoco_scada.py  # Compiler for 28 monthly POSOCO SCADA workbooks
│   ├── run_real_posoco_benchmark.py  # Master benchmark suite for real Indian grid data
│   ├── run_fair_comparison_arf3_vs_cgt.py # Fair comparison suite (CGT vs ARF-3 vs Random)
│   └── run_cost_aware_oms_mab.py     # Cost-aware MAB tuning sweep & benchmark runner
├── tests/                            # 40 automated unit, leakage, and physics tests (100% passing)
├── reports/
│   ├── master_research_compendium.md # 50-page comprehensive mathematical compendium
│   ├── presentation_deck.md          # 16-slide academic presentation deck with verdicts
│   └── viva_defense_guide.md         # Comprehensive viva defense and Q&A playbook
└── results/
    ├── tables/                       # Benchmark summaries, weekly block tests, hardware profiles
    └── figures/                      # Publication figures, Pareto curves, duty budget sweeps
```

---

## 8. Reproducibility & Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Prizznce/RM_PROJECT.git
cd RM_PROJECT

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### 2. Run Test Suite
Verify mathematical invariants, causal leakage prevention, and configuration hashes (40 tests):
```bash
pytest tests/ -v
```

### 3. Run Benchmark Pipelines
```bash
# Run fair comparison on primary benchmark stream (CGT vs ARF-3 vs Random-Gate)
python scripts/run_fair_comparison_arf3_vs_cgt.py

# Run cost-aware OMS-MAB baseline validation tuning and test evaluation
python scripts/run_cost_aware_oms_mab.py

# Compile 28 months of real POSOCO Indian grid SCADA data and run benchmark
python scripts/compile_real_posoco_scada.py
python scripts/run_real_posoco_benchmark.py
```

### 4. Launch Interactive Control Room Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 9. Citation & Authors
- **Prince Rajan** (Lead System Architecture & Embedded Profiling)
- **Rajan Patil** (Causal Gating & Drift Analysis)
- **Ankit Ekka** (Real-World SCADA Telemetry & Statistical Testing)
