# CGT: Meta-Controller for Resource-Efficient Data Stream Mining

A drift-aware, cost-aware model switching framework evaluated on continuous hourly multi-fuel electricity stream telemetry (2015–2018, 35,064 records, 22 features).

---

## 1. Formal Task Formulation & Label Definition

### 1.1 The Stream Mining Problem
We evaluate online machine learning models under the strict **Prequential (Test-Then-Train) Evaluation Protocol** (Gama et al., 2013). At each discrete hourly time step $t \in \{1, \dots, T\}$:
1. The model receives an incoming feature vector $x_t \in \mathbb{R}^{22}$.
2. The model outputs a directional prediction $\hat{y}_t \in \{0, 1\}$ before the ground truth label is observed.
3. The true ground-truth label $y_t$ is revealed, and the 0-1 loss $\ell_t = \mathbb{I}(\hat{y}_t \ne y_t)$ is recorded.
4. The model / meta-controller updates its internal state incrementally using $(x_t, y_t)$.

### 1.2 Target Label Definition ($y_t$)
The predictive target is the **1-step ahead hourly demand direction of change**:
$$y_t = \mathbb{I}(\text{Demand}_t > \text{Demand}_{t-1}) \in \{0, 1\}$$

- **$y_t = 1$ (Ramp-Up Event)**: Grid electrical demand increases in hour $t$ compared to hour $t-1$.
- **$y_t = 0$ (Ramp-Down / Steady Event)**: Grid electrical demand decreases or remains identical in hour $t$.

#### Why Directional Classification for Power Systems?
In grid operations and transmission system operator (TSO) dispatch:
- Positive ramping events ($y_t = 1$) require initiating fast-acting spinning reserves (hydro pumped storage, open-cycle gas turbine peakers) to counteract frequency drops below nominal 50.0 Hz.
- Negative ramping events ($y_t = 0$) require curtailment or throttling of dispatchable units without tripping baseload thermal or nuclear units.
- Anticipating ramp direction 1 hour ahead enables automated battery energy storage systems (BESS) and demand response aggregators to pre-commit ramp capacity at minimal operational expenditure.

### 1.3 Class Distribution & Trivial Baseline
Across the canonical 35,064-record hourly stream:
- **Full Stream ($N = 35,040$ after 24-hr warm-up)**:
  - Class 0 ($\le$ previous hour): **55.25%** (19,359 instances)
  - Class 1 ($>$ previous hour): **44.75%** (15,681 instances)
- **Frozen Test Stream ($N = 24,528$ instances, 70% split)**:
  - Class 0 ($\le$ previous hour): **55.49%** (13,610 instances)
  - Class 1 ($>$ previous hour): **44.51%** (10,918 instances)
- **Zero-Skill Majority Class Predictor**: **55.48% accuracy**. Any valid machine learning model must substantially exceed 55.48% to demonstrate non-trivial predictive utility.

### 1.4 Input Feature Space ($x_t \in \mathbb{R}^{22}$)
All features are constructed strictly using causal historical telemetry observed prior to hour $t$ (guaranteeing zero future leakage, verified by `tests/test_leakage.py`):
1. **Demand Historical Lags**: $\text{Demand}_{t-1}, \text{Demand}_{t-2}, \text{Demand}_{t-24}$ (MW).
2. **Short-Term & Diurnal Momentum**: $\Delta \text{Demand}_{1-2} = \text{Demand}_{t-1} - \text{Demand}_{t-2}$, and diurnal 24-hour cycle momentum $\Delta \text{Demand}_{1-24} = \text{Demand}_{t-1} - \text{Demand}_{t-24}$.
3. **Multi-Fuel Generation Telemetry**: $\text{Nuclear}_{t-1}, \text{Gas}_{t-1}, \text{Hard Coal}_{t-1}, \text{Hydro Reservoir}_{t-1}, \text{Solar}_{t-1}, \text{Wind}_{t-1}, \text{Total Generation}_{t-1}$ (MW).
4. **Grid Dispatch Dynamics**: Net load ($\text{Demand}_{t-1} - [\text{Solar}_{t-1} + \text{Wind}_{t-1}]$) and renewable penetration ratio $(\frac{\text{Solar}_{t-1} + \text{Wind}_{t-1}}{\text{Total Generation}_{t-1}})$.
5. **Electricity Spot Price Lags**: $\text{Price}_{t-1}$ (EUR/MWh) and hourly price gradient $\Delta \text{Price}_{1-2}$.
6. **Cyclical Calendar Harmonics**: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, and $\text{DayOfWeek} \in [0, 6]$.

---

## 2. Experimental Protocol & Freeze Integrity

- **Validation Split (30%, 10,512 instances)**: Used strictly for hyperparameter calibration (stability horizon $K$, buffer size $B$, ADWIN sensitivity $\delta$, margin thresholds).
- **Test Stream Split (70%, 24,528 instances)**: Evaluated strictly once under frozen parameter configurations.
- **Reproducibility Guarantees**:
  - **Config v1 (Baseline)**: Locked in [`configs/frozen.yaml`](file:///d:/SEM5/ResearchMgmt-2/configs/frozen.yaml) with SHA-256 `fdf3d7e401fffb7c4c9a9f25145c3e0f473c5aeb4bd6f4cf42acefc0ad9f5747`.
  - **Config v2 (Dual-Gate)**: Locked in [`configs/frozen_v2.yaml`](file:///d:/SEM5/ResearchMgmt-2/configs/frozen_v2.yaml) with SHA-256 `87eaead5e13fe02d6f356148410a3aae30d4a5c912e65dbb27907c14e167291c`.
  - **Config v3 (Learned-Gate)**: Locked in [`configs/frozen_v3.yaml`](file:///d:/SEM5/ResearchMgmt-2/configs/frozen_v3.yaml) with SHA-256 `b61c961dd2d938318b875dd30d9307f0dc034eb9da50c478344c7f041c926fc3`.
  - Automated verification via `tests/test_reproducibility.py`.

---

## 3. Issues List Ranked by Risk to Final Claims

| Rank | Severity | Issue & Risk Description | Root Cause Diagnosis | Remediation Plan |
| :---: | :---: | :--- | :--- | :--- |
| **#1** | **RESOLVED** | **CGT Marginal Accuracy Ceiling (+0.06% to +0.14% over HT)**<br>CGT-v1 had tiny effect size because pure ADWIN triggered only 2-3 times, missing steady-state ensemble gains. | ADWIN only alarms on macro concept drift, ignoring HT's steady-state errors. | **Resolved in CGT-v2** (`src/controllers/cgt_v2.py`): Integrated probability margin uncertainty gating ($M(x) < 0.25$) with cooldown ($H=100$) and compute budget cap ($\rho \le 0.15$). Lifts validation accuracy by **$+0.76\%$** while preserving 94.1% dormancy. |
| **#2** | **RESOLVED** | **A4 Learned Router Degeneracy (Identical to HT at 85.07%)**<br>Original router collapsed due to excessive cost penalty. | Penalty $\lambda_c = 0.3$ with costs $1.0$ vs $0.1$ created a 27% loss offset against ARF. | **Resolved in A4FixedRouter**: calibrated $\lambda_c = 0.02$, added $\epsilon$-greedy exploration ($\epsilon=0.05$), recovering **88.73%** accuracy. |
| **#3** | **HIGH** | **Single-Seed Statistical Vulnerability**<br>Benchmarks currently rely on a single execution run with `seed: 42`. | Streaming algorithms like ARF rely on Poisson bootstrapping ($\lambda=6$) and random subspace feature sampling. A single random seed does not capture variance, confidence intervals, or effect sizes (Cohen's $d$, Cliff's delta). | Expand benchmark suite to execute across 5 or 10 distinct random seeds on the stream, reporting mean $\pm$ std, 95% confidence intervals, and non-parametric effect sizes. |
| **#4** | **MEDIUM** | **Dataset Provenance & Scope Discrepancy**<br>Initial project documentation specified Grid-India hourly (2024–2025: Demand, Wind, Solar). Current evaluation runs on ENTSO-E hourly (2015–2018). | While the 2015–2018 multi-fuel dataset is richer (22 features vs 4), altering the dataset without cross-dataset validation exposes the project to critique regarding dataset opportunism. | Add a secondary stream evaluation (e.g. the 2024–2025 Grid-India stream or an out-of-distribution grid segment) as an external transfer benchmark to prove generalization across different grid topographies. |
| **#5** | **RESOLVED** | **Missing Baseline: OMS-MAB (Wilson et al., 2024)**<br>Primary literature paper was cited but not implemented. | Omission of standard baseline. | **Resolved in Day 4**: Implemented [`src/controllers/oms_mab.py`](file:///d:/SEM5/ResearchMgmt-2/src/controllers/oms_mab.py) with Discounted-UCB and per-arm HDDM_A drift monitors. |
| **#6** | **LOW/MED** | **Heuristic Edge ROI Assumptions**<br>Hardware savings (65W TDP, \$1.50/M cloud calls, 0.72 kg CO₂/kWh) are point estimates without parameterization. | Evaluators and viva examiners can challenge static assumptions as arbitrary or overly optimistic. | Implement a parameterized sensitivity analysis module in `src/stats/roi_calculator.py` sweeping over edge hardware profiles (Microcontroller 5W, Edge Gateway 25W, Server 65W) and cloud tier pricings. |

---

## 4. Repository Structure & Execution
```
├── configs/
│   ├── default.yaml            # Base experimental configuration
│   ├── frozen.yaml             # Frozen experimental parameters v1 (SHA-256 locked)
│   ├── frozen_hash.txt         # Verification hash v1
│   ├── frozen_v2.yaml          # Frozen experimental parameters v2 (Tuned on validation only)
│   ├── frozen_v2_hash.txt      # Verification hash v2
│   ├── frozen_v3.yaml          # Frozen experimental parameters v3 (Learned-gate freeze)
│   └── frozen_v3_hash.txt      # Verification hash v3
├── data/
│   └── energy_dataset_clean.csv # 35,064 hourly records (2015–2018, 22 features)
├── src/
│   ├── data/                   # Causal stream loader and audit scripts
│   ├── models/                 # Base learners (HT, ARF) & stronger baselines (HAT, SGD, kNN, LevBag, SRP)
│   ├── controllers/            # Meta-controllers (CGT-v1, CGT-v2, CGT-v3, OMS-MAB, A1-A4, Oracle)
│   ├── eval/                   # Prequential evaluation engines and clean benchmark runner
│   ├── analysis/               # Gate diagnosis scripts and Day 2 experiments
│   ├── stats/                  # Wilcoxon tests, ROI calculator, latency profiling
│   └── plots/                  # Publication figures generator (fig1-fig4, day2, day3, day6)
├── tests/                      # 15 automated unit, leakage, and reproducibility tests
└── results/
    ├── raw/                    # Per-instance step predictions & controlled nanosecond timing
    ├── tables/                 # Benchmark tables, validation sweeps, and contingency counts
    └── figures/                # Publication figures (fig1-fig4, day2, day3, day6, day7_drift, day7_pareto)
```

## Quick Start
```bash
# Run all 15 automated unit, leakage, and reproducibility tests
.venv\python.exe -m pytest tests/ -v

# Run clean frozen benchmark across all 18 models on 24,528 test instances
.venv\python.exe -m src.eval.run_final_test_benchmark

# Run Day 4 OMS-MAB and Day 5 validation tuning sweep
.venv\python.exe -m src.eval.tune_validation_v2

# Run full baseline suite (16 models)
.venv\python.exe -m src.eval.run_day3_baselines

# Launch live Streamlit dashboard
.venv\python.exe -m streamlit run app/dashboard.py
```
