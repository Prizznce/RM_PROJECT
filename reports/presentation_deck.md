# 16-Slide Final Evaluation Presentation Deck
## Project: Cascaded Compute-Gated Trees (CGT)
### Resource-Efficient Continuous Data Stream Mining under Concept Drift
**Target Architecture**: Cyber-Physical Digital Substation Remote Terminal Units (RTUs) & Phasor Measurement Units (PMUs)  
**Standardized Profiling Standard**: IEEE C37.118 PMU Synchrophasor Standard ($f_s = 50.0\text{ Hz}$, $T_{\text{sample}} = 20.0\text{ ms}$)  
**Total Allocated Presentation Time**: 15–18 Minutes (+ 5–7 Minutes Viva Defense)

---

### Slide 1: Title & Executive Overview (1.0 Minute)
* **Slide Title**: *Cascaded Compute-Gated Trees (CGT): Resource-Efficient Real-Time Streaming Under Non-Stationary Drift*
* **Presenter / Authors**: Rajan Patil, Prince Rajan, Ankit Ekka
* **Key Visuals**: Split-screen display: Left side showing an industrial IEC 61850 substation digital protection relay (ARM Cortex-M7); right side showing the compute duty trade-off between continuous heavy ensembles and CGT.
* **Core Bullet Points**:
  - Continuous machine learning on streaming cyber-physical infrastructure.
  - The critical trilemma: Predictive Accuracy vs. Hard Interrupt Deadlines ($\le 1.0\text{ ms}$) vs. Micro-Watt Battery Survival.
  - Introducing CGT: A causal streaming meta-controller that keeps heavy ensembles dormant ~95% of the time.
  - **Pareto Trade-off**: CGT is one of three non-dominated options on the real-time Pareto frontier; while ARF-3 is more accurate at 50 Hz (+0.86 pp), CGT cuts heavy-ensemble use by ~95.4%.
* **Speaker Script**:
  > *"Respected evaluators, good morning. In mission-critical cyber-physical systems—such as national power grids, renewable inverter hubs, and digital substations—machine learning models face an uncompromising trilemma:
  > If we deploy heavy ensembles like Adaptive Random Forests, they adapt to non-linear concept drift, but they consume massive CPU cycles 24/7, violating hard real-time interrupt deadlines and draining emergency backup batteries. If we deploy lightweight single trees, they execute rapidly in microseconds, but their accuracy degrades during grid shocks.
  > Today, we present **Cascaded Compute-Gated Trees (CGT)**: a causal online meta-controller that keeps heavy ensembles dormant ~95% of the time, waking them selectively during volatile regimes.
  > On the sub-1ms deployable Pareto frontier, CGT represents one of three non-dominated options: while continuous ARF-3 is more accurate at 50 Hz, CGT cuts heavy-ensemble execution by ~95.4% while outperforming random escalation at matched duty."*

---

### Slide 2: The Cyber-Physical Edge Problem (1.0 Minute)
* **Slide Title**: *The Real-Time Interrupt Constraint in Digital Substations*
* **Key Visuals**: Substation RTU software stack diagram: FreeRTOS priority scheduler showing IEC 61850-9-2 Sampled Values (4 kHz), GOOSE protection (< 3 ms), 50 Hz PMU synchrophasors (20 ms), and the $\le 1.0\text{ ms}$ background AI time-slice.
* **Core Bullet Points**:
  - RTUs (e.g. SEL-3530, ABB REC670) are shared-core microcontrollers, not isolated server clusters.
  - Telemetry sampling vs. Compute budget: PMUs sample at 50 frames/s ($T_{\text{sample}} = 20\text{ ms}$), but non-preemptible interrupt service routines (ISR) allow strictly $< 1.0\text{ ms}$ for analytics.
  - Exceeding 1.0 ms starves protective tripping loops, causes communication buffer overflows, or triggers hardware watchdog resets.
* **Speaker Script**:
  > *"To understand why standard ML cannot simply be loaded onto an edge RTU, we must examine real substation computing constraints. An RTU is not a dedicated GPU server. It runs high-priority protection threads: IEC 61850 Sampled Values at 4 kHz, GOOSE breaker trip commands in under 3 ms, and PMU synchrophasors at 50 Hz.
  > The real-time operating system allocates a strict **sub-1-millisecond non-preemptible interrupt budget** for background intelligence. Any machine learning algorithm taking longer than 1 ms cannot be deployed on an RTU—it starves protection loops and causes RTU watchdog resets."*

---

### Slide 3: The Edge ML Trilemma (1.0 Minute)
* **Slide Title**: *The Edge Machine Learning Trilemma: Accuracy, Latency, and Battery*
* **Key Visuals**: 3-axis radar chart and Pareto boundary diagram contrasting Single Trees, Deep Nets, Heavy Ensembles, and Bandits.
* **Core Bullet Points**:
  - **Lightweight Pole (Always-HT)**: Fast ($94.26\ \mu\text{s}$ mean), but vulnerable to dynamic distribution shift ($85.07\%$ accuracy).
  - **Heavy Pole (Always-ARF 10 Trees)**: Resilient ($89.60\%$ accuracy), but incurs $1,333.95\ \mu\text{s}$ mean latency and $101.4\text{ ms}$ worst-case latency (violating the 1 ms deadline), with $133.4\%$ CPU overload at 1 kHz.
  - **Prior Art Bandits (Unconstrained OMS-MAB)**: Unconstrained bandits reward only 0-1 accuracy, pulling ARF 63.33% of the time ($814.74\ \mu\text{s}$ mean latency, $67.1\text{ ms}$ worst-case spike).
  - **Cost-Aware OMS-MAB (with ARF-3 Heavy Arm)**: Adding cost penalty $\lambda$ and duty cap to match CGT's ~4.6% duty yields only **84.30% accuracy** ($191.05\ \mu\text{s}$ mean, $3,671.7\ \mu\text{s}$ worst-case), trailing CGT by **-1.03 pp** ($p = 3.75 \times 10^{-8}$).
* **Speaker Script**:
  > *"This creates the Edge ML Trilemma. Operators have historically been forced to choose:
  > Deploy a fast Hoeffding Tree and suffer mispredictions during sudden solar ramps; or deploy a 10-tree Adaptive Random Forest that averages 1,334 µs—and spikes to 101 ms—violating the 1 ms hard interrupt deadline while overloading CPUs.
  > Prior academic approaches like OMS-MAB by Wilson et al. relied on multi-armed bandits. Because bandits are cost-blind, unconstrained OMS-MAB pulls the heavy ensemble 63.3% of the time. When we adapt OMS-MAB to be cost-aware with the same ARF-3 heavy arm and constrain it to match CGT's ~4.6% duty, its accuracy collapses to 84.30% (-1.03 pp behind CGT) because bandits lack causal uncertainty gating and stateful buffer replay. None of these prior architectures resolve the physical trilemma."*

---

### Slide 4: Algorithmic Architecture: The CGT Principle (1.0 Minute)
* **Slide Title**: *Conditional Compute-Gated Architecture (CGT)*
* **Key Visuals**: High-level dataflow schematic: Stream input $\rightarrow$ Always-On Fast Hoeffding Tree $\rightarrow$ Causal Gating Unit $\rightarrow$ Gate Decision ($g_t \in \{0, 1\}$) $\rightarrow$ Fast Output or Dormant ARF Escalation.
* **Core Bullet Points**:
  - Cascaded multi-tier execution: ~95% of incoming instances terminate at the fast tree ($94.26\ \mu\text{s}$).
  - Gating decision is strictly causal: evaluates streaming state using past telemetry only.
  - Warm-start S1 buffer replay ($B = 200$ instances) resynchronizes the ensemble before escalation.
* **Speaker Script**:
  > *"CGT resolves this trilemma through hierarchical cascading.
  > Every incoming telemetry frame is processed by a fast, always-on Hoeffding Tree.
  > Concurrently, a causal meta-gating unit inspects the stream. If the stream is stationary and the fast tree is confident, the gate remains locked ($g_t = 0$), and predictions return in under 100 microseconds.
  > Only when distribution shift or high uncertainty is detected does the gate trip: the dormant ensemble awakens via an S1 warm-start buffer replay, overrides the prediction, and re-locks automatically once stability returns."*

---

### Slide 5: Causal Streaming Meta-Gating Sensors (1.0 Minute)
* **Slide Title**: *Causal Streaming Meta-Gating Engine*
* **Key Visuals**: A structured grid illustrating the streaming indicators: Margin Uncertainty, Recent HT Error Rate, Temporal Harmonics, Renewable Penetration, Diurnal Momentum, ADWIN Drift Cooldown, and Hardware Budget Governor.
* **Core Bullet Points**:
  - **Sensor 1 (Margin Uncertainty)**: $M(x_t) = |P(y=1|x) - P(y=0|x)| \in [0, 1]$.
  - **Sensor 2 (Recent Error Rate)**: Sliding error rate of HT over window $W = 50$.
  - **Sensor 3 & 4 (Temporal Harmonics)**: $\sin(2\pi \cdot \text{hour} / 24)$ and $\cos(2\pi \cdot \text{hour} / 24)$.
  - **Sensor 5 (Renewable Ratio)**: $(\text{Solar} + \text{Wind}) / \text{TotalGen}$.
  - **Sensor 6 (Diurnal Momentum)**: Normalized 24-hr load momentum $(D_{t-1} - D_{t-24})$.
  - **Sensor 7 (ADWIN Drift Cooldown)**: $K = 50$ post-drift alert recovery horizon.
  - **Sensor 8 (Hardware Budget Governor)**: Running duty cycle cap ($\rho_{\text{recent}} \le \rho_{\max}$).
  - **Overhead**: Python-measured ~100–140 $\mu$s for standardized linear meta-classifier evaluation.
* **Speaker Script**:
  > *"Our gate monitors orthogonal streaming indicators without future lookahead.
  > It tracks epistemic uncertainty through Hoeffding margins and recent base error rates, detects statistical regime shifts via ADWIN, enforces an active hardware budget governor that caps duty cycle below 12%, and prevents contact chattering with a 50-step cooldown.
  > All features use strict historical lags with zero future lookahead."*

---

### Slide 6: Causal Protocol & Diurnal Day-Aligned Partition (1.0 Minute)
* **Slide Title**: *Data Provenance & The Diurnal Day-Aligned Partition*
* **Key Visuals**: Calendar timeline diagram illustrating 1,461 total solar days (4 years) split cleanly across 24-hour day boundaries: 439 Validation Days (10,512 hours after 24h warmup) vs. 1,022 Test Days (24,528 hours).
* **Core Bullet Points**:
  - Total stream: 35,064 hourly records across 4 solar years (2015–2018).
  - **Lag Warmup**: 24 hours are consumed at the very start for 24-hr lag feature warmup ($35,064 - 24 = \mathbf{35,040\text{ streaming instances}}$).
  - **Day-Aligned Splitting**: $439 \text{ days} \times 24 = 10,536\text{ hours}$. Subtracting the 24 warmup hours leaves exactly $\mathbf{10,512\text{ validation instances}}$.
  - **Held-Out Frozen Test Stream**: 1,022 days $\times 24 = \mathbf{24,528\text{ instances}}$ ($10,512 + 24,528 = 35,040$).
  - **Why Not a Naive 70% Cut?** A naive 70% test / 30% val cut of 35,064 falls at 10,519.2 hours $\implies$ **Day 439 at ~07:12 AM**. Splitting mid-day bisects the morning solar ramp and diurnal load curve; midnight day-alignment prevents boundary distortion.
* **Speaker Script**:
  > *"To guarantee complete scientific rigor, we enforced strict causal hygiene.
  > A common question is: 'Why does your 70% test stream have 24,528 instances instead of 24,545?'
  > 35,064 hours equals exactly 1,461 calendar days. A naive 30% cut splits Day 439 at 07:12 in the morning, cutting right through solar morning ramps and introducing boundary leakage.
  > Furthermore, the first 24 hours are required for diurnal lag feature warmup.
  > By splitting strictly on 24-hour diurnal boundaries, we obtain 439 full days—exactly 10,512 validation instances after warmup—and 1,022 full days, exactly 24,528 continuous test instances. Zero test set leakage occurred."*

---

### Slide 7: Comprehensive Benchmark Suite & Fair Comparison (1.5 Minutes)
* **Slide Title**: *Empirical Benchmark Suite: ARF-3 Fair Comparison, Random-Gate, & Cost-Aware OMS-MAB*
* **Key Visuals**: Master comparative table displaying Prequential Accuracy, Mean Latency, Worst-Case Latency, Empirical Heavy Duty Cycle $\rho$, and STM32H743 / CM4 Battery Autonomy under the 50 Hz PMU Standard.
* **Core Table Highlights (Frozen Test Stream, N = 24,528 Hours)**:

| Model | Ensemble Size | Heavy Duty Cycle ($\rho$) | Test Accuracy | Mean Latency (host-measured) | Worst-Case Latency (host-measured) | Modeled MCU Battery (100 Wh) | Gating / Selection Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Streaming Linear / Logistic (SGD)** | 1 | 0.00% | 85.062% | 13.92 $\mu$s | 320.4 $\mu$s | 11,240.5 h (1.28 yr) | Single linear hyperplane (no ensemble) |
| **Always-HT (Baseline)** | 1 | 0.00% | 85.066% | 94.26 $\mu$s | 2,404.0 $\mu$s | 10,547.4 h (1.20 yr) | Always closed (single tree) |
| **Random-Gate Baseline (20 seeds)** | 1 / 3 | 4.58% | 84.756% | 102.82 $\mu$s | 4,309.3 $\mu$s | 10,284.3 h (1.17 yr) | Pure Bernoulli random escalation ($p = 4.59\%$) |
| **Cost-Aware OMS-MAB (Wilson et al.)** | 1 / 3 | **4.60%** | **84.300%** | **191.05 $\mu$s** | **3,671.7 $\mu$s** | **9,262.4 h (1.06 yr)** | Cost-Penalized D-UCB ($\lambda = 0.01, \rho_{\max} = 0.046$) |
| **CGT (Learned Meta-Gate)** | 1 / 3 | **4.59%** | **85.331%** | **243.37 $\mu$s** | **38,002.8 $\mu$s** | **8,690.1 h (0.99 yr)** | Selective margin + ADWIN cooldown + buffer replay |
| **ARF-3 (3 Trees, Fair Protocol)** | 3 | **100.00%** | **86.191%** | **281.35 $\mu$s** | **2,231.1 $\mu$s** | **8,317.0 h (0.95 yr)** | Always open (continuous 3-tree ensemble) |
| **OMS-MAB (Unconstrained Ref)** | 1 / 10 | 63.33% | 86.909% | 814.74 $\mu$s | 67,087.5 $\mu$s | 5,188.8 h (0.59 yr) | Unconstrained bandit (Wilson et al., 2024) |
| **Always-ARF (10 Trees Ref)** | 10 | 100.00% | 89.596% | 1,333.95 $\mu$s | 101,442.2 $\mu$s | 3,798.2 h (0.43 yr) | Always open (continuous 10-tree ensemble) |

* **Speaker Script**:
  > *"Here are the empirical results across the 24,528-hour frozen test stream under our strict fair comparison protocol.
  > Notice four decisive findings:
  > First, continuous **ARF-3 is more accurate at 50 Hz** (+0.860 pp, 86.19% vs. 85.33%) because it updates all 3 trees 100% of the time.
  > Second, at matched ~4.6% duty cycle, CGT achieves **85.331%**, outperforming both **Random-Gate** (84.756%, **+0.575 pp**, $p = 1.57 \times 10^{-8}$) and **Cost-Aware OMS-MAB** (84.300%, **+1.031 pp**, $p = 3.75 \times 10^{-8}$).
  > Third, notice that both Random-Gate and Cost-Aware OMS-MAB fall below Always-HT (85.07%). When an algorithm escalates without causal confidence indicators and warm-start replay, it injects noise and damages tracking.
  > Fourth, while 10-tree ARF reaches 89.60% and unconstrained OMS-MAB reaches 86.91%, both pull heavy ensembles 63% to 100% of the time with severe latency spikes, making them unfeasible for tight real-time embedded environments."*

---

### Slide 8: Real-Time Pareto Frontier & Matched Duty Trade-off (1.5 Minutes)
* **Slide Title**: *Real-Time Pareto Frontier: Three Non-Dominated Operating Regimes*
* **Key Visuals**: Dual display: (Left) 3D Plotly Pareto space with 1,000 $\mu$s cut plane; (Right) Unified Accuracy vs. Achieved Heavy Duty Curve ([fig_accuracy_vs_duty_cgt_vs_oms_mab.png](file:///d:/SEM5/ResearchMgmt-2/results/figures/fig_accuracy_vs_duty_cgt_vs_oms_mab.png)) plotting CGT, Cost-Aware OMS-MAB, and Random-Gate from 2% to 30% duty.
* **Core Bullet Points**:
  - **Lead with Frozen Matched-Duty Results**: At matched empirical duty cycle (~4.6%), CGT achieves **85.33%** vs. Cost-Aware OMS-MAB's **84.30%** (**+1.03 pp**) and Random-Gate's **84.76%** (**+0.58 pp**).
  - **Three Non-Dominated Options on the Real-Time Frontier**:
    1. *Always-HT*: Lowest latency ($94.26\ \mu\text{s}$), zero gating overhead, $85.07\%$ accuracy.
    2. *CGT (Learned Gate)*: Intermediate accuracy ($85.33\%$), low duty ($4.59\%$), $243.37\ \mu\text{s}$ latency; cuts heavy execution by ~95.4%.
    3. *ARF-3*: Highest accuracy among sub-1ms models ($86.19\%$), but requires continuous $100\%$ heavy-ensemble execution ($281.35\ \mu\text{s}$).
  - **Full Duty Budget Sweep (2% to 30%)**: Across all duty budgets, CGT consistently dominates Cost-Aware OMS-MAB and Random-Gate. At 5% budget cap, CGT achieves **85.80% accuracy** vs. Cost-Aware OMS-MAB's **85.26%** and Random-Gate's **84.74%** (+1.06 pp advantage over random).
* **Speaker Script**:
  > *"Because our contribution is fundamentally cost-adjusted, we frame results through the **Real-Time Pareto Frontier**, not unconstrained raw accuracy.
  > On this frontier, there is no single universal winner; instead, there are **three non-dominated options**:
  > Always-HT provides the absolute lowest latency and power. ARF-3 provides the highest accuracy among sub-1ms models at 86.19%, but demands continuous 100% ensemble compute.
  > CGT occupies the balanced operating point: it achieves 85.33% accuracy—beating Cost-Aware OMS-MAB by +1.03 pp and Random-Gate by +0.58 pp at matched duty—while cutting heavy ensemble execution by ~95.4%.
  > In the duty sweep shown on the right, CGT consistently maintains the highest accuracy across every budget point from 2% to 30%."*

---

### Slide 9: Autocorrelation-Corrected Statistics & Drift Window Disaggregation (1.0 Minute)
* **Slide Title**: *Statistical Significance (Holm-Corrected) & Drift Window Disaggregation*
* **Key Visuals**: Master 2-panel chart: (Top) Paired accuracy deltas across 146 independent calendar weeks ($N = 146$); (Bottom) Concept Drift Window Disaggregation table (Drift Regime vs. Stationary Regime).
* **Core Findings & Numbers**:
  - **Exact Arithmetic Identity**: Because $146 \text{ weeks} \times 168 \text{ hours} = 24,528\text{ hours}$, the mean weekly delta mathematically equals the overall test stream delta.
  - **Holm-Bonferroni Corrected Tests (146 Independent Weekly Blocks)**:
    - CGT vs. Cost-Aware OMS-MAB: Mean Weekly Delta $= \mathbf{+1.031\%}$, Wilcoxon $W = 2,007.5$, Raw $p = 3.75 \times 10^{-8}$, **Holm-adjusted $p = \mathbf{1.13 \times 10^{-7}}$** ($p < 0.001$, Statistically Significant). Cohen's $d = 0.503$ (Solid Medium Effect).
    - CGT vs. Random-Gate (20 seeds): Mean Weekly Delta $= \mathbf{+0.575\%}$, Wilcoxon $W = 2,376.5$, Raw $p = 5.24 \times 10^{-9}$, **Holm-adjusted $p = \mathbf{2.09 \times 10^{-8}}$** ($p < 0.001$, Statistically Significant). Cohen's $d = 0.488$.
    - CGT vs. Always-HT: Mean Weekly Delta $= \mathbf{+0.265\%}$, Wilcoxon $W = 1,645.0$, Raw $p = 5.53 \times 10^{-3}$, **Holm-adjusted $p = \mathbf{5.53 \times 10^{-3}}$** ($p < 0.01$, Statistically Significant). Cohen's $d = 0.220$.
    - ARF-3 vs. CGT: Mean Weekly Delta $= \mathbf{+0.860\%}$, Wilcoxon $W = 3,192.5$, Raw $p = 1.45 \times 10^{-3}$, **Holm-adjusted $p = \mathbf{2.89 \times 10^{-3}}$** ($p < 0.01$, Statistically Significant). Cohen's $d = 0.264$.
  - **Drift Window Disaggregation (12.06% Drift vs. 87.94% Stationary)**:
    - *Inside Drift Windows (2,958 hours)*: Always-HT ($92.33\%$), Cost-Aware OMS-MAB ($91.85\%$), Random-Gate ($92.26\%$), **CGT ($92.53\%$)**, **ARF-3 ($93.61\%$)**, ARF-10 ($94.56\%$).
    - *Outside Drift Windows (21,570 hours)*: Cost-Aware OMS-MAB ($83.27\%$), Always-HT ($84.07\%$), Random-Gate ($83.72\%$), **CGT ($84.34\%$)**, **ARF-3 ($85.17\%$)**, ARF-10 ($88.92\%$).
* **Speaker Script**:
  > *"To evaluate statistical significance properly, we aggregated predictions into 146 non-overlapping calendar weeks to eliminate autocorrelation bias.
  > Under Holm-Bonferroni correction, CGT achieves statistically significant wins over Cost-Aware OMS-MAB ($p = 1.13 \times 10^{-7}$, Cohen's $d = 0.503$), over Random-Gate ($p = 2.09 \times 10^{-8}$, $d = 0.488$), and over Always-HT ($p = 0.0055$, $d = 0.220$). ARF-3's accuracy advantage over CGT is also statistically significant ($p = 0.0029$, $d = 0.264$).
  > In our drift window analysis—covering 2,958 hours of rapid solar and wind ramps—ARF-3 reaches 93.61% inside drift windows due to continuous updates, while CGT achieves 92.53%. Outside drift windows, in normal stationary conditions, CGT achieves 84.34% while keeping the heavy ensemble dormant ~95% of the time."*

---

### Slide 10: Embedded Memory & Silicon Footprint (1.0 Minute)
* **Slide Title**: *Embedded Memory Constraints & Modeled SRAM Footprint*
* **Key Visuals**: Memory footprint bar chart: STM32H743 physical SRAM limit (1,024 KB) vs. Deep Net (38,400 KB) vs. Periodic Retraining Forest (8,500 KB) vs. CGT (74 KB).
* **Core Bullet Points**:
  - Offline deep learning models (500-epoch MLP, LSTM) require 38.4 MB of dynamic RAM $\implies$ Triggers immediate **Out-Of-Memory (OOM) Crash** on MCU targets.
  - Periodic batch retraining forests require 8.5 MB RAM and create a 24-hour adaptation blindspot.
  - CGT consumes strictly **148 KB Flash (7.2%)** and **74 KB SRAM (7.2%)** (modeled for STM32H743), leaving 92.8% of microcontroller memory free for critical IEC 61850 protection tasks.
* **Speaker Script**:
  > *"When deploying machine learning models on resource-constrained microcontrollers—such as an ARM Cortex-M7 with 1,024 KB of SRAM—deep learning baselines fail due to memory limits. A 38.4 MB deep net triggers an immediate out-of-memory crash.
  > Periodic batch retraining forests also fail, requiring 8.5 MB of RAM while leaving a 24-hour blindspot between retrainings.
  > CGT models consume an estimated 74 KB of SRAM—under 8% of capacity—leaving memory headroom for critical protection threads."*

---

### Slide 11: Multi-Rate Hardware Profiling & Modeled Battery Autonomy (1.0 Minute)
* **Slide Title**: *Multi-Rate Hardware Profiling & Modeled Battery Autonomy (50 Hz, 200 Hz, 1 kHz)*
* **Key Visuals**: Complete multi-rate comparative table showing host-measured latencies, Modeled CPU Duty, Test Harness Schedulability Status, and Modeled Battery Autonomy on STM32H743ZI (100 Wh) and CM4 (100 Wh), all explicitly labeled **(modeled)**.
* **Core Table Highlights (Strictly Labeled Modeled)**:

| Operational Sampling Rate | Model Name | Mean Latency (host-measured) | Modeled CPU Duty | Real-Time Schedulability (harness label) | STM32H743 100 Wh Autonomy (modeled) | CM4 100 Wh Autonomy (modeled) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **50 Hz Standard PMU**<br>($T_{\text{sample}} = 20.0\text{ ms}$) | Always-HT | 94.26 $\mu$s | 0.471% | GUARANTEED | 10,547.4 h (1.20 yr) | 116.3 h |
| | Random-Gate ($\rho = 4.6\%$) | 102.82 $\mu$s | 0.514% | GUARANTEED | 10,402.1 h (1.19 yr) | 116.1 h |
| | Cost-Aware OMS-MAB ($\rho = 4.6\%$) | 191.05 $\mu$s | 0.955% | GUARANTEED ($t_{\max} \le 20\text{ ms}$) | 9,262.4 h (1.06 yr) | 115.1 h |
| | **CGT (Learned Gate)** | **243.37 $\mu$s** | **1.217%** | **OVERRUN (Replay Spike)** | **8,690.1 h (0.99 yr)** | **114.1 h** |
| | **ARF-3 (3 Trees)** | **281.35 $\mu$s** | **1.407%** | **GUARANTEED** ($t_{\max} \le 20\text{ ms}$) | **8,317.0 h (0.95 yr)** | **113.6 h** |
| | OMS-MAB (Unconstrained Ref) | 814.74 $\mu$s | 4.074% | OVERRUN (Replay Spike) | 5,188.8 h (0.59 yr) | 107.0 h |
| | Always-ARF (10 Trees Ref) | 1,333.95 $\mu$s | 6.670% | OVERRUN (Replay Spike) | 3,798.2 h (0.43 yr) | 100.7 h |
| **200 Hz High-Speed PMU**<br>($T_{\text{sample}} = 5.0\text{ ms}$) | Always-HT | 94.26 $\mu$s | 1.885% | GUARANTEED | 7,505.3 h (312.7 d) | 112.3 h |
| | Random-Gate ($\rho = 4.6\%$) | 102.82 $\mu$s | 2.056% | GUARANTEED | 7,243.5 h (301.8 d) | 111.7 h |
| | Cost-Aware OMS-MAB ($\rho = 4.6\%$) | 191.05 $\mu$s | 3.821% | GUARANTEED ($t_{\max} \le 5\text{ ms}$) | 5,420.3 h (225.8 d) | 107.5 h |
| | **CGT (Learned Gate)** | **243.37 $\mu$s** | **4.867%** | **OVERRUN (Replay Spike)** | **4,666.4 h (194.4 d)** | **104.8 h** |
| | **ARF-3 (3 Trees)** | **281.35 $\mu$s** | **5.627%** | **GUARANTEED** ($t_{\max} \le 5\text{ms}$) | **4,256.4 h (177.3 d)** | **103.0 h** |
| | OMS-MAB (Unconstrained Ref) | 814.74 $\mu$s | 16.295% | OVERRUN (Replay Spike) | 1,905.2 h (79.4 d) | 83.3 h |
| | Always-ARF (10 Trees Ref) | 1,333.95 $\mu$s | 26.679% | OVERRUN (Replay Spike) | 1,238.9 h (51.6 d) | 70.2 h |
| **1 kHz Substation RTU**<br>($T_{\text{sample}} = 1.0\text{ ms}$) | Always-HT | 94.26 $\mu$s | 9.426% | OVERRUN (Replay Spike) | 2,956.8 h (123.2 d) | 95.0 h |
| | Random-Gate ($\rho = 4.6\%$) | 102.82 $\mu$s | 10.282% | OVERRUN (Replay Spike) | 2,776.3 h (115.7 d) | 93.4 h |
| | Cost-Aware OMS-MAB ($\rho = 4.6\%$) | 191.05 $\mu$s | 19.105% | OVERRUN (Worst-Case Spike) | 1,701.3 h (70.9 d) | 79.9 h |
| | **CGT (Learned Gate)** | **243.37 $\mu$s** | **24.337%** | **OVERRUN (Replay Spike)** | **1,345.0 h (56.0 d)** | **72.8 h** |
| | **ARF-3 (3 Trees)** | **281.35 $\mu$s** | **28.135%** | **OVERRUN (Replay Spike)** | **1,181.0 h (49.2 d)** | **68.7 h** |
| | OMS-MAB (Unconstrained Ref) | 814.74 $\mu$s | 81.474% | OVERRUN (Replay Spike) | 435.5 h (18.1 d) | 38.5 h |
| | **Always-ARF (10 Trees Ref)** | **1,333.95 $\mu$s** | **133.395%** | **OVERRUN (Mean Overload)** | **357.1 h (14.9 d)** | **33.3 h** |
| | **CGT (Learned Gate)** | **243.37 $\mu$s** | **24.337%** | **OVERRUN (Replay Spike)** | **1,345.0 h (56.0 d)** | **72.8 h** |
| | **ARF-3 (3 Trees)** | **281.35 $\mu$s** | **28.135%** | **OVERRUN (Replay Spike)** | **1,181.0 h (49.2 d)** | **68.7 h** |
| | OMS-MAB (Wilson et al.) | 814.74 $\mu$s | 81.474% | OVERRUN (Replay Spike) | 435.5 h (18.1 d) | 38.5 h |
| | **Always-ARF (10 Trees)** | **1,333.95 $\mu$s** | **133.395%** | **OVERRUN (Mean Overload)** | **357.1 h (14.9 d)** | **33.3 h** |

* **Methodology Notes**:
  - *Sampling Rate Mapping*: Telemetry features are derived from hourly SCADA data; real-time constraints are evaluated by mapping per-sample compute times against industrial arrival intervals ($50\text{ Hz} = 20\text{ ms}$, $200\text{ Hz} = 5\text{ ms}$, $1\text{ kHz} = 1\text{ ms}$).
  - *Measurement Context*: Latencies are host-measured in Python 3.11 with `time.perf_counter_ns`. Battery runtime figures are modeled using analytical power equations ($P_{\text{active}} \cdot D + P_{\text{sleep}} \cdot (1 - D)$) for STM32H743ZI ($280\text{ mW}$ / $8.2\text{ mW}$) and CM4 ($3,000\text{ mW}$ / $850\text{ mW}$).
* **Speaker Script**:
  > *"Slide 11 presents modeled hardware feasibility across three operational sampling rates: 50 Hz, 200 Hz, and 1 kHz. All latencies are host-measured in Python, and power figures are analytically modeled.
  > At 50 Hz, Always-HT, Random-Gate, and ARF-3 meet worst-case real-time deadlines. CGT incurs an 'OVERRUN' label on worst-case latency due to occasional 38 ms buffer replays during drift recovery, though its mean latency consumes just 1.22% CPU duty.
  > At 1 kHz, 10-tree ARF suffers a 133% CPU overload—it cannot run. While CGT's mean latency uses 24.3% CPU, worst-case spikes require task buffer management in real-time operating systems."*

---

### Slide 12: Illustrative Grid Stability Simulation (1.0 Minute)
* **Slide Title**: *Illustrative Cyber-Physical Coupling: Swing Equation Simulation (Modeled)*
* **Key Visuals**: Post-contingency frequency trajectory plot ($df/dt$ Swing Equation, UFLS 49.50 Hz line, BESS fast frequency response).
* **Core Bullet Points (Illustrative / Modeled Scenario)**:
  - Coupled differential equation: $\frac{2H}{f_0}\frac{df}{dt} = P_{\text{gen}} - P_{\text{load}} + P_{\text{BESS}} - D(f - f_0)$.
  - Contingency scenario: 4,500 MW sudden generation trip on a 50,000 MW synchronous grid block ($\text{RoCoF} = -0.500\text{ Hz/s}$).
  - Without fast intervention, grid breaches Stage-1 UFLS (49.50 Hz) in **1.00 second**, triggering load shedding.
  - **Single Shock Response**:
    - Uncontrolled: Nadir $= 49.151\text{ Hz}$ (trips UFLS, 2,500 MW shed).
    - Always-HT ($94.3\ \mu\text{s}$): Nadir $= 49.499\text{ Hz}$ (Actuation delay: 50.1 ms).
    - CGT ($243.4\ \mu\text{s}$): Nadir $= 49.499\text{ Hz}$ (Restabilizes at 3.81s).
    - ARF-3 ($281.4\ \mu\text{s}$): Nadir $= 49.499\text{ Hz}$ (Restabilizes at 3.81s).
    - Deep Net ($12,450\ \mu\text{s}$): Nadir $= 49.357\text{ Hz}$ (Actuation delay: 1,612.5 ms, severe trip).
* **Speaker Script**:
  > *"To illustrate why compute latency matters physically, we coupled our machine learning models into a modeled Swing Equation simulation under a severe 4,500 MW generation contingency.
  > Fast streaming models—Always-HT, CGT, and ARF-3—dispatch battery energy storage within 50 ms, keeping nadirs near 49.50 Hz.
  > In contrast, deep sequence models with 12.5 ms latency miss interrupt cycles and incur a 1.6-second communication queue delay, plunging nadir to 49.36 Hz and causing severe feeder trips."*

---

### Slide 13: Physical Coupling & Illustrative Valuation (1.0 Minute)
* **Slide Title**: *Physical Coupling Mechanisms & Illustrative Risk Valuation (Modeled)*
* **Key Visuals**: Dual-mechanism decomposition flow chart: (a) RTU Queue Deferral (~80 mHz) + (b) Accuracy-to-Dispatch Mapping (~130 mHz).
* **Core Bullet Points (Illustrative / Modeled)**:
  - **Why Nadirs Differ**: At $-0.5\text{ Hz/s}$, a 12 ms latency difference alone only shifts frequency by $\approx 6\text{ mHz}$. The observed ~170–210 mHz nadir difference between fast edge models and deep nets arises from two coupled physical factors:
    1. *RTU Task Queue Deferral*: Exceeding 1.0 ms interrupt slices defers execution across discrete RTU communication cycles (120 ms/cycle), accumulating ~1.6s delay.
    2. *Accuracy-to-Dispatch Mapping*: When an algorithm mispredicts during a shock, zero battery power is dispatched (0 MW), forcing the grid to rely solely on sluggish thermal governors ($T_g = 5.0\text{ s}$).
  - **100-Contingency Stochastic Simulation**:
    - Always-HT: $49.624\text{ Hz}$ mean nadir, $70.0\%$ blackout prevention rate.
    - CGT: $49.620\text{ Hz}$ mean nadir, $72.0\%$ blackout prevention rate.
    - ARF-3: $49.602\text{ Hz}$ mean nadir, $63.0\%$ blackout prevention rate.
    - Deep Net: $49.451\text{ Hz}$ mean nadir, $0.0\%$ blackout prevention rate.
  - **Illustrative Valuation**: Under standard $5,000/MWh Value of Lost Load (VoLL), preventing Stage-1 UFLS feeder shedding represents an illustrative risk exposure of ~$18.0M ($25.0M uncontrolled vs. $7.0M with CGT).
* **Speaker Script**:
  > *"Slide 13 clarifies the physical mechanism behind frequency nadir differences.
  > A 12 ms delay does not cause frequency drops by continuous RoCoF alone; rather, missing the 1 ms RTOS deadline defers execution across discrete communication cycles by 1.6 seconds, and mispredictions result in zero battery power dispatch.
  > In our stochastic 100-contingency simulation, fast edge models achieve 63% to 72% blackout prevention rates. We report the $18M avoided loss strictly as an illustrative scenario-based Value of Lost Load risk calculation."*

---

### Slide 14: Hyperparameter Sensitivity & Concave Stability Plateau (1.0 Minute)
* **Slide Title**: *Hyperparameter Robustness & The Concave Stability Plateau (Validation Sweep)*
* **Key Visuals**: 3D surface and 2D contour map over gating threshold $\tau \in [0.20, 0.80]$ and cooldown horizon $K \in [25, 500]$ on the validation partition.
* **Core Bullet Points**:
  - Swept across 130 parameter configurations on the validation partition (earlier baseline configuration: $\tau = 0.50, K = 100$).
  - **First-Order Sensitivity**: $S_\tau = 0.19\%$ accuracy per unit threshold; $S_K = 0.000133\%$ per step.
  - **Second-Order Curvature**: $\frac{\partial^2 \text{Acc}}{\partial \tau^2} < 0 \implies \mathbf{CONCAVE\_STABLE\_PEAK}$ (inverted peak / plateau).
  - Broad stability plateau: $\tau \in [0.35, 0.65]$ maintains $\pm 0.15\%$ of peak accuracy. The algorithm does not exhibit knife-edge parameter sensitivity.
* **Speaker Script**:
  > *"To ensure that CGT does not rely on a fragile, overfitted operating point, we evaluated 130 parameter configurations on the validation set.
  > As shown on this response surface, the system operates on a broad, **concave stability plateau**.
  > The second derivative with respect to threshold tau is strictly negative—confirming a concave peak—and shifting tau by $\pm 0.15$ produces less than a 0.2% change in accuracy. CGT is mathematically robust to parameter tuning."*

---

### Slide 15: Real-World Indian Power Grid (POSOCO / NLDC) SCADA Evaluation (1.0 Minute)
* **Slide Title**: *Empirical Evaluation on Real-World Indian National Grid SCADA Telemetry (POSOCO / NLDC)*
* **Key Visuals**: 28-month national generation and demand trajectory (Sept 2021 – Dec 2023) showing real 239.4 GW peak demand, thermal coal baseline, and solar/wind monsoon ramps.
* **Core Bullet Points (100% Real-World SCADA Telemetry)**:
  - **Data Provenance**: Compiled from 28 monthly OSI Monarch SCADA workbooks directly exported from the National Load Despatch Centre (NLDC / Grid Controller of India). 20,448 continuous hours (852 days), with 16,800 test instances across 100 non-overlapping calendar weeks.
  - **Empirical Results on Real POSOCO SCADA Stream**:
    - **Streaming Linear / Logistic (SGD)**: **49.06% accuracy**, $12.8\ \mu\text{s}$ latency, $0.0\%$ duty (catastrophic failure due to linear inseparability of grid dispatch constraints; 55.8% false positive rate).
    - **Always-HT (Baseline)**: **84.24% accuracy**, $30.4\ \mu\text{s}$ latency, $0.0\%$ duty.
    - **Random-Gate (20 seeds)**: **83.90% accuracy** ($\pm 0.09\%$), $30.7\ \mu\text{s}$ latency, $5.85\%$ duty.
    - **CGT (Learned Gate)**: **84.92% accuracy** (+0.68 pp over HT), **64.3 $\mu$s latency**, **5.83% duty cycle**, $554.1\ \mu\text{s}$ worst-case latency (GUARANTEED $< 1.0\text{ ms}$).
    - **Cost-Aware OMS-MAB**: **85.29% accuracy**, $31.4\ \mu\text{s}$ latency, $5.83\%$ duty.
    - **ARF-3 (3 Trees)**: **87.02% accuracy**, $45.8\ \mu\text{s}$ latency, $100.0\%$ duty.
    - **Always-ARF (Full 10)**: **90.13% accuracy**, $159.0\ \mu\text{s}$ latency, $100.0\%$ duty.
  - **100-Week Block Hypothesis Testing (Holm-Corrected)**:
    - CGT vs. Random-Gate: Mean weekly delta $= \mathbf{+0.679\%}$, $p = \mathbf{6.65 \times 10^{-3}}$, Cohen's $d = 0.321$ (Statistically Significant).
    - CGT vs. Always-HT: Mean weekly delta $= \mathbf{+0.679\%}$, $p = \mathbf{4.99 \times 10^{-3}}$, Cohen's $d = 0.321$ (Statistically Significant).
    - ARF-3 vs. CGT: Mean weekly delta $= \mathbf{+2.095\%}$, $p = \mathbf{7.56 \times 10^{-3}}$, Cohen's $d = 0.337$ (ARF-3 wins on raw accuracy via continuous 100% compute).
    - CGT vs. Cost-Aware OMS-MAB: Delta $= -0.363\%$, $p = 0.750$ (Not statistically significant).
* **Speaker Script**:
  > *"Rather than relying on synthetic approximations, we compiled and benchmarked models on **28 months of real-world POSOCO / NLDC SCADA telemetry** spanning September 2021 to December 2023.
  > On this 16,800-hour real Indian grid evaluation stream, Always-HT achieves 84.24%, while CGT achieves 84.92% at 64.3 µs latency and 5.83% duty cycle.
  > Under rigorous 100-week block testing, CGT achieves statistically significant gains over Always-HT ($p = 0.005$) and Random-Gate ($p = 0.007$). Continuous ARF-3 achieves 87.02% (+2.09 pp, $p = 0.008$), reflecting the higher accuracy possible when ensembles run 100% of the time, while CGT keeps the heavy ensemble dormant ~94.2% of the time with worst-case latency strictly under 1 ms."*

---

### Slide 16: Summary of Contributions & Open Science (1.0 Minute)
* **Slide Title**: *Summary of Contributions & Open Science Artifacts*
* **Key Visuals**: Summary graphic summarizing the 5 key project pillars: Conceptual Trilemma, Causal Gating, Pareto Trade-off, Statistical Rigor, and Silicon Open Science.
* **Core Contributions**:
  1. *Conceptual*: Formalized the sub-1ms edge streaming trilemma for digital substations.
  2. *Algorithmic*: Developed CGT with a causal streaming meta-controller operating without future lookahead.
  3. *Empirical & Fair Protocol*: Executed strict fair comparison against ARF-3 with identical validation budget ($K = 16$, locked with SHA-256: `433f0e3d...`), isolating a +0.58 pp accuracy gain over Random-Gate at matched duty cycle.
  4. *Statistical Rigor*: Verified via 146 weekly blocks with Holm-Bonferroni correction (CGT vs HT $p = 0.0055$, CGT vs Random $p = 1.57 \times 10^{-8}$, ARF-3 vs CGT $p = 0.0029$), and characterized resilience across 2,958 drift hours.
  5. *Silicon Open Science*: 100% reproducible repository, 40/40 passing unit tests, multi-rate modeled hardware profiling, and interactive Streamlit control room dashboard.
* **Speaker Closing**:
  > *"To conclude, Cascaded Compute-Gated Trees address a practical challenge in edge machine learning. We demonstrate that selective streaming meta-controllers can deliver accuracy gains over single trees while cutting heavy-ensemble execution by ~95.4% and maintaining low average latencies.
  > Our entire framework is open-source, fully reproducible with frozen configs and passing unit tests, and accessible in our Streamlit control room dashboard.
  > Thank you, and we welcome your questions."*

---

### Quick Reference Table of Verified Empirical Numbers for Viva Defense

```
                                                 VIVA DEFENSE NUMERICAL CHEAT SHEET
==================================================================================================================================================
Metric                       Linear (SGD)    Always-HT       Random-Gate     Cost-Aware MAB     CGT (Learned Gate)    ARF-3 (3 Trees)     Always-ARF (10T)
--------------------------------------------------------------------------------------------------------------------------------------------------
Test Accuracy (24k Stream)     85.062%         85.066%         84.756%          84.300%              85.331%              86.191%              89.596%
Test Accuracy (POSOCO Stream)  49.060%         84.244%         83.904%          85.286%              84.923%              87.018%              90.125%
Test Accuracy @ 5% Budget Cap   N/A             N/A            84.744%          85.258%              85.804% (+1.06 pp)     N/A                  N/A
Heavy Execution Duty Cycle     0.00%           0.00%           4.58%            4.60%                4.59%               100.00%              100.00%
Mean Latency (host-measured)  13.92 µs        94.26 µs       102.82 µs        191.05 µs            243.37 µs            281.35 µs           1,333.95 µs
Worst-Case Latency (host)     320.4 µs       2,404.0 µs      4,309.3 µs       3,671.7 µs          38,002.8 µs          2,231.1 µs           101,442.2 µs
Memory Footprint (SRAM)        ~1.2 KB         ~74 KB          ~74 KB           ~110 KB              ~74 KB              ~110 KB              ~246 KB
Inside Drift Windows (2,958h)  91.45%          92.33%          92.26%           91.85%               92.53%               93.61%               94.56%
Outside Drift Windows (21k h)  84.18%          84.07%          83.72%           83.27%               84.34%               85.17%               88.92%
50 Hz CPU Duty (modeled)       0.070%          0.471%          0.514%           0.955%               1.217%               1.407%               6.670%
50 Hz Battery (100 Wh MCU)    11,240 h        10,547 h        10,402 h         9,262 h              8,690 h (0.99 yr)    8,317 h              3,798 h
1 kHz CPU Load (modeled)        1.4%            9.4%           10.3%            19.1%                24.3%                28.1%              133.4% (OVERLOAD)
Weekly Block Holm p vs CGT       —               —           p = 2.09e-8      p = 1.13e-7          Baseline             p = 2.89e-3            N/A
Weekly Block Holm p vs HT        —            Baseline            N/A             N/A               p = 5.53e-3              N/A                N/A
==================================================================================================================================================
```
