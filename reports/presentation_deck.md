# Final Evaluation Presentation Script (5–7 Minutes)
## Project: Meta-Controller for Resource-Efficient Data Stream Mining

---

### Slide 1: Title & The Core Dilemma (1 Minute)
* **Title**: Meta-Controller for Resource-Efficient Data Stream Mining under Concept Drift
* **Team**: Rajan Patil, Prince Rajan, Ankit Ekka
* **Speaker Script**:
  > *"Good morning respected evaluators. In real-time cyber-physical systems—such as national power grids and SCADA IoT sensors—machine learning models face a fundamental dilemma:  
  > If we deploy heavy ensembles like Adaptive Random Forests, they are resilient to concept drift, but they burn continuous CPU power 24/7, even when the grid is completely quiet. If we deploy lightweight models like Hoeffding Trees, they run fast, but their accuracy collapses when non-linear concept drift hits.  
  > Our project solves this dilemma: we introduce **Conditional Gated Training (CGT)**, an intelligent meta-controller that keeps heavy models dormant 88% of the time, cutting compute latency by over 82% while adapting to real-time concept drift."*

---

### Slide 2: Proposed Methodology — Conditional Gated Training (1.5 Minutes)
* **Key Visuals**: Architectural diagram of ADWIN Interrupt $\rightarrow$ Gate Opening $\rightarrow$ Warm-Start Buffer Replay.
* **Speaker Script**:
  > *"Instead of running all models or using Multi-Armed Bandits that ignore hardware latency, CGT implements a conditional hardware interrupt:  
  > 1. An always-on Hoeffding Tree predicts and learns on every incoming stream instance.  
  > 2. An ADWIN drift detector monitors the prequential prediction error in real time using rigorous Hoeffding concentration bounds.  
  > 3. When ADWIN detects a statistically significant concept drift, an interrupt fires: the gate opens, and the dormant Adaptive Random Forest is awakened.  
  > 4. We execute an **S1 Warm-Start Replay**, feeding the last $B=300$ buffered instances to re-synchronize the ensemble before it takes over predictions.  
  > 5. Once the stream stabilizes for $K=1000$ consecutive alarm-free steps, the gate safely re-locks to save compute."*

---

### Slide 3: Live Interactive Demo (2 Minutes)
* **Action**: Switch screen to Streamlit app (`.venv\python.exe -m streamlit run app/dashboard.py`).
* **Speaker Script**:
  > *"Here is our live streaming simulator running on 35,064 hourly records.  
  > Notice that right now, the gate is green—meaning ARF is completely dormant, and our system is operating at only $90\ \mu\text{s}$ per instance.  
  > Now, let us simulate a sudden solar ramping shock using our drift injection slider. Notice how ADWIN instantly detects the drift, the red alarm flashes, and the gate opens. The heavy ensemble awakens via warm-start replay, stabilizes the error, and re-locks automatically once the grid recovers.  
  > Our cumulative CPU savings counter shows an **82.15% reduction in processor time** compared to Always-ARF."*

---

### Slide 4: Empirical Benchmark & Hypothesis Testing (1 Minute)
* **Key Visuals**: Benchmark comparison table and Pareto Frontier Curve (`results/figures/fig3_pareto_accuracy_cost.png`).
* **Speaker Script**:
  > *"We evaluated our system on a 70% frozen test stream of 24,528 continuous instances.  
  > - Our resource hypothesis ($\kappa \ge 30\%$) was **super-exceeded by 2.7x**: CGT reduced per-instance latency from $1303.9\ \mu\text{s}$ down to $232.8\ \mu\text{s}$, delivering **82.15% CPU savings**.  
  > - A Wilcoxon signed-rank test confirms that CGT's accuracy gains over Hoeffding Tree are statistically significant ($p = 1.11 \times 10^{-3}$).  
  > - On the Pareto frontier, CGT-Plus achieves the single best efficiency trade-off across all 11 evaluated policies."*

---

### Slide 5: Industrial ROI & Green AI Impact (30 Seconds)
* **Key Visuals**: Sustainability metrics table (`results/tables/roi_and_sustainability.csv`).
* **Speaker Script**:
  > *"In an industrial edge deployment of 1,000 SCADA nodes, this 82% duty-cycle reduction saves over 2,600 CPU hours annually, reduces processor thermal throttling, extends battery life, and prevents direct kilowatt-hours of carbon emissions.  
  > Thank you, and we welcome your questions."*

