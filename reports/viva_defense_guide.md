# Viva Voce & Examiner Defense Cheat Sheet
## Anticipating Tough Questions for Final Project Evaluation

---

### Q1: "Why not simply use Linear Regression or XGBoost for this dataset?"
* **Answer**:  
  *"Linear regression assumes a stationary joint distribution $P(Y \mid X)$ and fails permanently when real-world concept drift occurs. Standard XGBoost is an offline batch model requiring the entire dataset in memory upfront, violating the constant-memory streaming constraint. In cyber-physical streaming systems, each instance arrives in a time-ordered sequence and must be processed under the **Prequential (test-then-train)** protocol in milliseconds."*

---

### Q2: "How does ADWIN guarantee that its drift detections are not false alarms?"
* **Answer**:  
  *"ADWIN (Bifet & Gavaldà, 2007) maintains variable-length sliding windows $W$ and compares sub-windows $W_0$ and $W_1$. It computes Hoeffding concentration bounds:  
  $$\epsilon_{cut} = \sqrt{\frac{1}{2m} \ln\left(\frac{4|W|}{\delta}\right)}$$  
  where $m$ is the harmonic mean of the sub-window lengths and $\delta$ is the user-defined confidence bound. Under the null hypothesis of stationarity, the probability of false drift detection is bounded strictly below $\delta$. In our frozen setup, $\delta=0.02$, mathematically guaranteeing a false alarm rate below 2%."*

---

### Q3: "Always-ARF achieved 89.60% accuracy, while your CGT achieved 85.21%. Why should a grid operator accept this 4.39% accuracy loss?"
* **Answer**:  
  *"In real-time edge environments (such as remote SCADA terminal units or solar inverters), running Always-ARF requires $1,304\ \mu\text{s}$ per inference, consuming 5.6x more CPU cycles, causing processor overheating, and draining battery backup.  
  CGT cuts latency to $232.8\ \mu\text{s}$—delivering an **82.15% compute saving**—while maintaining high 85.2% operational awareness. On the Pareto frontier, CGT provides the optimal trade-off between predictive reliability and physical hardware feasibility."*

---

### Q4: "What is the difference between your warm-start strategies S1, S2, and S3?"
* **Answer**:  
  *"When the gate opens upon an ADWIN alarm:  
  - **S1 (Our Champion)** resumes ARF's frozen memory and replays a FIFO buffer of the last $B=300$ instances to bring its split statistics up to date with the recent drift.  
  - **S2** completely resets ARF and trains from scratch on the buffer, which causes catastrophic forgetting of historical patterns.  
  - **S3** resumes the frozen state with zero replay, which leads to slow adaptation because ARF begins learning only from current instances without recent context.  
  Empirically, S1 outperforms S2 and S3 across both accuracy and stability."*

---

### Q5: "How did you prevent data leakage in your lag features?"
* **Answer**:  
  *"Our stream loader (`ElectricityStreamLoader`) strictly constructs $x_t$ using only historical values $t-1, t-2, t-24$ and cyclical calendar coordinates. We validated this with an automated adversarial test in `tests/test_leakage.py`: we synthetically perturbed future demand at time $t$ to $9,999,999\ \text{MW}$ and mathematically verified that $x_t$ remained 100% invariant."*

---

### Q6: "Why did you use Wilcoxon signed-rank test instead of a standard t-test?"
* **Answer**:  
  *"Streaming prequential errors and sliding accuracy over non-stationary data streams violate the Gaussian normality assumption required by standard Student's t-tests. The Wilcoxon signed-rank test is a non-parametric test that assesses whether paired median differences are significant without assuming normal error distributions. Our test confirmed $p = 1.11 \times 10^{-3}$, proving the gains are statistically robust."*

