# NeuroTrace: System Architecture & Mathematical Foundations

## 1. Architectural Overview

NeuroTrace is an interpretable, multi-modal machine learning platform designed for early risk assessment and screening of neurodegenerative disorders: **Parkinson's Disease (PD)** and **Alzheimer's Disease (AD)**.

The system is decoupled into four modular layers:
1. **Data Ingestion & Integrity Layer**: Automated acquisition from the UCI Machine Learning Repository and Washington University OASIS cohorts with checksum verification and clinical range assertions.
2. **Feature Engineering & Preprocessing Layer**: Outlier-resilient scaling (`RobustScaler`), education-stratified median imputation, and subject-level grouping to prevent data leakage.
3. **Multi-Model Ensembling & Calibration Layer**: Soft-voting synthesis of complementary tree-based estimators (Random Forest, Gradient Boosting, Extra Trees) and margin classifiers (Support Vector Machines).
4. **Clinical Decision Support & Explainability Layer**: Local additive attribution (counterfactual reference deviations) and a production REST API / interactive diagnostic UI.

```
+---------------------------------------------------------------------------------------------------------+
|                                      NEUROTRACE SYSTEM ARCHITECTURE                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ Ingestion & Validation ]                [ Preprocessing & Scaling ]          [ Ensembled Models ]   |
|   UCI Voice (22 Biomarkers)   ------>   RobustScaler (Tremor Defense)   ------>  Random Forest (100)    |
|   OASIS MRI (Demographics)    ------>   Group-Stratified Imputation     ------>  Gradient Boosting      |
|   Clinical Range Assertions                                             ------>  Calibrated SVC (RBF)   |
|                                                                         ------>  Extra Trees Classifier |
|                                                                                             |           |
|                                                                                             v           |
|   [ Decision Support & UX ]                 [ Explainability (XAI) ]             [ Soft-Voting Fusion ] |
|   Flask Clinical Dashboard    <------   Additive Feature Attribution    <------  Posterior Probability  |
|   REST API v1 (/predict)      <------   Population Counterfactuals               Stratified Thresholds  |
|   Print Diagnostic Report                                                                               |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulations

### 2.1 Vocal Acoustic Biomarkers (Parkinson's Dysphonia)

#### Jitter (Period Perturbation)
Jitter quantifies cycle-to-cycle variability in fundamental frequency ($T_0$):
$$\text{Jitter (Absolute)} = \frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|$$
$$\text{Jitter (\%)} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^N T_i} \times 100$$

#### Shimmer (Amplitude Perturbation)
Shimmer measures cycle-to-cycle variability in vocal waveform amplitude ($A_i$):
$$\text{Shimmer (dB)} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left| 20 \log_{10}\left(\frac{A_{i+1}}{A_i}\right) \right|$$

#### Harmonics-to-Noise Ratio (HNR)
HNR evaluates the proportion of periodic tonal energy relative to turbulent glottal noise:
$$\text{HNR (dB)} = 10 \log_{10}\left( \frac{\mathcal{E}_{\text{harmonic}}}{\mathcal{E}_{\text{noise}}} \right)$$

#### Pitch Period Entropy (PPE)
PPE captures the entropy of pitch variations on a logarithmic semitone scale, remaining robust against natural voluntary intonation while isolating involuntary parkinsonian tremors:
$$r_i = 12 \log_2 \left(\frac{F_{0, i}}{\bar{F}_0}\right)$$
$$\text{PPE} = - \sum_{k} p(r_k) \log_2 p(r_k)$$

---

### 2.2 Structural MRI Volumetrics (Alzheimer's Atrophy)

#### Normalized Whole Brain Volume (nWBV)
Ratio of total brain parenchymal volume (gray matter + white matter) relative to estimated intracranial capacity:
$$\text{nWBV} = \frac{\text{Total Brain Tissue Volume}}{\text{Estimated Total Intracranial Volume (eTIV)}}$$
Healthy young adults exhibit $\text{nWBV} \approx 0.80 - 0.85$, whereas advanced Alzheimer's dementia typically results in $\text{nWBV} < 0.70$ due to cortical and hippocampal loss.

#### Atlas Scaling Factor (ASF)
Determinant of the affine transformation matrix registering individual MRIs into Talairach stereotaxic space:
$$\text{ASF} = \det(\mathbf{M}_{\text{affine}})$$

---

### 2.3 Soft-Voting Probability Fusion

Given $M$ independent models with parameters $\theta_m$, the ensemble posterior probability for disease positive ($y=1$) is given by the weighted average:
$$P(y=1 \mid \mathbf{x}) = \sum_{m=1}^M w_m P_m(y=1 \mid \mathbf{x}; \theta_m), \quad \sum_{m=1}^M w_m = 1$$

Predicted diagnosis is determined via clinical risk stratification thresholds:
$$\hat{y} = \begin{cases} 
\text{High Risk}, & P(y=1 \mid \mathbf{x}) \ge 0.70 \\ 
\text{Moderate Risk}, & 0.35 \le P(y=1 \mid \mathbf{x}) < 0.70 \\ 
\text{Low Risk / Normal}, & P(y=1 \mid \mathbf{x}) < 0.35 
\end{cases}$$

---

### 2.4 Explainable AI (XAI) Local Attribution

For patient vector $\mathbf{x} = [x_1, \dots, x_D]$ and reference population background distribution $\mathcal{B}$:
$$\phi_j(\mathbf{x}) = P(y=1 \mid \mathbf{x}) - P(y=1 \mid \mathbf{x}_{\setminus j}, \tilde{x}_j)$$
where $\tilde{x}_j = \text{median}_{b \in \mathcal{B}}(b_j)$ represents the expected normative baseline value.
Features with $\phi_j > 0$ are classified as **Risk Escalators**, while features with $\phi_j < 0$ represent **Protective Clinical Factors**.
