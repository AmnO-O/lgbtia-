# Architectural Proposal & Formal Plan: Hierarchical Conditional Head for Task B Hate Speech Classification

## 1. Executive Summary & Problem Formulation
In the SemEval StereoQueer Task B classification challenge, instances are classified into three mutually exclusive categories:
- **0: Non-Hate (`no`)**
- **1: Implicit Hate (`yes_implicit`)**
- **2: Explicit Hate (`yes_explicit`)**

### 1.1 The Probability Cannibalization Flaw in Flat 3-Class Argmax
In a standard flat 3-class classifier, logits $[z_0, z_1, z_2]$ undergo a single competition via 3-way Softmax:
$$P(y=c) = \frac{e^{z_c}}{\sum_{k=0}^2 e^{z_k}}$$

**Theoretical & Empirical Vulnerability:**
- Class 1 (`yes_implicit`) and Class 2 (`yes_explicit`) share the identical semantic super-category: $\text{Hate Speech} = 1$.
- In ambiguous comments (e.g. nuanced microaggressions or sarcasm requiring video description disambiguation), the model spreads probability mass across both hate classes.
- *Concrete Failure Scenario:* 
  $$\begin{aligned}
  P(\text{no}) &= 0.40 \\
  P(\text{yes\_implicit}) &= 0.35 \\
  P(\text{yes\_explicit}) &= 0.25
  \end{aligned}$$
  - The aggregate probability that the utterance constitutes Hate Speech is $P(\text{Hate}) = 0.35 + 0.25 = \mathbf{0.60}$.
  - Under $\operatorname{argmax}_{c \in \{0,1,2\}} P(y=c)$, the prediction is mistakenly assigned to **`no`** (0.40 > 0.35 and 0.40 > 0.25), even though the model is 60% confident the comment is toxic!

---

## 2. Mathematical Foundation of Hierarchical Conditional Heads

We reformulate the joint probability distribution using the **Law of Total Probability and the Chain Rule**:

$$\begin{aligned}
P(y = \text{no} \mid \mathbf{z}) &= 1 - P(\text{hate} \mid \mathbf{z}) \\
P(y = \text{yes\_implicit} \mid \mathbf{z}) &= P(\text{hate} \mid \mathbf{z}) \times P(\text{implicit} \mid \text{hate}, \mathbf{z}) \\
P(y = \text{yes\_explicit} \mid \mathbf{z}) &= P(\text{hate} \mid \mathbf{z}) \times P(\text{explicit} \mid \text{hate}, \mathbf{z})
\end{aligned}$$

where:
- $\mathbf{z} \in \mathbb{R}^{2d}$ is the joint multi-aspect embedding ($z = \text{RMSNorm}([h_T; h_C'])$).
- **Stage 1 (Binary Super-Class Gate):** Computes $p_{\text{hate}} = \sigma(f_{\text{binary}}(\mathbf{z})) \in (0, 1)$.
- **Stage 2 (Conditional Fine-Grained Head):** Computes $p_{\text{type}} = \operatorname{Softmax}(f_{\text{type}}(\mathbf{z})) \in \Delta^1$ (or a sigmoid scalar for binary binary implicit vs explicit).

### Mathematical Conservation of Probability:
$$P(\text{no}) + P(\text{yes\_implicit}) + P(\text{yes\_explicit}) = (1 - p_{\text{hate}}) + p_{\text{hate}} \cdot p_{\text{imp}} + p_{\text{hate}} \cdot (1 - p_{\text{imp}}) \equiv \mathbf{1.0}$$

---

## 3. Neural Architecture Design

```
                     z_fusion [B, 2*d = 1536]
                                |
            +-------------------+-------------------+
            |                                       |
    [Binary Super-Head]                     [Fine-Grained Sub-Head]
    Linear(1536 -> 384)                     Linear(1536 -> 384)
    GELU + RMSNorm                          GELU + RMSNorm
    MSD (Multi-Sample Dropout)              MSD (Multi-Sample Dropout)
    Linear(384 -> 1)                        Linear(384 -> 1)
            |                                       |
    logit_hate [B, 1]                       logit_type [B, 1]
            |                                       |
    p_hate = sigmoid(logit_hate)            p_imp = sigmoid(logit_type)
            \                                       /
             \------------------+------------------/
                                |
                   [Hierarchical Probability Fusion]
          P(no)       = 1 - p_hate
          P(implicit) = p_hate * p_imp
          P(explicit) = p_hate * (1 - p_imp)
                                |
                 Full Log-Probabilities / Probabilities [B, 3]
```

### 3.1 Parameter Isolation & Feature Specialization
- **Binary Super-Head:** Specializes in identifying slurs, hostility, targeted derogatory intent, and boundary conditions between safe speech and harmful speech.
- **Fine-Grained Sub-Head:** Specializes in cross-context linguistic markers (e.g. comparing comment semantic drift against title/description to distinguish veiled stereotyping vs direct aggression).

---

## 4. Multi-Task Conditional Objective Function

To train both stages stably without gradient contamination, we formulate the **Conditional Hierarchical Multi-Task Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{binary}}(y_{\text{hate}}, p_{\text{hate}}) + \lambda_{\text{type}} \cdot \mathbb{I}_{\{y_{\text{hate}} = 1\}} \cdot \mathcal{L}_{\text{fine}}(y_{\text{type}}, p_{\text{type}})$$

### 4.1 Loss Components:
1. **Binary Detection Loss ($\mathcal{L}_{\text{binary}}$):**
   - Computed on **100% of samples**.
   - Binary Focal Loss or Weighted Binary Cross Entropy:
     $$\mathcal{L}_{\text{binary}} = - \left( w_{\text{hate}} \cdot y_{\text{hate}} \log(p_{\text{hate}}) + w_{\text{no}} \cdot (1 - y_{\text{hate}}) \log(1 - p_{\text{hate}}) \right)$$
2. **Conditional Fine-Grained Loss ($\mathcal{L}_{\text{fine}}$):**
   - Masked strictly by $\mathbb{I}_{\{y_{\text{hate}} = 1\}}$: **Calculated ONLY on hateful instances**.
   - **Crucial Benefit:** Non-hate samples produce exactly **0 gradient** backpropagating through the fine-grained head, preventing non-hate noise from distorting the implicit/explicit decision boundary.
3. **Consistency Loss ($\mathcal{L}_{\text{joint}}$):**
   - Standard negative log-likelihood on the full 3-class compound probabilities $\mathbf{P} \in \mathbb{R}^{B \times 3}$:
     $$\mathcal{L}_{\text{joint}} = - \log P(y_{\text{true}} \mid \mathbf{z})$$

$$\mathcal{L} = \alpha \mathcal{L}_{\text{binary}} + \beta \left( \frac{1}{\sum y_{\text{hate}}} \sum_{i: y_i \ge 1} \mathcal{L}_{\text{fine}, i} \right) + \gamma \mathcal{L}_{\text{joint}}$$
*(Recommended hyperparameters: $\alpha = 0.5, \beta = 0.5, \gamma = 1.0$).*

---

## 5. Inference & Decision Threshold Calibration

During inference, we have two operational modes:

### Mode A: Full Compound Probability Argmax (Strict Calibration)
$$\hat{y} = \operatorname{argmax} \left[ 1 - p_{\text{hate}}, \quad p_{\text{hate}} \cdot p_{\text{imp}}, \quad p_{\text{hate}} \cdot (1 - p_{\text{imp}}) \right]$$

### Mode B: Threshold-Gated Hierarchical Routing (Tunable for F1-Optimization)
Given a threshold $\tau \in (0, 1)$ (default $\tau = 0.50$, tunable via validation search):
$$\hat{y} = \begin{cases}
0 \text{ (`no`)}, & \text{if } p_{\text{hate}} < \tau \\
1 \text{ (`yes_implicit`)}, & \text{if } p_{\text{hate}} \ge \tau \text{ and } p_{\text{imp}} \ge 0.50 \\
2 \text{ (`yes_explicit`)}, & \text{if } p_{\text{hate}} \ge \tau \text{ and } p_{\text{imp}} < 0.50
\end{cases}$$
*(This allows precision-recall tuning specifically targeting the hardest minority class `yes_implicit`).*

---

## 6. Implementation Roadmap & Verification Plan

| Phase | Milestone | File(s) | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Config & Constants | `pipeline/config.py` | Add `use_hierarchical_head: bool = True`, `hierarchical_threshold: float = 0.50`, loss weight coefficients. |
| **Phase 2** | Hierarchical Classifier Module | `pipeline/models/task_b_cross_context.py` | Implement `HierarchicalMSDClassifier` with dual-branch Multi-Sample Dropout (Super Head + Sub Head). |
| **Phase 3** | Loss Integration | `pipeline/losses.py` | Implement `HierarchicalTaskBLoss` with masked conditional BCE and joint NLL. |
| **Phase 4** | Trainer Adaptation | `pipeline/task_b_cross_trainer.py` | Update forward/loss calculation, batch metric evaluation, and threshold calibration search in `eval_epoch`. |
| **Phase 5** | Empirical Validation | Kaggle Notebook | Run 2-Phase training and verify Macro-F1 progression against baseline `0.5911`. |

---

## 7. Expected Empirical & Research Paper Impact

1. **Direct Macro-F1 Gains:** 
   - Directly elevates `Implicit F1` (previously 0.4518) by stopping probability leakage to `No-Hate`.
2. **Interpretability & Diagnostics (Paper Section 4.3):**
   - We can plot a 2D scatter space: $X$-axis = $P(\text{Hate})$, $Y$-axis = $P(\text{Implicit} \mid \text{Hate})$, demonstrating clear visual separation of queer-hate microaggressions.
3. **Publication Quality:**
   - Provides a novel, theoretically sound architectural contribution beyond standard text classification benchmarks.
