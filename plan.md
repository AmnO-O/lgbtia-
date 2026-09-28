# Architectural Migration Plan: 4-Expert Dynamic Query-Bank MoE with Context-Aware Router

## Executive Summary
This document provides the complete, mathematically rigorous engineering specification and implementation roadmap for migrating the Task B model to a **4-Expert Mixture of Latent Query Banks (MoE-Query)** with a **Context-Aware Masked Mean Router**.

---

## 1. High-Level Architecture Diagram

```
                                      Input Tokens + Role IDs
                                                │
                                                ▼
                                    mmBERT Backbone + Role Embeddings
                                                │
                                    H_role ∈ ℝ^[B, S, 768]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    [Nhánh 1: Context Router]                                    [Nhánh 2: 4-Expert Query Banks]
  1. Masked Mean Pooling:                                       Parallel Pre-Norm Multihead Cross-Attention:
     h_ctx = Σ(m_i · H_i) / Σ(m_i)  [B, 768]                     - Expert 1 (Non-Hate Bank)      ──► z_1 [B, 768]
  2. Router MLP (768 ─► 256 ─► 4):                               - Expert 2 (Implicit Hate Bank) ──► z_2 [B, 768]
     logits_router ∈ ℝ^[B, 4]                                   - Expert 3 (Explicit Hate Bank) ──► z_3 [B, 768]
  3. Temperature Softmax Gating:                                - Expert 4 (Context Shift Bank) ──► z_4 [B, 768]
     g = Softmax(logits_router / τ) ∈ ℝ^[B, 4]                                 │
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                               Differentiable Convex Combination:
                                   z_final = Σ (g_i · z_i) ∈ ℝ^[B, 768]
                                                │
                                                ▼
                                [ Task B Classifier + MSD Head ]
                                 - Dense (768 ─► 384) + RMSNorm + GELU
                                 - 5-Branch Multi-Sample Dropout (p=0.1~0.3)
                                 - Linear (384 ─► 3) ──► Logits [B, 3]
                                                │
                                                ▼
                                   Task C Latent Bridge h_B
```

---

## 2. Mathematical Formalization

### 2.1. Nhánh 1: Context-Aware Router
1. **Masked Mean Pooling**:
   $$\mathbf{h}_{\text{ctx}} = \frac{\sum_{j=1}^S m_j \cdot \mathbf{H}_{\text{role}, j}}{\sum_{j=1}^S m_j + \epsilon} \in \mathbb{R}^{B \times 768}$$
   *where $m_j \in \{0, 1\}$ is the `attention_mask`.*

2. **Router Gating MLP**:
   $$\mathbf{h}_{\text{router}} = \text{GELU}(\text{RMSNorm}(W_1 \mathbf{h}_{\text{ctx}} + b_1)) \in \mathbb{R}^{B \times d_r} \quad (d_r = 256)$$
   $$\mathbf{u} = W_2 \mathbf{h}_{\text{router}} + b_2 \in \mathbb{R}^{B \times 4}$$
   $$\mathbf{g} = \text{Softmax}\left(\frac{\mathbf{u}}{\tau}\right) \in \mathbb{R}^{B \times 4} \quad \text{where } \sum_{i=1}^4 g_i = 1$$

### 2.2. Nhánh 2: 4-Expert Latent Query Banks
Four specialized continuous latent query prototypes $\mathcal{Q} \in \mathbb{R}^{4 \times K \times 768}$ ($K=1$ slot/expert default):
* **Expert 1 ($\mathcal{Q}_0$)**: *Non-Hate / Respectful / Affirmative* prototype.
* **Expert 2 ($\mathcal{Q}_1$)**: *Implicit / Nuanced Sarcasm / Subtle Stereotypes* prototype.
* **Expert 3 ($\mathcal{Q}_2$)**: *Explicit / Direct Slurs / Targeted Hate* prototype.
* **Expert 4 ($\mathcal{Q}_3$)**: *Contextual Discrepancy (Title/Desc vs Comment)* prototype.

For each expert $i \in \{0, 1, 2, 3\}$:
$$\mathbf{Q}_i = \text{RMSNorm}(\mathcal{Q}_i) \in \mathbb{R}^{B \times K \times 768}$$
$$\mathbf{K} = \mathbf{V} = \text{RMSNorm}(\mathbf{H}_{\text{role}}) \in \mathbb{R}^{B \times S \times 768}$$
$$\tilde{\mathbf{z}}_i = \text{MultiheadAttention}(\mathbf{Q}_i, \mathbf{K}, \mathbf{V}, \text{key\_padding\_mask}=\mathbf{m}^c) \in \mathbb{R}^{B \times K \times 768}$$
$$\mathbf{z}_i = \text{Mean}_{\text{slots}}(\mathcal{Q}_i + \text{Dropout}(\tilde{\mathbf{z}}_i)) \in \mathbb{R}^{B \times 768}$$

### 2.3. Convex Combination & Classification
$$\mathbf{z}_{\text{final}} = \sum_{i=0}^3 g_i \cdot \mathbf{z}_i \in \mathbb{R}^{B \times 768}$$
$$\mathbf{h}_{\text{dense}} = \text{GELU}(\text{RMSNorm}(W_h \mathbf{z}_{\text{final}} + b_h)) \in \mathbb{R}^{B \times 384}$$
$$\hat{\mathbf{y}} = \frac{1}{M} \sum_{m=1}^M W_{\text{out}} \cdot \text{Dropout}_m(\mathbf{h}_{\text{dense}}) \in \mathbb{R}^{B \times 3}$$

---

## 3. Loss Functions & Anti-Collapse Regularization

To ensure that the router does not collapse to a single dominant expert (a common MoE hazard), we use a composite objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TaskB}}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}}$$

### 3.1. Primary Classification Loss ($\mathcal{L}_{\text{TaskB}}$)
* **Focal Loss with Label Smoothing** ($\gamma=1.5, \epsilon=0.05$) to handle severe class imbalance and prevent overconfidence on explicit hate.

### 3.2. Router Load Balancing Regularizer ($\mathcal{L}_{\text{balance}}$)
Computes the coefficient of variation / entropy over the batch average routing probabilities $\bar{g}_i = \frac{1}{B} \sum_{b=1}^B g_{b, i}$:
$$\mathcal{L}_{\text{balance}} = 4 \sum_{i=0}^3 \bar{g}_i^2 - 1.0$$
* Minimizing this forces the mean gate load across the batch to be uniform ($\bar{g}_i \approx 0.25$), preventing dead experts while allowing individual sample sparsity.
* Hyperparameter: $\lambda_{\text{balance}} = 0.01$.

---

## 4. Implementation Steps & Code Changes

### Phase 1: Model Architecture (`pipeline/models/task_b_class_aware.py`)
1. Implement `masked_mean_pooling(h, mask)`.
2. Implement `ContextRouter(d_model=768, num_experts=4, hidden_dim=256, temp=1.0)`.
3. Implement `ParallelQueryBankCrossAttention(d_model=768, num_experts=4, num_heads=8)`.
4. Refactor `TaskBClassAwareAttentionModel` to assemble the router, 4 expert banks, weighted blending, MSD, and expose routing gates `g` for interpretability.

### Phase 2: Loss Function (`pipeline/losses.py`)
1. Add `MoELoadBalanceLoss(num_experts=4)`.
2. Update `MultiTaskLoss` / `TaskBLoss` to incorporate `load_balance_loss` when router gates are returned.
3. Clamp probabilities to ensure numerical stability under AMP FP16.

### Phase 3: Configuration & Hyperparameters (`pipeline/config.py`)
1. Add `num_experts: int = 4`.
2. Add `router_hidden_dim: int = 256`.
3. Add `router_temperature: float = 1.0`.
4. Add `loss_balance_weight: float = 0.01`.

### Phase 4: Trainer & Visualizations (`pipeline/task_b_trainer.py` & Notebooks)
1. Record average expert utilization per epoch: $\bar{g}_0, \bar{g}_1, \bar{g}_2, \bar{g}_3$.
2. Export gate distributions alongside predictions in `task_b_val_predictions.csv` for post-hoc analysis.
3. Plot expert routing heatmaps across classes (No-Hate vs Implicit vs Explicit).

### Phase 5: Verification & Testing
1. Run `smoke_test.py` with mock batch forward/backward pass.
2. Verify gradient flow to all 4 expert query tensors and router weights.
3. Confirm memory footprint and training throughput on single GPU.
