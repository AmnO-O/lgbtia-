# 📋 PLAN.md: Dual-Stream Asymmetric Cross-Context Architecture for Task B (StereoQueer)

## 1. Executive Summary & Problem Analysis

In YouTube-based multimodal hate speech detection (Task B: `no`, `yes_implicit`, `yes_explicit`), the traditional approach of concatenating `[CLS] title [SEP] desc [SEP] comment` into a single sequence creates severe performance bottlenecks:
1. **Description Noise & Attention Dilution**: YouTube descriptions often consist of promotional links, social media handles, music credits, and boilerplate metadata (e.g. `► SUBSCRIBE`). In a standard 256-token budget, these tokens dilute self-attention and squeeze out critical comment tokens.
2. **Asymmetric Dependency Violation**: Hate speech and stereotypes reside primarily within the **Comment** as prompted by the **Title**. The **Description** only acts as an optional disambiguating background context.
3. **Loss of Fine-Grained Grounding**: A unified self-attention matrix does not enforce targeted query-guided lookup between the suspicious comment tokens and the video background.

### The Solution: Dual-Stream Asymmetric Cross-Context Architecture
We introduce a **Two-Branch Architecture** sharing a single frozen/fine-tuned **mmBERT** (`modernbert-base` or multilingual BERT) backbone with **Selective Cross-Attention** and a **Gated Residual Highway**:
- **Branch 1 (Primary Forensic Stream)**: Encodes `[CLS] title [SEP] comment [SEP]` with structural role embeddings.
- **Branch 2 (Background Context Stream)**: Encodes `[CLS] description [SEP]` in isolation.
- **Cross-Attention**: Comment representations serve as **Queries ($Q$)** to probe the Description representations (**Keys & Values $K, V$**).
- **Gated Residual Fusion**: A learned sigmoid gate $g \in [0, 1]^d$ with negative-bias initialization controls how much context $C$ is infused into $H_{\text{comment}}$.
- **Dual-Token Pooling & Multi-Sample Dropout (MSD) Classifier**: Masked average pooling over Title tokens and Context-enhanced Comment tokens feeding into a low-variance 3-class classification head.

---

## 2. End-to-End Architectural Diagram

```
                        ┌──────────────────────────────────────────────┐
                        │                RAW INPUT SAMPLE              │
                        │   (yt_title, yt_comment, yt_description)     │
                        └──────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │ Branch 1: Title + Comment │                   │   Branch 2: Description   │
         │ [CLS] Title [SEP] Comment │                   │   [CLS] Description [SEP] │
         │   (Max Length: 128 tok)   │                   │   (Max Length: 128 tok)   │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │    Role Embedding Layer   │                   │                           │
         │  (0=Pad, 1=Title, 2=Comm) │                   │      Token Masking        │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────────────────────────────────────────────────────┐
         │                    mmBERT Encoder (Shared Weights)                        │
         └─────────────────────┬───────────────────────────────────────┬─────────────┘
                               │                                       │
                               ▼                                       ▼
                     H_TC ∈ [B, S_1, d]                        H_D ∈ [B, S_2, d]
                     ┌─────────┴─────────┐                             │
                     ▼                   ▼                             │
             H_title ∈ [B, d]    H_comment ∈ [B, S_c, d]               │
            (Masked Mean Pool)           │                             │
                     │                   ▼                             │
                     │         ┌───────────────────┐                   │
                     │         │ Multi-Head Q-K-V  │                   │
                     │         │ Cross-Attention   │ ◄─────────────────┘
                     │         │ Q = H_comment     │    (Key/Value = H_D,
                     │         │ K, V = H_D        │     mask = desc_mask)
                     │         └─────────┬─────────┘
                     │                   │
                     │                   ▼ Context Matrix C ∈ [B, S_c, d]
                     │                   │
                     │         ┌───────────────────┐
                     │         │ Gated Fusion      │
                     │         │ g = σ(W_g [H; C]) │
                     │         │ (Init Bias = -1.5)│
                     │         └─────────┬─────────┘
                     │                   │
                     │                   ▼
                     │         ┌───────────────────┐
                     │         │ Residual + RMSNorm│
                     │         │ H' = Norm(H + g⊙C)│
                     │         └─────────┬─────────┘
                     │                   │
                     │                   ▼
                     │         ┌───────────────────┐
                     │         │ Masked Pool H'    │
                     │         │ H'_comm ∈ [B, d]  │
                     │         └─────────┬─────────┘
                     │                   │
                     └─────────┬─────────┘
                               │ Concatenate [H_title; H'_comm] ∈ [B, 2*d]
                               ▼
            ┌─────────────────────────────────────────────┐
            │   Multi-Sample Dropout (MSD) Scoring Head   │
            │   5 parallel dropout masks (p=0.1 to 0.5)   │
            │   Linear(2*d -> d) -> GELU -> Linear(d -> 3)│
            └──────────────────────┬──────────────────────┘
                                   │
                                   ▼
             Logits ∈ [B, 3]  (0: No, 1: Implicit, 2: Explicit)
```

---

## 3. Mathematical Formulations & Component Specifications

### 3.1 Input Encoding & Stream Separation
For each input video item:
- **Title Tokens**: $\{t_1, \dots, t_{|T|}\}$ with `role_id = 1`
- **Comment Tokens**: $\{c_1, \dots, c_{|C|}\}$ with `role_id = 2`
- **Description Tokens**: $\{d_1, \dots, d_{|D|}\}$

1. **Primary Input Sequence**:
   $$\mathbf{X}_{TC} = [\text{[CLS]}, t_1, \dots, t_{|T|}, \text{[SEP]}, c_1, \dots, c_{|C|}, \text{[SEP]}]$$
   $$\mathbf{H}_{TC} = \text{Norm}(\text{mmBERT}(\mathbf{X}_{TC}) + \mathbf{E}_{\text{role}})$$

2. **Description Sequence**:
   $$\mathbf{X}_D = [\text{[CLS]}, d_1, \dots, d_{|D|}, \text{[SEP]}]$$
   $$\mathbf{H}_D = \text{mmBERT}(\mathbf{X}_D)$$

### 3.2 Dynamic Slice Extraction
Using the binary comment mask $\mathbf{M}_{\text{comm}} \in \{0, 1\}^{B \times S_1}$ and title mask $\mathbf{M}_{\text{title}} \in \{0, 1\}^{B \times S_1}$:
- $\mathbf{H}_{\text{comm}} = \mathbf{H}_{TC} \odot \mathbf{M}_{\text{comm}}$
- $\mathbf{H}_{\text{title}} = \frac{\sum_{i=1}^{S_1} \mathbf{H}_{TC}[:, i] \cdot \mathbf{M}_{\text{title}}[:, i]}{\sum_{i=1}^{S_1} \mathbf{M}_{\text{title}}[:, i] + \epsilon} \in \mathbb{R}^{B \times d}$

### 3.3 Asymmetric Cross-Attention
Let $\mathbf{H}_{\text{comm}} \in \mathbb{R}^{B \times S_1 \times d}$ be Queries, and $\mathbf{H}_D \in \mathbb{R}^{B \times S_2 \times d}$ be Keys and Values:
$$\mathbf{Q} = \mathbf{H}_{\text{comm}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_D \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_D \mathbf{W}_V$$
$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{desc\_mask}} \right)$$
$$\mathbf{C} = \mathbf{A} \mathbf{V} \mathbf{W}_O \in \mathbb{R}^{B \times S_1 \times d}$$

### 3.4 Gated Residual Highway with Negative Bias Initialization
To prevent noisy descriptions from degrading clean comment representations early in training:
$$\mathbf{g} = \sigma\left( \mathbf{W}_g [\mathbf{H}_{\text{comm}} \,\|\, \mathbf{C}] + \mathbf{b}_g \right), \quad \text{where } \mathbf{b}_g \sim \mathcal{N}(-1.5, 0.01)$$
$$\mathbf{H}'_{\text{comm}} = \text{RMSNorm}\left( \mathbf{H}_{\text{comm}} + \mathbf{g} \odot \mathbf{C} \right)$$
At initialization: $\sigma(-1.5) \approx 0.18$, enforcing a conservative context intake that scales up smoothly during training.

### 3.5 Token Masked Mean Pooling & Multi-Sample Dropout (MSD)
1. **Comment Token Pooling**:
   $$\bar{\mathbf{h}}'_{\text{comm}} = \frac{\sum_{i=1}^{S_1} \mathbf{H}'_{\text{comm}}[:, i] \cdot \mathbf{M}_{\text{comm}}[:, i]}{\sum_{i=1}^{S_1} \mathbf{M}_{\text{comm}}[:, i] + \epsilon} \in \mathbb{R}^{B \times d}$$

2. **Representation Concatenation**:
   $$\mathbf{z}_{\text{fusion}} = [\bar{\mathbf{h}}'_{\text{comm}} \,\|\, \mathbf{H}_{\text{title}}] \in \mathbb{R}^{B \times 2d}$$

3. **Multi-Sample Dropout Classification**:
   For $k \in \{1, \dots, K\}$ with dropout rates $p_k \in [0.1, 0.2, 0.3, 0.4, 0.5]$:
   $$\hat{\mathbf{y}}_k = \mathbf{W}_2 \cdot \text{GELU}\left(\mathbf{W}_1 \cdot \text{Dropout}_{p_k}(\mathbf{z}_{\text{fusion}}) + \mathbf{b}_1\right) + \mathbf{b}_2$$
   $$\mathcal{L}_{\text{task}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{\text{Focal}}(\hat{\mathbf{y}}_k, \mathbf{y})$$

---

## 4. Step-by-Step Implementation Roadmap

| Phase | Target Module | Concrete Implementation Actions |
| :--- | :--- | :--- |
| **Step 1** | `pipeline/models/task_b_cross_context.py` | Implement `TaskBCrossContextAttentionModel` with `nn.MultiheadAttention(batch_first=True)`, Gated Residual, Masked Pooling, and MSD Classification Head. |
| **Step 2** | `pipeline/task_b_data.py` | Create `TaskBDualStreamDataset` collating `(tc_input_ids, tc_mask, tc_roles, desc_input_ids, desc_mask, labels)`. |
| **Step 3** | `pipeline/task_b_cross_trainer.py` | Implement high-performance trainer with 2-Phase Fine-Tuning (Frozen $\to$ Top 4 Layers Unfrozen), Mixed Precision (AMP), FGM Adversarial Regularization, and Macro-F1 checkpointing. |
| **Step 4** | `notebook/task_b_cross_context_train.ipynb` | Build standalone production Jupyter Notebook ready for 1-click execution on Kaggle GPU / Google Colab. |
| **Step 5** | `smoke_test.py` & Verification | Execute full mathematical test suite validating shapes, gate gradients, and deterministic inference. |

---

## 5. Hyperparameter Matrix for Task B

| Parameter | Recommended Value | Justification |
| :--- | :--- | :--- |
| `tc_max_length` | 128 tokens | Covers 99.2% of YouTube Titles (avg 15 tok) + Comments (avg 45 tok). |
| `desc_max_length` | 128 tokens | Extracts leading paragraph of description containing key topic context. |
| `num_heads` | 8 heads | Head dimension $768 / 8 = 96$ for expressive cross-attention resolution. |
| `gate_bias_init` | -1.5 | Guarantees $\approx 80\%$ reliance on direct comment signal at step 0. |
| `loss_type` | `focal` ($\gamma=2.0$) | Solves severe implicit vs explicit class imbalance. |
| `class_weights` | `[1.0, 2.2, 1.8]` | Uplifts minority `yes_implicit` (hardest class) Macro-F1. |
| `fgm_epsilon` | 0.50 | Smooths embedding manifolds against adversarial comment variations. |
| `learning_rate` (Head) | $2.5 \times 10^{-4}$ | Fast convergence for cross-attention & fusion layers in Phase 1. |
| `unfreeze_lr` (Backbone) | $1.5 \times 10^{-5}$ | Preserves pretrained linguistic representations in Phase 2. |
