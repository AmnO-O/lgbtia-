# Mathematical Formulation & Engineering Specification: Privileged Additive Latent Guidance

## Executive Summary & Theoretical Grounding

StereoQueerEval 2027 Task B (Hate Speech Classification) is characterized by severe pragmatic ambiguity between:
- Class 0: `no` (non-hate / benign / counterspeech)
- Class 1: `yes_implicit` (sarcasm, dog-whistles, moral lecturing, faux concern)
- Class 2: `yes_explicit` (overt slurs, violent threats)

Standard multi-class cross-entropy on frozen or fine-tuned representations struggles with `yes_implicit` because the surface lexical distribution overlaps heavily with `no` (e.g. polite vocabulary expressing exclusionary intent).

To resolve this without test-time overhead or shortcut leakage, we employ **Learning Using Privileged Information (LUPI)** via an **Additive Latent Residual Adapter** governed by **Cosine Curriculum Annealing** and **Unidirectional Consistency Distillation**.

---

## 1. Mathematical Architecture & Continuous Prototype Representation

Let $\mathbf{x} = (\text{Title}, \text{Description}, \text{Comment})$ denote the standard input text, and $\mathbf{x}^*$ denote the privileged sociolinguistic diagnostic analysis extracted offline by the teacher.

### 1.1 Dual-Stream Encoding & Additive Latent Injection

1. **Primary Representation Stream:**
   $$\mathbf{H}_x = \text{mmBERT}(\mathbf{x}) \in \mathbb{R}^{B \times S \times d}$$
   $$\mathbf{H}_{\text{role}} = \text{RMSNorm}\big(\mathbf{H}_x + \mathbf{E}_{\text{role}}\big)$$

2. **Privileged Guidance Stream:**
   $$\mathbf{h}^* = \text{Adapter}\Big( \text{mmBERT}(\mathbf{x}^*)_{\text{[CLS]}} \Big) \in \mathbb{R}^{B \times d}$$

3. **Additive Latent Fusion:**
   $$\mathbf{H}_{\text{fused}} = \mathbf{H}_{\text{role}} + \alpha(e) \cdot m \cdot \mathbf{h}^* \mathbf{1}^T$$
   - $m \sim \text{Bernoulli}(1 - p_{\text{drop}})$ is a stochastic Bernoulli mask.
   - $\alpha(e) \in [1.0, 0.0]$ is the **Cosine Curriculum Annealing Factor** at epoch $e$.
   - At inference / test time, $\alpha(e) = 0.0 \implies \mathbf{H}_{\text{fused}} \equiv \mathbf{H}_x$ (100% pure representation, zero test-time overhead, zero distributional shift).

---

## 2. Multi-Objective Loss Formulation

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, p_u) + \lambda_c(e) \cdot \mathcal{L}_{\text{task}}(y, p_c) + \lambda_{\text{cons}}(e) \cdot \mathcal{D}_{\text{KL}}\Big( \text{stop\_gradient}(p_c) \,\|\, p_u \Big)}$$

### 2.1 Component Breakdown:
1. **Class-Balanced Focal Task Loss ($\mathcal{L}_{\text{task}}$):**
   $$\mathcal{L}_{\text{task}}(y, p) = -\sum_{k=0}^{2} w_k \cdot y_k \cdot (1 - p_k)^\gamma \log(p_k)$$
   - Class weight vector $w = [1.0, 2.2, 1.8]$ counters class imbalance.
2. **Unidirectional Consistency Distillation with Stop-Gradient:**
   $$\mathcal{L}_{\text{cons}} = \mathcal{D}_{\text{KL}}\big( \text{sg}(p_c) \,\|\, p_u \big) = \sum_{k=0}^{2} \text{sg}(p_{c,k}) \log\left( \frac{\text{sg}(p_{c,k})}{p_{u,k}} \right)$$
   - Gradients flow *exclusively* into $p_u$, forcing the unguided network parameters to mirror the teacher's latent decision boundary.
3. **Dynamic Annealing Schedule ($\alpha(e), \lambda_c(e), \lambda_{\text{cons}}(e)$):**
   $$\alpha(e) = \frac{1}{2}\left(1 + \cos\left(\frac{e}{E_{\text{total}}}\pi\right)\right)$$
   $$\lambda_c(e) = 0.40 \cdot \alpha(e)$$
   $$\lambda_{\text{cons}}(e) = 0.30 \cdot (1 - \alpha(e))$$

---

## 3. Quality-Gated & Leak-Proof Diagnostic Extractor (`gen/llm_rationalize.py`)

### 3.1 Strict Leak-Proof Sanitizer (Regex Masking)
All occurrences of label tokens (`implicit`, `explicit`, `hate`, `non-hate`, `neutral`) in free text (`why`, `boundary`) are programmatically masked to `[MASKED]` before writing to disk.

### 3.2 5-Axis Discrete Decomposition & Judge Self-Verification
- **5 Axes:** `direct_hostility`, `indirect_subtext`, `context_dependence`, `counter_speech`, `target_reference`
- **Independent LLM Judge:** `stereotype`, `hate_speech`, `target_identities`, `target_scope`, `confidence`
- **Quality Control (QC Gate):**
  - High quality if `pred_hs == gold_hs` OR (`gold_hs == 'yes_implicit'` and `confidence >= 0.60`).
  - If high quality: compiles concise $\le 64$-token hint:
    ```text
    axes: indirect_subtext=..., context_dependence=... | why: <8 words> | flip: <8 words>
    ```
  - If rejected: safe fallback `hint = ""` (row safely trains on pure unguided baseline).

---

## 4. End-to-End Implementation State-of-Truth

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Quality-Gated Offline Diagnostic Extraction                             │
│ • Run `python gen/llm_rationalize.py --api gemini --model gemini-2.5-flash`      │
│ • Enforces 5-axis reasoning, independent LLM judge, gold QC gate, & [MASKED]    │
│ • Caches validated results to `LGBT/rationales.json`                             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Additive Latent Model & Dataset Pipeline                                │
│ • `pipeline/task_b_data.py`: Dual-stream tokenization (`hint_col='hint'`)       │
│ • `pipeline/models/task_b_class_aware.py`: Continuous Class Query Cross-Attn     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Consistency & Annealing Trainer Execution                               │
│ • `pipeline/task_b_trainer.py`: 2-Phase differential unfreezing + MSD + FGM      │
│ • `notebook/task_b_class_aware_train.ipynb`: Trains with class_weights=[1,2.2,1.8]│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Final Validation & Leaderboard Submission                               │
│ • Evaluate strictly in unguided mode (α = 0, 100% human ground truth)            │
│ • Generates SemEval submission TSV via Latent Bridge (Tasks A, B, C)             │
└──────────────────────────────────────────────────────────────────────────────────┘
```
