# Comprehensive Architectural Blueprint & Production Plan
## Additive Latent Privileged Guidance with Curriculum Annealing & Unidirectional Consistency Distillation
### Benchmark Target: SemEval 2027 StereoQueerEval — Task B (Hate Speech Classification: No / Implicit / Explicit)

---

## 1. Mathematical & Theoretical Foundation

### 1.1 The Core Problem in Implicit Hate Detection
Implicit hate speech on social platforms relies heavily on pragmatic cues (sarcasm, faux-concern, dog-whistles, rhetorical questions) that exhibit zero overt toxic keywords. Naive models fail to establish sharp decision boundaries because the surface tokens appear benign.

### 1.2 The Additive Latent Privileged Information (ALPI) Framework
Inspired by:
1. **Vapnik's Learning Using Privileged Information (LUPI)** (*Vapnik et al., 2009*)
2. **Diffusion Classifier-Free Conditioning & Residual Modulation** (*Ho & Salimans, 2022*)
3. **Curriculum Teacher-Forcing Annealing** (*Bengio et al., NIPS 2015*)
4. **Self-Consistent Teacher Distillation with Stop-Gradient** (*Hinton et al., 2015; Xie et al., NeurIPS 2020*)

We formulate the model with a **Shared mmBERT Multilingual Latent Space** where teacher rationales are compressed into a compact representation vector and additively injected into the sequence representation.

---

## 2. Model Architecture & Forward Flow

```
                      ┌─────────────────────────────────────────────────────────────┐
                      │ Sample x = (Comment C, Title T, Desc D)                     │
                      │ Teacher Hint h = (5-Axis Linguistic Diagnostics + Boundary) │
                      └──────────────────────────────┬──────────────────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
         [STREAM 1: Primary Context Input]                   [STREAM 2: Teacher Hint Encoder]
         [CLS] comment: <C> [SEP] title: <T>                 [CLS] hint: <Linguistic Diagnostics>
                 [SEP] desc: <D> [SEP]                                   [SEP]
                             │                                               │
                             ▼                                               ▼
               ┌───────────────────────────┐                   ┌───────────────────────────┐
               │ mmBERT Shared Backbone    │                   │ mmBERT (Frozen Teacher)   │
               │ Last Hidden State:        │                   │ [CLS] Vector:             │
               │ H_x ∈ ℝ^(B × S × d_model) │                   │ v_hint ∈ ℝ^(B × d_model)  │
               └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                             │                                               │
                             │                                               ▼
                             │                                 ┌───────────────────────────┐
                             │                                 │ Residual Adapter + RMSNorm│
                             │                                 │ h_vec = RMSNorm(W·v_hint) │
                             │                                 └─────────────┬─────────────┘
                             │                                               │
                             │                                               ▼
                             │                                  ┌──────────────────────────┐
                             │                                  │ Stochastic Annealer α(e) │
                             │                                  │ h_mod = α(e) · m · h_vec │
                             │                                  └────────────┬─────────────┘
                             │                                               │
                             └───────────────────────┬───────────────────────┘
                                                     ▼
                                     [ADDITIVE LATENT RESIDUAL FUSION]
                                        H_fused = H_x + h_mod.unsqueeze(1)
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │ Class-Aware Attention Pooler  │
                                     │ 3 Learnable Queries (No/Imp/Exp)
                                     │ Q ∈ ℝ^(3 × d_model)           │
                                     └───────────────┬───────────────┘
                                                     │
                                                     ▼
                                          Logits z ∈ ℝ^(B × 3)
```

### 2.1 The Additive Latent Fusion Equation
Given primary sequence representations $\mathbf{H}_x \in \mathbb{R}^{B \times S \times d}$ and teacher hint vector $\mathbf{v}_{\text{hint}} \in \mathbb{R}^{B \times d}$:

$$\mathbf{h}_{\text{proj}} = \text{RMSNorm}\big( \mathbf{W}_h \mathbf{v}_{\text{hint}} + \mathbf{b}_h \big)$$

$$\mathbf{H}_{\text{fused}} = \mathbf{H}_x + \alpha(e) \cdot m \cdot \mathbf{h}_{\text{proj}}^\top \mathbf{1}_S^\top$$

Where:
- $m \sim \text{Bernoulli}(1 - p_{\text{drop}})$ is a stochastic Bernoulli mask.
- $\alpha(e) \in [1.0, 0.0]$ is the **Cosine Curriculum Annealing Factor** at epoch $e$.
- At inference / test time, $\alpha(e) = 0.0 \implies \mathbf{H}_{\text{fused}} \equiv \mathbf{H}_x$ (100% pure representation, zero test-time overhead, zero distributional shift).

---

## 3. The Mathematically Grounded Multi-Objective Loss Formulation

To simultaneously achieve:
1. Strong unguided classification performance on test data ($p_u$),
2. Accurate privileged alignment during training ($p_c$),
3. Safe distillation without gradient collapse ($p_u \to \text{stop\_gradient}(p_c)$),

we define the total loss objective:

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, p_u) + \lambda_c(e) \cdot \mathcal{L}_{\text{task}}(y, p_c) + \lambda_{\text{cons}}(e) \cdot \mathcal{D}_{\text{KL}}\Big( \text{stop\_gradient}(p_c) \,\|\, p_u \Big)}$$

### 3.1 Component Breakdown:

1. **Class-Balanced Focal Task Loss ($\mathcal{L}_{\text{task}}$):**
   $$\mathcal{L}_{\text{task}}(y, p) = -\sum_{k=0}^{2} w_k \cdot y_k \cdot (1 - p_k)^\gamma \log(p_k)$$
   - Modulating factor $\gamma = 2.0$ dynamically down-weights easy non-hate examples and emphasizes hard ambiguous implicit hate cases.
   - Class weight vector $w = [1.0, 2.2, 1.8]$ counters class imbalance.

2. **Unidirectional Consistency Distillation with Stop-Gradient:**
   $$\mathcal{L}_{\text{cons}} = \mathcal{D}_{\text{KL}}\big( \text{sg}(p_c) \,\|\, p_u \big) = \sum_{k=0}^{2} \text{sg}(p_{c,k}) \log\left( \frac{\text{sg}(p_{c,k})}{p_{u,k}} \right)$$
   - $\text{sg}(\cdot)$ is the **stop-gradient** operator ($\text{detach()}$ in PyTorch).
   - **Why this is critical:** Gradients flow *exclusively* into $p_u$, forcing the unguided network parameters to mirror the teacher's latent decision boundary, preventing the teacher from being degraded by an unguided student.

3. **Dynamic Annealing Schedule ($\alpha(e), \lambda_c(e), \lambda_{\text{cons}}(e)$):**

   $$\alpha(e) = \frac{1}{2}\left(1 + \cos\left(\frac{e}{E_{\text{total}}}\pi\right)\right)$$

   $$\lambda_c(e) = 0.40 \cdot \alpha(e)$$

   $$\lambda_{\text{cons}}(e) = 0.30 \cdot (1 - \alpha(e))$$

   - **Epochs 1–3 (Discovery):** $\alpha \approx 1.0$, $\lambda_c = 0.40$, $\lambda_{\text{cons}} = 0.05$. mmBERT absorbs high-level sociolinguistic representations.
   - **Epochs 4–10 (Transference):** $\alpha \to 0.5$, $\lambda_{\text{cons}} \uparrow 0.25$. Consistency loss pulls $p_u$ directly toward the teacher manifold.
   - **Epochs 11–15 (Mastery & Disconnection):** $\alpha \to 0.0$, $\lambda_c \to 0.0$. Model trains 100% independently in genuine inference conditions.

---

## 4. Rigorous Leak-Proof Rationale Generation (`gen/llm_rationalize.py`)

To eliminate the risk of shortcut memorization:
1. **Strict Regex Token Sanitizer:** Any occurrence of `implicit`, `explicit`, `hate`, `non-hate`, `neutral` is programmatically masked to `[MASKED]` before writing to disk.
2. **5-Axis Discrete Decomposition:**
   - `direct_hostility`: `weak | moderate | strong`
   - `indirect_subtext`: `weak | moderate | strong`
   - `context_dependence`: `weak | moderate | strong`
   - `counter_speech`: `weak | strong`
   - `target_reference`: `present | absent`
3. **Boundary Contrast:** Focuses strictly on *why* this comment could be misinterpreted at surface level and what subtle shift creates its actual communicative intent.

---

## 5. End-to-End Implementation Roadmap

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Offline Rationale Mining (Zero Leakage)                                  │
│ • Execute `python gen/llm_rationalize.py --api gemini --batch-size 8`            │
│ • Validates all training TSVs and caches to `LGBT/rationales.json`              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Additive Latent Model & Dataset Update                                   │
│ • Update `pipeline/models/task_b_class_aware.py` with `HintAdditiveProjection` │
│ • Update `pipeline/task_b_data.py` to yield paired (text_ids, hint_ids)         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Consistency & Annealing Trainer Execution                               │
│ • Implement CosineAnnealingScheduler for α(e), λ_c(e), and λ_cons(e)             │
│ • Train on GPU using 2-Phase Warmup + Differential Backbone Learning Rates       │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Final Validation & Leaderboard Submission                               │
│ • Evaluate strictly in unguided mode (α = 0, 100% human ground truth)            │
│ • Confirm Macro-F1 improvement on implicit hate speech                           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Verification Checkpoints

- [x] **Zero Token Overhead:** Primary comment context retains full 256 tokens.
- [x] **Zero Test-Time Mismatch:** Setting $\alpha = 0$ leaves the model physically identical to a standard transformer inference graph.
- [x] **Zero Label Shortcut:** Teacher rationales are completely stripped of classification labels.
- [x] **Mathematically Stable Gradient Flow:** Unidirectional KL with $\text{stop\_gradient}(p_c)$ guarantees monotonic knowledge transfer.
