# 📑 RESEARCH & IMPLEMENTATION BLUEPRINT: PRETRAINED ATTENTION WEIGHT BORROWING (TASK B)
## Isolated Hypothesis Testing: Deep Pretrained Attention Projections as Evidence Retrieval Operators

---

## 🎯 1. CORE RESEARCH HYPOTHESIS & SCIENTIFIC FRAMING

### 1.1 The Exact Research Question & Precise Academic Formulation:
> *"Can trainable latent class queries learn useful evidence retrieval directions when interacting through frozen, deep pretrained ModernBERT attention projection matrices ($W_{QKV}, W_O$)?"*

### 1.2 Methodological Honesty: Precise Architectural Terminology
We strictly maintain academic precision and methodological transparency: **we do NOT claim to reproduce native ModernBERT self-attention**. 

Instead, we explicitly define our operation as:
> *"We transplant pretrained Q/K/V/O projections while applying asymmetric RoPE to the context keys."*

We document two intentional architectural divergences:
1. **Topology Replacement (Local/Global Self-Attn $\to$ Global Latent-to-Token Cross-Attn)**:
   - ModernBERT alternates between local sliding-window attention (window = 128) and global self-attention every 3 layers ($i \pmod 3 == 0$).
   - In our architecture:
     > *"We transplant the linear projection parameters while replacing the pretrained self-attention topology with global latent-to-token cross-attention."*
   - Methodological rationale: Class queries represent abstract global concepts that must maintain unconstrained, full-span visibility across the entire context sequence.
2. **Intentional Asymmetric RoPE ($Q = Z W_Q$ Unrotated, $K = \operatorname{RoPE}(H W_K)$ Rotated)**:
   - ModernBERT natively rotates both $Q$ and $K$ within its self-attention blocks (`apply_rotary_pos_emb(query_states, key_states, cos, sin)`).
   - In our architecture:
     - **Unrotated $Q$ ($Q = Z W_Q$)**: The $K$ query latents $Z \in \mathbb{R}^{B \times K \times D}$ represent abstract, position-free conceptual probes (`No Hate`, `Implicit Hate`, `Explicit Hate`). Imposing sequence-position RoPE onto $Z$ would introduce arbitrary, linguistically baseless relative-distance penalties among the classes.
     - **Rotated $K$ ($K = \operatorname{RoPE}(H W_K)$)**: The context tokens $H \in \mathbb{R}^{B \times L \times D}$ represent natural language text whose syntactic relationships depend critically on token order. Rotary embeddings on $K$ preserve the full linguistic geometry of the context.
   - **RoPE Module Resolution**: We extract the official `ModernBertRotaryEmbedding` from the parent `ModernBertModel` and wire it directly into the borrowed cross-attention layers.

---

## 🏗️ 2. DEDICATED MODULE: `PretrainedCrossAttentionLayer`

### 2.1 Code Implementation Supporting Clean Experimental Controls & Strict Asymmetric RoPE:
- **Experiment A (Pure Projection Transplantation)**: $Q = Z W_Q, \quad K = \operatorname{RoPE}(H W_K)$
- **Experiment B (Pretrained Pre-Normalization)**: $Q = \operatorname{LN}(Z) W_Q, \quad K = \operatorname{RoPE}(\operatorname{LN}(H) W_K)$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding exclusively to Keys (K) (Asymmetric RoPE).
    tensor: [B, num_heads, L, head_dim]
    cos, sin: [1, 1, L, head_dim] or [B, 1, L, head_dim]
    """
    if cos.dim() == 4 and cos.shape[1] == tensor.shape[2] and cos.shape[2] == 1:
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (tensor * cos) + (rotate_half(tensor) * sin)


class PretrainedCrossAttentionLayer(nn.Module):
    """
    Asymmetric Cross-Attention whose Q/K/V/O projections are borrowed
    directly from one pretrained ModernBERT attention block.
    
    Academic Framing:
      "We transplant pretrained Q/K/V/O projections while applying 
       asymmetric RoPE to the context keys."
    """
    def __init__(
        self,
        pretrained_attn: nn.Module,
        rotary_emb: Optional[nn.Module] = None,
        pretrained_norm: Optional[nn.Module] = None,
        use_prenorm: bool = False
    ):
        super().__init__()
        self.attn = pretrained_attn
        self.use_prenorm = use_prenorm
        
        self.Wqkv = pretrained_attn.Wqkv
        self.Wo = pretrained_attn.Wo
        self.norm = pretrained_norm if use_prenorm else None
        
        # Wire rotary embedding explicitly from ModernBertModel
        self.rotary_emb = rotary_emb or getattr(pretrained_attn, "rotary_emb", None)
        
        self.num_heads = pretrained_attn.config.num_attention_heads
        self.head_dim = pretrained_attn.head_dim
        self.scaling = self.head_dim ** -0.5

    def forward(self, z, h_context, context_mask=None, position_ids=None):
        B, K, D = z.shape
        _, L, _ = h_context.shape

        z_in = self.norm(z) if (self.use_prenorm and self.norm is not None) else z
        h_in = self.norm(h_context) if (self.use_prenorm and self.norm is not None) else h_context

        q_weight, k_weight, v_weight = self.Wqkv.weight.chunk(3, dim=0)
        q = F.linear(z_in, q_weight)
        k = F.linear(h_in, k_weight)
        v = F.linear(h_in, v_weight)

        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Intentional Asymmetric RoPE: Rotates only context keys (K), leaves queries (Q) unrotated
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=h_context.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, position_ids)
            k = apply_rotary_pos_emb_single(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if context_mask is not None:
            mask_4d = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -10000.0
            scores = scores + mask_4d

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)

        z_out = z + self.Wo(attn_out)
        return z_out, attn_weights.mean(dim=1)
```

---

## 🔒 3. SCIENTIFIC FREEZE ENFORCEMENT PROTOCOL

To strictly isolate the hypothesis:
$$\begin{aligned}
\text{Backbone Layers 1--17} &\implies \textbf{Frozen} \quad (\nabla_{\theta} = 0) \\
\text{Transplanted } W_Q, W_K, W_V, W_O \text{ (Layers 18--22)} &\implies \textbf{Frozen} \quad (\nabla_{W} = 0) \\
\text{Latent Class Probes } Z_0 = [q_N, q_I, q_E] &\implies \textbf{Trainable} \quad (\nabla_{Z} \neq 0) \\
\text{Hierarchical Classification Head } f_{\text{tree}} &\implies \textbf{Trainable} \quad (\nabla_{\phi} \neq 0)
\end{aligned}$$

This guarantees that performance gains arise exclusively from **trainable class queries discovering optimal evidence retrieval trajectories** through the frozen geometry of pretrained projection operators.

---

## 🔬 4. MATHEMATICAL SPECIFICATIONS OF EXPERIMENTS A & B

Given context tokens $X$, we compute $H_{17} = \operatorname{mmBERT}_{1:17}(X) \in \mathbb{R}^{B \times L \times D}$.
We initialize $K$ query latents $Z_0 = [q_N, q_I, q_E] \in \mathbb{R}^{B \times 3 \times D}$.

### Experiment A (Main: Pure Projection Transplantation - No Normalization):
For layers $l \in [18, 19, 20, 21, 22]$:
$$\begin{aligned}
Q_l &= Z_{l-1} \cdot W_Q^{(l)} \quad \text{(Position-Free, Unrotated)} \\
K_l &= \operatorname{RoPE}\left(H_{17} \cdot W_K^{(l)}, \text{pos}_{0:L}\right) \quad \text{(Asymmetric RoPE on Keys)} \\
V_l &= H_{17} \cdot W_V^{(l)} \\
A_l &= \operatorname{Softmax}\left(\frac{Q_l K_l^T}{\sqrt{64}} + M_{\text{non\_pad}}\right) \\
Z_l &= Z_{l-1} + W_{O, l}(A_l V_l)
\end{aligned}$$

### Experiment B (Critical Control: Pretrained Pre-Normalization):
For layers $l \in [18, 19, 20, 21, 22]$:
$$\begin{aligned}
Q_l &= \operatorname{LN}(Z_{l-1}) \cdot W_Q^{(l)} \quad \text{(Position-Free, Unrotated)} \\
K_l &= \operatorname{RoPE}\left(\operatorname{LN}(H_{17}) \cdot W_K^{(l)}, \text{pos}_{0:L}\right) \quad \text{(Asymmetric RoPE on Keys)} \\
V_l &= \operatorname{LN}(H_{17}) \cdot W_V^{(l)} \\
A_l &= \operatorname{Softmax}\left(\frac{Q_l K_l^T}{\sqrt{64}} + M_{\text{non\_pad}}\right) \\
Z_l &= Z_{l-1} + W_{O, l}(A_l V_l)
\end{aligned}$$

Final representations $Z_5 = [z_N, z_I, z_E]$ feed directly into the unchanged **Hierarchical Head**:
$$\begin{aligned}
p_{\text{hate}} &= \sigma(f_{\text{hate}}(z_N, z_I, z_E)) \\
p_{\text{implicit}} &= \sigma(f_{\text{fine}}(z_I, z_E)) \\
P_{\text{no}} &= 1 - p_{\text{hate}} \\
P_{\text{implicit}} &= p_{\text{hate}} \cdot p_{\text{implicit}} \\
P_{\text{explicit}} &= p_{\text{hate}} \cdot (1 - p_{\text{implicit}})
\end{aligned}$$

---

## 📈 5. STRUCTURED ABLATION MATRIX

| Model Variant | Formulation | Projections Status | RoPE Handling | Pretrained FFN | Classification Head |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A (Main Core)** | Pure Projections Borrowing | 🔒 **Frozen** | Asymmetric RoPE ($Q$ unrotated, $K$ rotated) | ❌ None | Hierarchical Tree |
| **Exp B (Control)** | Pre-Normalized Borrowing | 🔒 **Frozen** | Asymmetric RoPE ($Q$ unrotated, $K$ rotated) | ❌ None | Hierarchical Tree |
| **Ablation 1 (Init)** | Pure Projections + Semantic Init | 🔒 **Frozen** | Asymmetric RoPE ($Q$ unrotated, $K$ rotated) | ❌ None | Hierarchical Tree |
| **Ablation 2 (Unfrozen)**| Fine-tuned Projections Control | 🔓 **Trainable** | Asymmetric RoPE ($Q$ unrotated, $K$ rotated) | ❌ None | Hierarchical Tree |
| **Ablation 3 (Dual RoPE)**| Symmetric RoPE (Q & K rotated) | 🔒 **Frozen** | Symmetric RoPE (Arbitrary relative pos) | ❌ None | Hierarchical Tree |
| **Ablation 4 (FFN)** | Projections + FFN Transplant | 🔒 **Frozen** | Asymmetric RoPE ($Q$ unrotated, $K$ rotated) | ✅ Transplanted $\text{FFN}^{(l)}$ | Hierarchical Tree |
