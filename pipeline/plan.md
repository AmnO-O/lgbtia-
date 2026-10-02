# 📑 RESEARCH & IMPLEMENTATION BLUEPRINT: PRETRAINED ATTENTION WEIGHT BORROWING (TASK B)
## Isolated Hypothesis Testing: Deep Pretrained Attention Projections as Evidence Retrieval Operators

---

## 🎯 1. CORE RESEARCH HYPOTHESIS & SCIENTIFIC FRAMING

### 1.1 The Exact Research Question & Precise Academic Formulation:
> *"Can trainable latent class queries learn useful evidence retrieval directions when interacting through frozen, deep pretrained ModernBERT attention projection matrices ($W_{QKV}, W_O$)?"*

### 1.2 Methodological Honesty: Topology Replacement & Attention Modification
We explicitly document two intentional architectural choices when writing the paper:
1. **Topology Replacement (Local/Global Self-Attn $\to$ Global Latent-to-Token Cross-Attn)**:
   - ModernBERT alternates between local sliding-window attention (window = 128) and global self-attention every 3 layers ($i \pmod 3 == 0$).
   - In our architecture:
     > *"We transplant the linear projection parameters while replacing the pretrained self-attention topology with global latent-to-token cross-attention."*
   - This replacement is methodologically motivated: Class queries represent abstract global concepts that must maintain unconstrained, full-span visibility across the entire context sequence.
2. **Asymmetric Rotary Embeddings (Unrotated $Q$ + Rotated $K$)**:
   - **Why Unrotated $Q$ ($Q = Z W_Q$)**: The $K$ query latents $Z \in \mathbb{R}^{B \times K \times D}$ represent abstract, position-free conceptual probes (`No Hate`, `Implicit Hate`, `Explicit Hate`). Imposing sequence-position RoPE onto $Z$ would introduce arbitrary, linguistically baseless relative-distance penalties among the classes.
   - **Why Rotated $K$ ($K = \operatorname{RoPE}(H W_K)$)**: The context tokens $H \in \mathbb{R}^{B \times L \times D}$ represent natural language text whose syntactic relationships depend critically on token order. Rotary embeddings on $K$ preserve the full linguistic geometry of the context.

---

## 🏗️ 2. DEDICATED MODULE: `PretrainedCrossAttentionLayer`

### 2.1 Code Implementation Supporting Clean Experimental Controls (A vs B):
We provide an explicit `use_prenorm: bool = False` flag to cleanly separate:
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
    Applies Rotary Position Embedding exclusively to a single tensor (Keys).
    tensor: [B, num_heads, L, head_dim]
    cos, sin: [1, 1, L, head_dim] or [B, 1, L, head_dim]
    """
    return (tensor * cos) + (rotate_half(tensor) * sin)


class PretrainedCrossAttentionLayer(nn.Module):
    """
    Asymmetric Cross-Attention whose Q/K/V/O projections are borrowed
    directly from one pretrained ModernBERT attention block.
    
    Intentional Design Choices:
      - Q is unrotated (Position-Free conceptual query latents).
      - K preserves full ModernBERT RoPE to maintain contextual token geometry.
      - Global non-padding visibility (Transplanted topology from sliding-window to global cross-attention).
      - Configurable Pre-Normalization (Exp A vs Exp B).
    """
    def __init__(self, pretrained_attn, pretrained_norm=None, use_prenorm: bool = False):
        super().__init__()
        self.attn = pretrained_attn
        self.use_prenorm = use_prenorm
        
        # Borrow pretrained projection layers directly (no new parameters initialized)
        self.Wqkv = pretrained_attn.Wqkv
        self.Wo = pretrained_attn.Wo
        
        # Optional: Borrow pretrained LayerNorm / RMSNorm for Experiment B
        self.norm = pretrained_norm if use_prenorm else None
        
        # Borrow rotary embedding module from base ModernBERT attention
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)
        
        self.num_heads = pretrained_attn.config.num_attention_heads
        self.head_dim = pretrained_attn.head_dim
        self.scaling = self.head_dim ** -0.5

    def forward(self, z, h_context, context_mask=None, position_ids=None):
        """
        Args:
            z: [B, K, D] - Query latents (Position-Free, Unrotated)
            h_context: [B, L, D] - Context token representations
            context_mask: [B, L] - Non-padding mask (1 for valid, 0 for pad)
            position_ids: [B, L] - Positional indices for context tokens (0 to L-1)
        """
        B, K, D = z.shape
        _, L, _ = h_context.shape

        # Optional Pre-Normalization (Experiment B control)
        z_in = self.norm(z) if (self.use_prenorm and self.norm is not None) else z
        h_in = self.norm(h_context) if (self.use_prenorm and self.norm is not None) else h_context

        # 1. Chunk fused ModernBERT Wqkv weight [3D, D] into Q, K, V [D, D]
        q_weight, k_weight, v_weight = self.Wqkv.weight.chunk(3, dim=0)

        # 2. Linear Projections
        q = F.linear(z_in, q_weight)       # [B, K, D] from query latents
        k = F.linear(h_in, k_weight)       # [B, L, D] from context
        v = F.linear(h_in, v_weight)       # [B, L, D] from context

        # 3. Reshape into Multi-Head format [B, num_heads, SeqLen, head_dim]
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, d_h]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]

        # 4. Apply Native ModernBERT RoPE Exclusively to Keys (K)
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=h_context.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, position_ids) # ModernBERT rotary_emb signature
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            k = apply_rotary_pos_emb_single(k, cos, sin)

        # 5. Global Asymmetric Attention Matrix (Full Context Visibility)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, K, L]
        if context_mask is not None:
            # Mask out PAD tokens: shape [B, 1, 1, L]
            mask_4d = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -10000.0
            scores = scores + mask_4d

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [B, H, K, d_h]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)

        # 6. Pretrained Wo Linear Projection with Residual Connection
        z_out = z + self.Wo(attn_out)
        return z_out
```

---

## 🔬 3. MATHEMATICAL SPECIFICATIONS OF EXPERIMENTS A & B

Given context tokens $X$, we compute $H_{17} = \operatorname{mmBERT}_{1:17}(X) \in \mathbb{R}^{B \times L \times D}$.
We initialize $K$ query latents $Z_0 = [q_N, q_I, q_E] \in \mathbb{R}^{B \times 3 \times D}$.

### Experiment A (Main: Pure Projection Transplantation - No Normalization):
For layers $l \in [18, 19, 20, 21, 22]$:
$$\begin{aligned}
Q_l &= Z_{l-1} \cdot W_Q^{(l)} \\
K_l &= \operatorname{RoPE}\left(H_{17} \cdot W_K^{(l)}, \text{pos}_{0:L}\right) \\
V_l &= H_{17} \cdot W_V^{(l)} \\
A_l &= \operatorname{Softmax}\left(\frac{Q_l K_l^T}{\sqrt{64}} + M_{\text{non\_pad}}\right) \\
Z_l &= Z_{l-1} + W_{O, l}(A_l V_l)
\end{aligned}$$

### Experiment B (Critical Control: Pretrained Pre-Normalization):
For layers $l \in [18, 19, 20, 21, 22]$:
$$\begin{aligned}
Q_l &= \operatorname{LN}(Z_{l-1}) \cdot W_Q^{(l)} \\
K_l &= \operatorname{RoPE}\left(\operatorname{LN}(H_{17}) \cdot W_K^{(l)}, \text{pos}_{0:L}\right) \\
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

## 🎨 4. QUERY INITIALIZATION PROTOCOL & SCIENTIFIC MEANING

1. **Main Benchmark (Trainable Latent Queries)**:
   $$q_N, q_I, q_E \sim \mathcal{N}(0, 0.02)$$
   - *Scientific Hypothesis*: Tests whether trainable latent queries can learn useful evidence retrieval directions when interacting through frozen, deep pretrained attention projection matrices.
2. **Ablation Comparison (Semantic Embedding Initialization)**:
   $$q_c = \operatorname{mmBERT}\left(\text{Definition of Class } c\right)$$
   - *Scientific Hypothesis*: Tests whether seeding semantic prior definitions accelerates convergence or yields higher final retrieval resolution.

---

## 📈 5. STRUCTURED ABLATION MATRIX

| Model Variant | Formulation | Normalization | RoPE Handling | Pretrained FFN | Classification Head |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A (Main Core)** | Pure Projections Borrowing | ❌ None | Unrotated $Q$ + Rotated $K$ | ❌ None | Hierarchical Tree |
| **Exp B (Control)** | Pre-Normalized Borrowing | ✅ Pre-LN ($\text{LN}(Z), \text{LN}(H)$) | Unrotated $Q$ + Rotated $K$ | ❌ None | Hierarchical Tree |
| **Ablation 1 (Init)** | Pure Projections + Semantic Init | ❌ None | Unrotated $Q$ + Rotated $K$ | ❌ None | Hierarchical Tree |
| **Ablation 2 (FFN)** | Projections + FFN Transplant | Configurable | Unrotated $Q$ + Rotated $K$ | ✅ Transplanted $\text{FFN}^{(l)}$ | Hierarchical Tree |
| **Ablation 3 (Dual)** | Dual-Stream $H_{l-1} \to H_l$ Propagation | Configurable | Unrotated $Q$ + Rotated $K$ | ✅ Transplanted $\text{FFN}^{(l)}$ | Hierarchical Tree |
