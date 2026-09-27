import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union

CLASS_NO_HATE = 0
CLASS_IMPLICIT_HATE = 1
CLASS_EXPLICIT_HATE = 2
NUM_CLASSES = 3

# Role IDs for structured YouTube context
# 0: Pad / Special tokens
# 1: Title (<T>...</T>)
# 2: Description (<D>...</D>)
# 3: Comment (<C>...</C>)
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
NUM_ROLES = 4


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) - Zhang & Sennrich (2019).
    Scales inputs by the root mean square of activations for faster execution and stable gradients.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Pure Data-Driven Learnable Latent Query Cross-Attention Architecture for Task B.
    
    Architecture:
      1. Role Embeddings: Distinguishes Comment from Video Title and Description context.
      2. Learnable Latent Class Queries: Pure data-driven prototype tokens [No-Hate, Implicit, Explicit]
         trained end-to-end via gradient descent without hardcoded string dependencies.
      3. Cross-Attention: Latent class queries attend across the input sequence tokens.
      4. Multi-Sample Dropout (MSD): 5 parallel dropout paths for stable, low-variance classification.
      5. Task C Bridge: Hate-type-conditioned latent representation h_B for downstream target prediction.
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.20,
        num_slots_per_class: int = 1,
        use_query_interaction: bool = False,
        hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        **kwargs
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots_per_class = num_slots_per_class
        self.total_queries = NUM_CLASSES * num_slots_per_class
        self.use_query_interaction = use_query_interaction
        self.use_rmsnorm = use_rmsnorm
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ---------------------------------------------------------------------
        # LAYER 0: Role Embeddings (Title vs Description vs Comment)
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: Pure Learnable Class Queries & Cross-Attention (Pre-Norm MHCA)
        # ---------------------------------------------------------------------
        self.query_embeddings = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        self.norm_q_cross = NormClass(d_model)
        self.norm_kv_cross = NormClass(d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # Optional Inter-Query Interaction
        if self.use_query_interaction and self.total_queries > 1:
            self.norm_self = NormClass(d_model)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout_self = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 2: Classification Head with Multi-Sample Dropout (MSD)
        # ---------------------------------------------------------------------
        feat_dim = self.total_queries * d_model
        self.norm_head_in = NormClass(feat_dim)
        self.dense_intermediate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            NormClass(hidden_dim),
            nn.GELU()
        )

        rates = msd_dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30]
        if self.use_msd:
            self.msd_dropouts = nn.ModuleList([nn.Dropout(p) for p in rates])
        else:
            self.msd_dropouts = nn.ModuleList([nn.Dropout(dropout)])

        self.out_proj = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False,
        detach_bridge: bool = False,
        return_all_msd_logits: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            role_ids: [B, S]
            return_attention_map: Whether to compute & return the attention map [B, K, S]
            detach_bridge: Whether to detach h_B gradient for Task C pipeline
            return_all_msd_logits: If True during training, returns list of logits from each MSD branch
        """
        B, S = input_ids.shape

        # 1. Backbone + Role Injection
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        e_role = self.role_embeddings(role_ids)
        if e_role.dtype != h_mmbert.dtype:
            e_role = e_role.to(h_mmbert.dtype)

        h_final = self.norm_input(h_mmbert + e_role)
        h_final = self.dropout_input(h_final) # [B, S, d_model]

        # 2. Cross-Attention with Learnable Class Queries
        q_raw = self.query_embeddings.unsqueeze(0).expand(B, -1, -1) # [B, total_queries, d_model]
        if q_raw.dtype != h_final.dtype:
            q_raw = q_raw.to(h_final.dtype)

        key_padding_mask = (attention_mask == 0)
        q_normed = self.norm_q_cross(q_raw)
        kv_normed = self.norm_kv_cross(h_final)

        if return_attention_map:
            z_attn, attn_weights = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True # [B, total_queries, S]
            )
        else:
            z_attn, _ = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            attn_weights = None

        z = q_raw + self.dropout_cross(z_attn) # [B, total_queries, d_model]

        # Optional query self-interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = z + self.dropout_self(z_self)

        # 3. Multi-Sample Dropout (MSD) Classification Head
        z_flat = z.reshape(B, self.total_queries * self.d_model) # [B, feat_dim]
        h_dense = self.dense_intermediate(self.norm_head_in(z_flat)) # [B, hidden_dim]

        if self.training and return_all_msd_logits:
            msd_logits = [self.out_proj(drop(h_dense)) for drop in self.msd_dropouts]
            s = torch.mean(torch.stack(msd_logits, dim=0), dim=0)
            out_logits = msd_logits
        elif self.training:
            msd_logits = [self.out_proj(drop(h_dense)) for drop in self.msd_dropouts]
            s = torch.mean(torch.stack(msd_logits, dim=0), dim=0)
            out_logits = s
        else:
            s = self.out_proj(h_dense)
            out_logits = s

        # 4. Task C Bridge Representation h_B
        # If 1 slot per class: z is [B, 3, d_model]
        if self.num_slots_per_class == 1:
            z_classes = z
        else:
            z_classes = z.view(B, NUM_CLASSES, self.num_slots_per_class, self.d_model).mean(dim=2)

        probs = F.softmax(s, dim=-1).unsqueeze(-1) # [B, 3, 1]
        h_B = (probs * z_classes).sum(dim=1)       # [B, d_model]

        if detach_bridge:
            h_B = h_B.detach()

        return out_logits, h_B, attn_weights
