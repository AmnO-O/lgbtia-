import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Role IDs for explicit role injection
# 0: Pad / Special tokens
# 1: Title (<T>...</T>)
# 2: Description (<D>...</D>)
# 3: Comment (<C>...</C>)
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
NUM_ROLES = 4

# Class index ordering for Task B queries:
# 0: non_hate ('no')
# 1: implicit ('yes_implicit')
# 2: explicit ('yes_explicit')
CLASS_NON_HATE = 0
CLASS_IMPLICIT = 1
CLASS_EXPLICIT = 2
NUM_CLASSES = 3

class TaskBClassAwareAttentionModel(nn.Module):
    """
    Task B (Hate Speech) Class-Aware Attention Architecture with:
      - Layer 0: mmBERT backbone + Explicit Role Embeddings (Title vs Description vs Comment)
      - Layer 1: Class-Aware Multi-Head Cross-Attention (MHCA) with 3 learned label queries
                 [q_NonHate, q_Implicit, q_Explicit] ∈ [3, d_model]
      - Layer 2: Query Interaction Layer (MHSA) between label queries for boundary calibration
      - Layer 3: Shared Scoring Head f_θ (MLP) -> logits s ∈ [B, 3]
      - Task C Bridge: Hate-Type-Aware Representation h_B = ∑_c (p_c · z'_c) ∈ [B, d_model]
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_query_interaction: bool = True,   # Ablation hypothesis H2 toggle
        hidden_dim: Optional[int] = None,
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        hidden_dim = hidden_dim or d_model // 2

        # ---------------------------------------------------------------------
        # LAYER 0: Explicit Role Injection Embeddings
        # ---------------------------------------------------------------------
        # E_role for Title, Description, Comment, and Pad
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        self.layer_norm_input = nn.LayerNorm(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: Class-Aware Multi-Head Cross-Attention (MHCA)
        # ---------------------------------------------------------------------
        # 3 Learned Class Queries: [q_NonHate, q_Implicit, q_Explicit]
        self.query_embeddings = nn.Parameter(torch.empty(NUM_CLASSES, d_model))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        # Cross-Attention where Q=learned class queries, K=V=H_final
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 2: Query Interaction Layer (MHSA) [Ablation H2]
        # Self-Attention between the 3 class queries (NonHate <-> Implicit <-> Explicit)
        # ---------------------------------------------------------------------
        if self.use_query_interaction:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm_self = nn.LayerNorm(d_model)
            self.dropout_self = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 3: Joint Cross-Class Classification Head
        # Projects concatenated query representations [B, 3 * d_model] -> [B, NUM_CLASSES]
        # Enables joint comparative reasoning across (NonHate vs. Implicit vs. Explicit)
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(NUM_CLASSES * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S] (1 for valid tokens, 0 for pad)
            role_ids: [B, S] (0: pad, 1: title, 2: desc, 3: comment)
            return_attention_map: Whether to return the [B, 3, S] attention interpretability map
        Returns:
            logits: [B, 3] (raw classification scores s for [no, yes_implicit, yes_explicit])
            h_B: [B, d_model] (Hate-Type-Aware Representation for Task C bridge)
            attn_weights: [B, 3, S] if return_attention_map else None
        """
        B, S = input_ids.shape

        # ---------------------------------------------------------------------
        # LAYER 0: Encoder & Role Injection
        # ---------------------------------------------------------------------
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_mmbert = h_mmbert.to(torch.float32)  # [B, S, d_model]

        # Role embeddings
        e_role = self.role_embeddings(role_ids)  # [B, S, d_model]
        h_final = self.layer_norm_input(h_mmbert + e_role)
        h_final = self.dropout_input(h_final)    # [B, S, d_model]

        # ---------------------------------------------------------------------
        # LAYER 1: Class-Aware Multi-Head Cross-Attention (MHCA)
        # ---------------------------------------------------------------------
        # Expand learned queries for batch: Q_base [3, d_model] -> Q [B, 3, d_model]
        q = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, 3, d_model]

        # PyTorch MultiheadAttention key_padding_mask: True indicates tokens to ignore
        key_padding_mask = (attention_mask == 0)

        z_attn, attn_weights = self.cross_attention(
            query=q,
            key=h_final,
            value=h_final,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True  # Average across heads: [B, 3, S]
        )
        z = self.layer_norm_cross(q + self.dropout_cross(z_attn))  # [B, 3, d_model]

        # ---------------------------------------------------------------------
        # LAYER 2: Query Interaction Layer (MHSA) [Ablation Hypothesis H2]
        # ---------------------------------------------------------------------
        if self.use_query_interaction:
            z_self, _ = self.self_attention(
                query=z,
                key=z,
                value=z,
                need_weights=False
            )
            z_prime = self.layer_norm_self(z + self.dropout_self(z_self))  # [B, 3, d_model]
        else:
            z_prime = z  # [B, 3, d_model]

        # ---------------------------------------------------------------------
        # LAYER 3: Joint Cross-Class Classification Head
        # Concatenate 3 class representations: [B, 3, d_model] -> [B, 3 * d_model]
        # ---------------------------------------------------------------------
        z_flat = z_prime.reshape(B, NUM_CLASSES * self.d_model)  # [B, 3 * d_model]
        s = self.classifier(z_flat)                              # [B, 3] (raw logits)

        # ---------------------------------------------------------------------
        # TASK C BRIDGE: Hate-Type-Aware Representation h_B
        # h_B = ∑_c (p_c · z'_c) ∈ [B, d_model]
        # ---------------------------------------------------------------------
        probs = F.softmax(s, dim=-1).unsqueeze(-1)  # [B, 3, 1]
        h_B = (probs * z_prime).sum(dim=1)          # [B, d_model]

        return s, h_B, (attn_weights if return_attention_map else None)
