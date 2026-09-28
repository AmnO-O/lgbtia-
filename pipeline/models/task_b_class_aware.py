import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union

from .norm import RMSNorm
from .head import MultiSampleDropoutHead

CLASS_NO_HATE = 0
CLASS_IMPLICIT_HATE = 1
CLASS_EXPLICIT_HATE = 2
NUM_CLASSES = 3

# Structured context role IDs
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
NUM_ROLES = 4


class PureClassQueryScoringHead(nn.Module):
    """
    Shared Class Query Scoring Head.
    
    Given N class query representations [B, num_classes, d_model],
    maps each query representation z_c to a scalar score for class c via a shared projection f_theta: R^d -> R^1.
    Supports Multi-Sample Dropout (MSD) for low-variance generalization.
    """
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: Optional[int] = None,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        dropout: float = 0.20,
        use_rmsnorm: bool = True
    ):
        super(PureClassQueryScoringHead, self).__init__()
        self.d_model = d_model
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = NormClass(d_model)
        
        if self.use_msd:
            dropout_rates = msd_dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30]
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
            self.scorers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in dropout_rates
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout)])
            self.scorers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1)
                )
            ])

    def forward(
        self,
        z_classes: torch.Tensor,
        return_all_msd_logits: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z_classes: [B, NUM_CLASSES, d_model]
            return_all_msd_logits: If True and training, returns list of [B, NUM_CLASSES] logits per MSD branch
        Returns:
            logits: [B, NUM_CLASSES]
        """
        B, C, D = z_classes.shape
        z_normed = self.norm(z_classes) # [B, C, D]
        
        branch_logits: List[torch.Tensor] = []
        for drop, scorer in zip(self.dropouts, self.scorers):
            h = drop(z_normed) # [B, C, D]
            s = scorer(h).squeeze(-1) # [B, C]
            branch_logits.append(s)
            
        if self.training and return_all_msd_logits and self.use_msd:
            return branch_logits
            
        # Ensemble average across MSD branches
        avg_logits = torch.mean(torch.stack(branch_logits, dim=0), dim=0) # [B, C]
        return avg_logits


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Pure Learnable Class Queries Cross-Attention (Data-Driven) Architecture for Task B.
    
    Overfitting Defense & Advantages:
      1. Zero Gating Collapse: Replaces complex MoE routers and multi-branch gating with 3 continuous
         learnable class prototype vectors [q_NoHate, q_ImplicitHate, q_ExplicitHate].
      2. Direct Semantic Alignment: Each query directly queries the mmBERT sequence representations
         for affinity with its target hate class.
      3. Parameter Efficient: Uses a shared scoring projection across the 3 query representations,
         drastically reducing parameter count and preventing over-parameterization.
      4. Explicit Context Role Embeddings: Injects YouTube role tokens (<Title>, <Description>, <Comment>).
      5. Task C Latent Bridge: Output probability-weighted representation h_B = sum(p_c * z_c).
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_classes: int = NUM_CLASSES,
        num_slots_per_class: int = 1,
        num_heads: int = 8,
        dropout: float = 0.20,
        use_query_interaction: bool = False,
        hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        **kwargs # Ignores router kwargs (num_experts, router_hidden_dim, etc.) gracefully
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_slots_per_class = num_slots_per_class
        self.total_queries = num_classes * num_slots_per_class
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        self.use_rmsnorm = use_rmsnorm
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ---------------------------------------------------------------------
        # LAYER 0: Role Embeddings (PAD, Title, Description, Comment)
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: 3 Continuous Learnable Class Queries + Cross-Attention
        # ---------------------------------------------------------------------
        # [num_classes * num_slots_per_class, d_model]
        self.class_queries = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

        # Backward compatibility alias
        self.query_embeddings = self.class_queries

        self.norm_q = NormClass(d_model)
        self.norm_kv = NormClass(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 2: Optional Inter-Class Query Self-Attention Interaction
        # ---------------------------------------------------------------------
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
        # LAYER 3: Shared Class Query Scoring Head with Multi-Sample Dropout
        # ---------------------------------------------------------------------
        self.scoring_head = PureClassQueryScoringHead(
            d_model=d_model,
            hidden_dim=hidden_dim,
            use_msd=use_msd,
            msd_dropout_rates=msd_dropout_rates,
            dropout=dropout,
            use_rmsnorm=use_rmsnorm
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False,
        return_gates: bool = False,
        detach_bridge: bool = False,
        return_all_msd_logits: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ]:
        """
        Forward Pass for Pure Learnable Class Queries Cross-Attention Architecture.
        
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            role_ids: [B, S]
            return_attention_map: Whether to return cross-attention weights [B, num_classes, S]
            return_gates: Returns None for gates (for seamless trainer compatibility)
            detach_bridge: Whether to detach h_B gradient for Task C pipeline
            return_all_msd_logits: If True during training, returns list of logits from each MSD branch
        """
        B, S = input_ids.shape

        # 1. Backbone + Role Embedding Injection
        backbone_trainable = self.training and any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        e_role = self.role_embeddings(role_ids)
        if e_role.dtype != h_mmbert.dtype:
            e_role = e_role.to(h_mmbert.dtype)

        h_role = self.norm_input(h_mmbert + e_role)
        h_role = self.dropout_input(h_role) # [B, S, d_model]

        # 2. Expand 3 Class Prototype Queries to Batch: [B, total_queries, d_model]
        q_raw = self.class_queries.unsqueeze(0).expand(B, -1, -1)
        if q_raw.dtype != h_role.dtype:
            q_raw = q_raw.to(h_role.dtype)

        key_padding_mask = (attention_mask == 0) # True for padding positions

        # 3. Layer 1: Vectorized Class-Aware Cross-Attention
        q_normed = self.norm_q(q_raw)
        kv_normed = self.norm_kv(h_role)

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

        # 4. Layer 2: Optional Inter-Class Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = z + self.dropout_self(z_self)

        # Pool slots per class if num_slots_per_class > 1: [B, num_classes, d_model]
        if self.num_slots_per_class == 1:
            z_classes = z.view(B, self.num_classes, self.d_model)
        else:
            z_classes = z.view(B, self.num_classes, self.num_slots_per_class, self.d_model).mean(dim=2)

        # 5. Layer 3: Shared Class Query Scoring Head
        out_logits = self.scoring_head(
            z_classes=z_classes,
            return_all_msd_logits=return_all_msd_logits
        )

        # 6. Task C Latent Bridge: Probability-Weighted Blend
        # Compute probabilities for weighting class representations
        if isinstance(out_logits, list):
            eval_logits = torch.mean(torch.stack(out_logits, dim=0), dim=0)
        else:
            eval_logits = out_logits

        probs = F.softmax(eval_logits, dim=-1) # [B, 3]
        # h_B = sum_{c=0}^2 (p_c * z_c) -> [B, d_model]
        h_B = torch.sum(probs.unsqueeze(-1) * z_classes, dim=1)
        if detach_bridge:
            h_B = h_B.detach()

        if return_gates:
            # Returns None for gates so MoE-aware callers don't fail
            return out_logits, h_B, None, attn_weights

        return out_logits, h_B, attn_weights

    @property
    def num_queries(self) -> int:
        return self.total_queries

    def init_queries_from_text(self, tokenizer=None, device=None) -> None:
        """Compatibility helper - query weights are learned data-driven via SGD/AdamW."""
        pass
