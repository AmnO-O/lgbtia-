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
ROLE_HINT = 4
NUM_ROLES = 5


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

        if use_msd:
            self.msd_head = MultiSampleDropoutHead(
                in_features=d_model,
                out_features=1,
                dropout_rates=msd_dropout_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
            )
        else:
            self.linear = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                NormClass(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def forward(
        self,
        z_classes: torch.Tensor,
        return_all_msd_logits: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        B, num_classes, d = z_classes.shape
        # Use .contiguous().reshape(...) instead of .view(...) to handle non-contiguous memory layouts safely
        z_flat = z_classes.contiguous().reshape(B * num_classes, d)
        z_normed = self.norm(z_flat)

        if self.use_msd:
            msd_outputs = self.msd_head(z_normed) # List of [B * num_classes, 1]
            branch_logits = [out.contiguous().reshape(B, num_classes) for out in msd_outputs]
            if return_all_msd_logits and self.training:
                return branch_logits
            return torch.mean(torch.stack(branch_logits, dim=0), dim=0)
        else:
            out = self.linear(z_normed) # [B * num_classes, 1]
            return out.contiguous().reshape(B, num_classes)


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Production-Grade Task B Model with Learnable Class Prototype Queries
    and Additive Latent Privileged Information Injection:
    
    1. Layer 0: Explicit Context Role Injection (Title=1, Desc=2, Comment=3).
    2. Additive Latent Teacher Residual Injection:
       - Compresses teacher rationale h_vec via mmBERT [CLS] + RMSNorm Adapter.
       - Fuses directly: H_fused = H_seq + (alpha * h_vec.unsqueeze(1)).
       - When alpha=0.0 or hint is None: H_fused is 100% IDENTICAL to baseline unguided sequence.
    3. Layer 1: Vectorized Class-Aware Cross-Attention with 3 learnable prototype queries.
    4. Layer 3: Shared Class Query Scoring Head with Multi-Sample Dropout.
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
        **kwargs
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
        # LAYER 0: Role Embeddings (PAD, Title, Description, Comment, Hint)
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # PRIVILEGED INFORMATION: Additive Latent Teacher Adapter
        # ---------------------------------------------------------------------
        self.hint_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            NormClass(d_model),
            nn.Dropout(dropout)
        )

        # ---------------------------------------------------------------------
        # LAYER 1: 3 Continuous Learnable Class Queries + Cross-Attention
        # ---------------------------------------------------------------------
        self.class_queries = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)
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

    def extract_hint_vector(
        self,
        hint_ids: torch.Tensor,
        hint_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts and projects privileged rationale representation from teacher stream.
        Runs under no_grad to preserve frozen teacher representation and save VRAM.
        """
        with torch.no_grad():
            h_hint = self.mmbert(input_ids=hint_ids, attention_mask=hint_mask).last_hidden_state[:, 0, :] # [B, d_model]
        return self.hint_adapter(h_hint) # [B, d_model]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        hint_ids: Optional[torch.Tensor] = None,
        hint_mask: Optional[torch.Tensor] = None,
        hint_alpha: float = 0.0,
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
        B, S = input_ids.shape

        # 1. Primary Sequence Backbone Forward + Role Embedding Injection
        backbone_trainable = self.training and any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        e_role = self.role_embeddings(role_ids)
        if e_role.dtype != h_mmbert.dtype:
            e_role = e_role.to(h_mmbert.dtype)

        h_role = self.norm_input(h_mmbert + e_role)
        h_role = self.dropout_input(h_role) # [B, S, d_model]

        # 2. Additive Latent Privileged Guidance Fusion (Only if hint_alpha > 0 and hint provided)
        if hint_ids is not None and hint_mask is not None and hint_alpha > 0.0:
            hint_vec = self.extract_hint_vector(hint_ids, hint_mask) # [B, d_model]
            if hint_vec.dtype != h_role.dtype:
                hint_vec = hint_vec.to(h_role.dtype)
            # Broadcast addition across sequence length S: [B, S, d_model] + [B, 1, d_model]
            h_role = h_role + (hint_alpha * hint_vec.unsqueeze(1))

        # 3. Expand 3 Class Prototype Queries to Batch: [B, total_queries, d_model]
        # .contiguous() ensures memory layout compatibility across CUDA devices
        q_raw = self.class_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        if q_raw.dtype != h_role.dtype:
            q_raw = q_raw.to(h_role.dtype)

        key_padding_mask = (attention_mask == 0)

        # 4. Layer 1: Vectorized Class-Aware Cross-Attention
        q_normed = self.norm_q(q_raw)
        kv_normed = self.norm_kv(h_role)

        if return_attention_map:
            z_attn, attn_weights = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True
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

        z = (q_raw + self.dropout_cross(z_attn)).contiguous() # [B, total_queries, d_model]

        # 5. Layer 2: Optional Inter-Class Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = (z + self.dropout_self(z_self)).contiguous()

        # Pool slots per class if num_slots_per_class > 1
        if self.num_slots_per_class == 1:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.d_model)
        else:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.num_slots_per_class, self.d_model).mean(dim=2)

        # 6. Layer 3: Shared Class Query Scoring Head
        out_logits = self.scoring_head(
            z_classes=z_classes,
            return_all_msd_logits=return_all_msd_logits
        )

        # 7. Task C Latent Bridge: Probability-Weighted Blend
        if isinstance(out_logits, list):
            eval_logits = torch.mean(torch.stack(out_logits, dim=0), dim=0)
        else:
            eval_logits = out_logits

        probs = F.softmax(eval_logits, dim=-1)
        h_B = torch.sum(probs.unsqueeze(-1) * z_classes, dim=1)
        if detach_bridge:
            h_B = h_B.detach()

        if return_gates:
            return out_logits, h_B, None, attn_weights

        return out_logits, h_B, attn_weights

    @property
    def num_queries(self) -> int:
        return self.total_queries

    def init_queries_from_text(self, tokenizer=None, device=None) -> None:
        pass
