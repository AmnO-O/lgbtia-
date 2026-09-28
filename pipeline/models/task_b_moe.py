import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union

from .norm import RMSNorm
from .router import ContextRouter, masked_mean_pooling
from .query_bank import ParallelQueryBankCrossAttention
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


class TaskB4ExpertMoEModel(nn.Module):
    """
    4-Expert Mixture of Latent Query Banks (MoE-Query) with Context-Aware Masked Mean Router.
    
    Architecture:
      1. Role Embeddings: Injects YouTube context differentiation (<T>, <D>, <C>).
      2. Nhánh 1 (Context Router): Masked Mean Pooling -> MLP -> Softmax Gating g = [g_0, g_1, g_2, g_3].
      3. Nhánh 2 (4-Expert Query Banks): Parallel learnable prototype cross-attention across H_role.
         - Expert 0: Non-Hate / Supportive Semantics
         - Expert 1: Implicit Hate / Sarcasm & Stereotyping
         - Expert 2: Explicit Hate / Direct Slurs & Hostility
         - Expert 3: Context Mismatch / Video-Comment Semantic Discrepancy
      4. Weighted Blending: Differentiable convex combination z_final = sum(g_i * z_i).
      5. Classification Head: Multi-Sample Dropout (MSD) for robust low-variance predictions.
      6. Task C Latent Bridge: Hate-type-conditioned latent representation h_B = z_final.
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_experts: int = 4,
        num_slots_per_expert: int = 1,
        num_heads: int = 8,
        dropout: float = 0.20,
        router_hidden_dim: int = 256,
        router_temperature: float = 1.0,
        use_query_interaction: bool = False,
        hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        **kwargs
    ):
        super(TaskB4ExpertMoEModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots_per_expert = num_slots_per_expert
        self.total_queries = num_experts * num_slots_per_expert
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
        # NHÁNH 1: Context-Aware Dynamic Router
        # ---------------------------------------------------------------------
        self.router = ContextRouter(
            d_model=d_model,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            dropout=dropout * 0.5,
            use_rmsnorm=use_rmsnorm
        )

        # ---------------------------------------------------------------------
        # NHÁNH 2: Parallel 4-Expert Latent Query Banks Cross-Attention
        # ---------------------------------------------------------------------
        self.query_banks = ParallelQueryBankCrossAttention(
            d_model=d_model,
            num_experts=num_experts,
            num_slots_per_expert=num_slots_per_expert,
            num_heads=num_heads,
            dropout=dropout,
            use_query_interaction=use_query_interaction,
            use_rmsnorm=use_rmsnorm
        )

        # Backward compatibility aliases
        self.query_embeddings = self.query_banks.expert_queries

        # ---------------------------------------------------------------------
        # LAYER 3: Multi-Sample Dropout Classification Head
        # ---------------------------------------------------------------------
        self.classifier_head = MultiSampleDropoutHead(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            num_classes=NUM_CLASSES,
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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]
    ]:
        """
        Forward Pass for 4-Expert Dynamic MoE Architecture.
        
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            role_ids: [B, S]
            return_attention_map: Whether to return cross-attention weights [B, num_queries, S]
            return_gates: Whether to return router gates [B, num_experts]
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

        # 2. Nhánh 1: Context Router
        gates, _ = self.router(h_role, attention_mask) # [B, num_experts]

        # 3. Nhánh 2: 4-Expert Query Banks Cross-Attention
        z_experts, attn_weights = self.query_banks(
            h_role=h_role,
            attention_mask=attention_mask,
            return_attention_map=return_attention_map
        ) # [B, num_experts, d_model]

        # 4. Gated Convex Combination: z_final = sum(g_i * z_i)
        # gates: [B, num_experts, 1], z_experts: [B, num_experts, d_model]
        z_final = torch.sum(gates.unsqueeze(-1) * z_experts, dim=1) # [B, d_model]

        # 5. Multi-Sample Dropout Classification Head
        out_logits = self.classifier_head(
            z=z_final,
            return_all_msd_logits=return_all_msd_logits
        )

        # 6. Task C Bridge Representation
        h_B = z_final.detach() if detach_bridge else z_final

        if return_gates:
            return out_logits, h_B, gates, attn_weights

        return out_logits, h_B, attn_weights

    @property
    def num_queries(self) -> int:
        return self.total_queries

    def init_queries_from_text(self, tokenizer=None, device=None) -> None:
        """Compatibility helper - weights are learned end-to-end via gradient descent."""
        pass


# Aliased for 100% backward compatibility with existing scripts and checkpoints
TaskBClassAwareAttentionModel = TaskB4ExpertMoEModel
