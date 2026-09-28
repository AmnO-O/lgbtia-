import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .norm import RMSNorm


class ParallelQueryBankCrossAttention(nn.Module):
    """
    Parallel Multi-Expert Latent Query Bank Cross-Attention.
    
    Maintains N continuous learnable prototype query banks (e.g., Non-Hate, Implicit, Explicit, Context Mismatch).
    Executes vectorized, parallel Pre-Norm Multi-Head Cross-Attention across the entire input sequence H_role.
    """
    def __init__(
        self,
        d_model: int = 768,
        num_experts: int = 4,
        num_slots_per_expert: int = 1,
        num_heads: int = 8,
        dropout: float = 0.20,
        use_query_interaction: bool = False,
        use_rmsnorm: bool = True
    ):
        super(ParallelQueryBankCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots_per_expert = num_slots_per_expert
        self.total_queries = num_experts * num_slots_per_expert
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        # Continuous learnable prototype query vectors [num_experts, num_slots_per_expert, d_model]
        self.expert_queries = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.expert_queries, mean=0.0, std=0.02)
        
        # Pre-normalization layers
        self.norm_q = NormClass(d_model)
        self.norm_kv = NormClass(d_model)
        
        # Vectorized Multi-Head Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)
        
        # Optional Inter-Query Self-Attention
        if self.use_query_interaction and self.total_queries > 1:
            self.norm_self = NormClass(d_model)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout_self = nn.Dropout(dropout)

    def forward(
        self,
        h_role: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_role: Input sequence representations with role embeddings [B, S, d_model]
            attention_mask: Binary mask [B, S]
            return_attention_map: If True, returns cross-attention weights [B, total_queries, S]
        Returns:
            z_experts: Representations per expert [B, num_experts, d_model]
            attn_weights: Optional attention weights tensor
        """
        B, S, _ = h_role.shape
        
        # Expand queries to batch dimension: [B, total_queries, d_model]
        q_raw = self.expert_queries.unsqueeze(0).expand(B, -1, -1)
        if q_raw.dtype != h_role.dtype:
            q_raw = q_raw.to(h_role.dtype)
            
        key_padding_mask = (attention_mask == 0) # True for padding positions
        
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
            
        # Residual connection
        z = q_raw + self.dropout_cross(z_attn) # [B, total_queries, d_model]
        
        # Optional Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = z + self.dropout_self(z_self)
            
        # Reshape to [B, num_experts, num_slots_per_expert, d_model] and pool slots
        if self.num_slots_per_expert == 1:
            z_experts = z.view(B, self.num_experts, self.d_model) # [B, num_experts, d_model]
        else:
            z_experts = z.view(B, self.num_experts, self.num_slots_per_expert, self.d_model).mean(dim=2)
            
        return z_experts, attn_weights
