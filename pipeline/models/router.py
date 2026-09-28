import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .norm import RMSNorm


def masked_mean_pooling(h_role: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes attention-masked mean pooling across sequence tokens, exactly matching mmBERT/ModernBERT retrieval standard.
    
    Args:
        h_role: Hidden representations with role embeddings [B, S, d_model]
        attention_mask: Binary attention mask [B, S] (1 for valid token, 0 for pad)
    Returns:
        Pooled global context vector [B, d_model]
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(h_role).float()
    sum_embeddings = torch.sum(h_role * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class ContextRouter(nn.Module):
    """
    Context-Aware Dynamic Router for Mixture-of-Experts Query Banks.
    
    Aggregates whole-sequence contextual representations (Title + Description + Comment)
    via Masked Mean Pooling and routes dynamic weights g = [g_1, ..., g_N] across N experts.
    """
    def __init__(
        self,
        d_model: int = 768,
        num_experts: int = 4,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.10,
        use_rmsnorm: bool = True
    ):
        super(ContextRouter, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.temperature = max(float(temperature), 1e-3)
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm_in = NormClass(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            NormClass(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Initialize final layer weights with small std for balanced initial routing
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h_role: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_role: Sequence representations with role embeddings [B, S, d_model]
            attention_mask: Binary attention mask [B, S]
            return_logits: If True, also returns pre-softmax router logits [B, num_experts]
        Returns:
            gates: Normalized routing weights [B, num_experts] with sum(g) = 1
            router_logits: Optional pre-softmax logits [B, num_experts]
        """
        # 1. Masked Mean Context Pooling
        h_ctx = masked_mean_pooling(h_role, attention_mask) # [B, d_model]
        h_norm = self.norm_in(h_ctx)
        
        # 2. Router MLP
        router_logits = self.mlp(h_norm) # [B, num_experts]
        
        # 3. Temperature Softmax Gating
        gates = F.softmax(router_logits / self.temperature, dim=-1) # [B, num_experts]
        
        if return_logits:
            return gates, router_logits
        return gates, None
