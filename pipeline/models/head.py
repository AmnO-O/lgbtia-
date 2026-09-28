import torch
import torch.nn as nn
from typing import List, Optional, Union
from .norm import RMSNorm


class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout (MSD) Classification Head.
    
    Processes the fused expert representation through an intermediate projection,
    applies M parallel dropout masks with varying rates to generate multiple distinct sub-views,
    and projects each to class logits for low-variance gradient updates.
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 384,
        num_classes: int = 3,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        dropout: float = 0.20,
        use_rmsnorm: bool = True
    ):
        super(MultiSampleDropoutHead, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_msd = use_msd
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.norm_in = NormClass(in_dim)
        self.dense = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            NormClass(hidden_dim),
            nn.GELU()
        )
        
        rates = msd_dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30]
        if self.use_msd:
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in rates])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout)])
            
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        z: torch.Tensor,
        return_all_msd_logits: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z: Input representation [B, in_dim]
            return_all_msd_logits: If True during training, returns a list of tensors [logits_1, ..., logits_M]
        Returns:
            Single averaged logits tensor [B, num_classes] or list of tensors
        """
        h = self.dense(self.norm_in(z)) # [B, hidden_dim]
        
        if self.training:
            branch_logits = [self.classifier(drop(h)) for drop in self.dropouts]
            if return_all_msd_logits:
                return branch_logits
            return torch.mean(torch.stack(branch_logits, dim=0), dim=0)
        
        # Inference mode: direct projection without dropout
        return self.classifier(h)
