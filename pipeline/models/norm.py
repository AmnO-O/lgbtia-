import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) - Zhang & Sennrich (2019).
    Scales inputs by the root mean square of activations for faster execution,
    stable gradients, and lower memory footprint compared to LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed
