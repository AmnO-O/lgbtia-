import torch
import torch.nn as nn
from ..config import TARGET_DIM

def make_head(hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Standard MLP classification head with LayerNorm and Dropout."""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, out_dim),
    )

def make_transformer_head(in_dim: int, out_dim: int) -> nn.Sequential:
    """Classification head tailored for Transformer representations."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_dim // 2, out_dim),
    )

class FeatureClassifier(nn.Module):
    """
    Lightweight 3-head classifier operating on pre-extracted feature embeddings
    (e.g., pre-computed mmBERT pooled representations).
    """
    def __init__(self, in_dim: int = 768, target_dim: int = TARGET_DIM):
        super(FeatureClassifier, self).__init__()
        self.st_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, 1)
        )
        self.hs_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, 3)
        )
        self.tg_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, target_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.st_head(x), self.hs_head(x), self.tg_head(x)
