import torch
import torch.nn as nn
from typing import Optional
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
    or pooled token sequences.
    """
    def __init__(self, in_dim: int = 768, target_dim: int = TARGET_DIM, vocab_size: Optional[int] = None):
        super(FeatureClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_dim, padding_idx=0) if vocab_size is not None else None
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if x.dtype in (torch.int32, torch.int64):
            if self.embedding is not None:
                x = self.embedding(x)
                if mask is not None:
                    m = mask.unsqueeze(-1).float()
                    x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    x = x.mean(dim=1)
            else:
                x = x.float()
                if x.ndim == 3:
                    x = x.mean(dim=1)
        return self.st_head(x), self.hs_head(x), self.tg_head(x)


class MMBertFeatureClassifier(nn.Module):
    """mmBERT backbone followed by lightweight pooled FeatureClassifier heads."""
    def __init__(self, mmbert_model: nn.Module, d_model: int = 768, target_dim: int = TARGET_DIM):
        super().__init__()
        self.mmbert = mmbert_model
        self.classifier = FeatureClassifier(in_dim=d_model, target_dim=target_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden[:, 0, :]

        return self.classifier(pooled)
