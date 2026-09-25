import re
from typing import Optional
import torch
import torch.nn as nn
from ..config import TARGET_DIM

def layer_index_in_name(name: str) -> Optional[int]:
    """
    Extracts the encoder layer block index from the parameter name.
    Supports both ModernBERT ('layers.21...') and traditional BERT ('encoder.layer.9...').
    """
    m = re.search(r'(?:^|\.)(?:layers|layer|blocks)\.(\d+)\b', name)
    return int(m.group(1)) if m else None


def count_encoder_blocks(model: nn.Module) -> int:
    """Counts the total number of encoder transformer blocks by parameter name inspection."""
    indices = {layer_index_in_name(n) for n, _ in model.named_parameters()}
    indices.discard(None)
    return len(indices)


def unfreeze_last_n(model: nn.Module, n: int) -> int:
    """
    Unfreezes the last n encoder layers of the backbone model while keeping earlier layers frozen.
    Returns the total number of encoder layers found.
    """
    groups = {}
    for name, p in model.named_parameters():
        idx = layer_index_in_name(name)
        if idx is not None:
            groups.setdefault(idx, []).append(p)
            
    if not groups:
        return 0
        
    # Unfreeze the last n layers
    sorted_keys = sorted(groups.keys())
    target_keys = sorted_keys[-n:] if n > 0 else []
    for k in sorted_keys:
        requires = k in target_keys
        for p in groups[k]:
            p.requires_grad = requires
            
    return len(groups)


class MMBertTransformerModel(nn.Module):
    """
    Hybrid Architecture:
      1. Pretrained mmBERT (ModernBERT 22-layer backbone) for rich contextual token representations.
      2. Domain-specific PyTorch TransformerEncoder with padding masks.
      3. Masked mean pooling.
      4. Multi-task heads for Stereotype, Hate Speech, and Target identity detection.
    """
    def __init__(self, mmbert_model: nn.Module, d_model: int = 768, num_heads: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 1024, dropout: float = 0.3,
                 target_dim: int = TARGET_DIM):
        super(MMBertTransformerModel, self).__init__()
        self.mmbert = mmbert_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.st_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.hs_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        self.tg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Determine whether any backbone weights require gradients
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        if backbone_trainable:
            seq_hidden = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                seq_hidden = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        seq_hidden = seq_hidden.to(torch.float32)  # (batch_size, seq_len, d_model)

        # Apply domain transformer encoder with padding mask (True for pad tokens)
        src_key_padding_mask = (attention_mask == 0)
        x = self.transformer_encoder(seq_hidden, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling (disregards padded tokens)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.st_head(x), self.hs_head(x), self.tg_head(x)
