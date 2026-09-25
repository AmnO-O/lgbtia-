import math
import torch
import torch.nn as nn
from .classifier import make_transformer_head
from ..config import TARGET_DIM

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence order injection."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PytorchTransformerModel(nn.Module):
    """
    Standard PyTorch TransformerEncoder architecture built from scratch with
    learned embeddings and sinusoidal positional encoding, mapped to 3 multi-task heads.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.3,
                 target_dim: int = TARGET_DIM):
        super(PytorchTransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.st_head = make_transformer_head(embedding_dim, 1)
        self.hs_head = make_transformer_head(embedding_dim, 3)
        self.tg_head = make_transformer_head(embedding_dim, target_dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        # Global Average Pooling across sequence length
        x = x.mean(dim=1)

        return self.st_head(x), self.hs_head(x), self.tg_head(x)


class CustomTransformerModel(nn.Module):
    """Alternative Transformer with custom multi-head attention blocks."""
    def __init__(self, vocab_size: int, embedding_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, target_dim: int = TARGET_DIM):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.st_head = make_transformer_head(embedding_dim, 1)
        self.hs_head = make_transformer_head(embedding_dim, 3)
        self.tg_head = make_transformer_head(embedding_dim, target_dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.st_head(x), self.hs_head(x), self.tg_head(x)
