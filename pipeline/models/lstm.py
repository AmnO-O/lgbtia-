import torch
import torch.nn as nn
from .classifier import make_head
from ..config import TARGET_DIM

class PytorchRNNLSTM(nn.Module):
    """
    Bidirectional LSTM architecture for multi-task stereotype & hate speech detection.
    Extracts sequence context in both forward and backward directions with masked pooling
    to prevent context degradation on padded sequences.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256,
                 target_dim: int = TARGET_DIM, pretrained_embeddings=None, device: str = 'cpu'):
        super(PytorchRNNLSTM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        lstm_output_dim = hidden_dim * 2

        self.st_head = make_head(lstm_output_dim, 1)
        self.hs_head = make_head(lstm_output_dim, 3)
        self.tg_head = make_head(lstm_output_dim, target_dim)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded) # [B, S, 2 * hidden_dim]

        # Masked mean pooling across valid sequence tokens (BUG-05 fix)
        mask = (x != 0).unsqueeze(-1).float() # [B, S, 1]
        sum_pooled = (lstm_out * mask).sum(dim=1) # [B, 2 * hidden_dim]
        lengths = mask.sum(dim=1).clamp(min=1.0) # [B, 1]
        pooled_representation = sum_pooled / lengths # [B, 2 * hidden_dim]

        return (
            self.st_head(pooled_representation),
            self.hs_head(pooled_representation),
            self.tg_head(pooled_representation)
        )
