import torch
import torch.nn as nn
from .classifier import make_head
from ..config import TARGET_DIM

class VanillaRNNModel(nn.Module):
    """
    Scratch Vanilla Recurrent Neural Network baseline for multi-task
    stereotype, hate speech, and target classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256,
                 target_dim: int = TARGET_DIM, device: str = 'cpu'):
        super(VanillaRNNModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_to_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)

        self.st_head = make_head(hidden_dim, 1)
        self.hs_head = make_head(hidden_dim, 3)
        self.tg_head = make_head(hidden_dim, target_dim)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        embedded = self.embedding(x)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_t = torch.tanh(self.input_to_hidden(x_t) + self.hidden_to_hidden(h_t))

        return self.st_head(h_t), self.hs_head(h_t), self.tg_head(h_t)
