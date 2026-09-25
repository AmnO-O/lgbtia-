import torch
import torch.nn as nn
from .classifier import make_head
from ..config import TARGET_DIM

class PytorchRNNLSTM(nn.Module):
    """
    Bidirectional LSTM architecture for multi-task stereotype & hate speech detection.
    Extracts sequence context in both forward and backward directions.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256,
                 target_dim: int = TARGET_DIM, pretrained_embeddings=None, device: str = 'cpu'):
        super(PytorchRNNLSTM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

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
        lstm_out, _ = self.lstm(embedded)
        # Take hidden state corresponding to the final timestep
        final_hidden_state = lstm_out[:, -1, :]

        return (
            self.st_head(final_hidden_state),
            self.hs_head(final_hidden_state),
            self.tg_head(final_hidden_state)
        )
