from typing import List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EmbeddingProjectionHead(nn.Module if HAS_TORCH else object):
    """PyTorch projection head that projects dense embeddings and applies L2 normalization."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 256):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if HAS_TORCH:
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(p=0.1)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if not HAS_TORCH:
            return x

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)
