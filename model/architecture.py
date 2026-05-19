"""
GRU baseline for 1-hour directional prediction.

Pipeline:
  Input (B, T, F)
  → LayerNorm
  → GRU (n_layers, hidden_dim)  → last hidden state h[-1]
  → Dropout → Linear(hidden, 32) → GELU → Linear(32, 1)
  → logit scalar per sample

Sigmoid of the output gives P(1h return > 0).
"""

import torch
import torch.nn as nn


class CryptoGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            n_features, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B,) logit — sigmoid gives P(1h return > 0)
        """
        x = self.input_norm(x)
        _, h = self.gru(x)      # h: (n_layers, B, hidden_dim)
        out = self.head(h[-1])  # (B, 1)
        return out.squeeze(-1)  # (B,)
