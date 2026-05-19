"""
TCN + Transformer architecture for crypto time-series prediction.

Pipeline:
  Input (B, T, F)
  → Input LayerNorm
  → Per-feature linear projection to d_model
  → Coin embedding (added to projection)
  → Temporal Convolutional Network (dilated causal convolutions + stochastic depth)
  → PatchTST-style patch aggregation (mean-pool every P bars → fewer tokens)
  → Linear projection to d_model
  → Sinusoidal positional encoding
  → Transformer Encoder (multi-head self-attention, pre-LN)
  → Attention pooling over all patch tokens
  → Per-horizon output heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Stochastic depth (DropPath) ───────────────────────────────────────────────

class DropPath(nn.Module):
    """
    Drop entire residual branches with probability `drop_prob` during training.
    At eval, behaves as identity.  Applied to the learned residual (not the skip
    connection) so worst-case the block degrades to its skip connection.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # torch.bernoulli produces float 0.0/1.0 directly — avoids the bool*float
        # type-promotion path that MPS handles incorrectly (zeroes residuals).
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ── TCN components ────────────────────────────────────────────────────────────

class CausalConv1d(nn.Conv1d):
    """Causal (left-padded) dilated 1-D convolution."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self._causal_pad = padding

    def forward(self, x):
        out = super().forward(x)
        if self._causal_pad > 0:
            out = out[:, :, : -self._causal_pad]
        return out


class TCNBlock(nn.Module):
    """
    Residual temporal block:
      2 × (CausalConv → GroupNorm → GELU → Dropout)
    with a 1×1 projection residual when in/out channels differ.
    Stochastic depth (DropPath) regularises the learned residual branch.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.res_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = self.res_proj(x)
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.drop(self.act(self.norm2(self.conv2(out))))
        return self.drop_path(out) + residual


class TCN(nn.Module):
    """
    Stack of TCN blocks with exponentially increasing dilation.
    drop_path_rates: linearly increases from 0 → max_drop_path across blocks.
    """

    def __init__(self, in_ch, channels: list[int], kernel_size: int,
                 dropout: float, drop_path_rates: list[float] | None = None):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * len(channels)
        layers = []
        dilation = 1
        prev_ch = in_ch
        for out_ch, dpr in zip(channels, drop_path_rates):
            layers.append(TCNBlock(prev_ch, out_ch, kernel_size, dilation, dropout, dpr))
            prev_ch = out_ch
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.net(x)


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


# ── Attention pooling ─────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Weighted aggregation of all patch tokens via a learnable query vector.

    Replaces the naive "take last patch" approach: last-patch ignores all
    but the final 12-bar window; attention pooling lets the model weight every
    window and learn which parts of the 7-day history matter most for each
    forecast — e.g. the opening-hour spike from 3 days ago may be more
    predictive than the quiet close an hour ago.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        scale = x.size(-1) ** 0.5
        w = torch.softmax((x @ self.query) / scale, dim=1)   # (B, T)
        return (x * w.unsqueeze(-1)).sum(dim=1)               # (B, d_model)


# ── Full model ────────────────────────────────────────────────────────────────

class CryptoTransformer(nn.Module):
    """
    TCN + Transformer encoder for multi-horizon crypto return prediction.

    Improvements over v1:
      - Input LayerNorm before projection (stabilises early training)
      - Stochastic depth in TCN blocks (stronger regularisation than dropout alone)
      - Attention pooling over all patch tokens (vs taking only the last patch)
      - Per-horizon output heads (each horizon specialises independently)
    """

    def __init__(
        self,
        n_features: int,
        tcn_channels: list[int],
        kernel_size: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        n_transformer_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.25,
        output_dim: int = 4,
        num_coins: int = 0,
        patch_size: int = 12,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()

        self.patch_size = patch_size

        # Input normalisation: stabilises gradient magnitudes regardless of
        # how well the upstream rolling z-score handled each feature.
        self.input_norm = nn.LayerNorm(n_features)

        # Project raw features → TCN input channels
        self.input_proj = nn.Linear(n_features, tcn_channels[0])

        # Coin embedding: learnable symbol-specific offset added at projection
        self.coin_embed = (
            nn.Embedding(num_coins, tcn_channels[0]) if num_coins > 0 else None
        )

        # TCN with linearly-increasing stochastic depth across blocks
        n_blocks = len(tcn_channels)
        dprs = [drop_path_rate * i / max(1, n_blocks - 1) for i in range(n_blocks)]
        self.tcn = TCN(tcn_channels[0], tcn_channels, kernel_size, dropout, dprs)
        tcn_out = self.tcn.out_channels

        self.tcn_proj = nn.Linear(tcn_out, d_model)
        self.pos_enc  = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,     # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )

        # Attention pooling: weighted aggregate across all patch tokens
        self.attn_pool = AttentionPool(d_model)

        # Per-horizon output heads: each horizon specialises its own projection.
        # Shorter horizons need to learn finer-grained patterns; longer horizons
        # can rely on broader trend signals — giving each its own parameters
        # lets them diverge in what they attend to.
        head_hidden = d_model // 4
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(output_dim)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final layers start near zero to prevent large wrong-direction logits
        # from poisoning the directional loss at initialisation.
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=0.001)
            nn.init.zeros_(head[-1].bias)
        # Attention pool query starts at zero → uniform weights initially
        nn.init.zeros_(self.attn_pool.query)

    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:        (B, T, F)  normalised feature sequences
        coin_idx: (B,)       symbol IDs for coin embedding (optional)
        returns:  (B, output_dim) predicted log-returns per horizon
        """
        # Input norm + projection
        x = self.input_proj(self.input_norm(x))    # (B, T, tcn_in)

        if self.coin_embed is not None and coin_idx is not None:
            x = x + self.coin_embed(coin_idx).unsqueeze(1)

        # TCN: (B, C, T) → (B, C, T)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, tcn_out)

        # PatchTST aggregation: T bars → T//P patch tokens
        B, T, C = x.shape
        P = self.patch_size
        n_patches = T // P
        x = x[:, :n_patches * P, :].reshape(B, n_patches, P, C).mean(dim=2)

        # Transformer over patch tokens
        x = self.pos_enc(self.tcn_proj(x))    # (B, n_patches, d_model)
        x = self.transformer(x)               # (B, n_patches, d_model)

        # Attention pool → single representation
        x = self.attn_pool(x)                 # (B, d_model)

        # Per-horizon predictions concatenated along last dim
        return torch.cat([h(x) for h in self.heads], dim=-1)   # (B, output_dim)
