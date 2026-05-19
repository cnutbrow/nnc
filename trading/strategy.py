"""
Signal generation from model predictions.

The model outputs predicted log-returns at horizons [1h, 4h, 12h, 24h].
We combine them into a single composite signal and size positions using
a simplified Kelly criterion.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# Horizon weights: favour near-term predictions for actionable signals
HORIZON_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10])


@dataclass
class Signal:
    direction: int      # +1 long, -1 short, 0 flat
    strength: float     # 0.0 – 1.0  (used for position sizing)
    raw_pred: np.ndarray   # raw predicted log-returns per horizon


def generate_signal(pred: np.ndarray, cfg) -> Signal:
    """
    Args:
        pred: (4,) array of predicted log-returns for each horizon
        cfg:  TradingConfig

    Returns a Signal.
    """
    composite = float(np.dot(pred, HORIZON_WEIGHTS))

    if composite >= cfg.long_entry_threshold:
        direction = 1
        strength  = min(composite / (cfg.long_entry_threshold * 4), 1.0)
    elif composite <= cfg.short_entry_threshold:
        direction = -1
        strength  = min(abs(composite) / (abs(cfg.short_entry_threshold) * 4), 1.0)
    else:
        direction = 0
        strength  = 0.0

    return Signal(direction=direction, strength=strength, raw_pred=pred)


def kelly_position_size(
    strength: float,
    portfolio_value: float,
    price: float,
    cfg,
    volatility: float = 0.02,
) -> float:
    """
    Quarter-Kelly fraction scaled by signal strength.
    Returns position size in base currency units.
    """
    # f* = (edge) / (variance) — simplified
    edge     = strength * cfg.long_entry_threshold
    variance = volatility ** 2
    kelly    = edge / max(variance, 1e-9)

    # Apply fraction and portfolio cap
    fraction  = cfg.kelly_fraction * kelly
    fraction  = min(fraction, cfg.max_position_pct)
    usd_size  = portfolio_value * fraction
    units     = usd_size / price

    return max(units, 0.0)


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    sequences: torch.Tensor,
    coin_idx: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run inference on a batch of sequences.
    sequences: (B, T, F) float tensor
    coin_idx:  (B,) int64 tensor of coin IDs (optional)
    Returns:   (B, 4) numpy array
    """
    model.eval()
    model_device = next(model.parameters()).device
    sequences = sequences.to(model_device)
    if coin_idx is not None:
        coin_idx = coin_idx.to(model_device)
    preds = model(sequences, coin_idx=coin_idx)
    return preds.cpu().numpy()
