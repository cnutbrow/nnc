"""
Signal generation from model predictions.

The model outputs a logit; sigmoid gives P(1h return > 0).
We size positions by how far the probability is from 0.5 (confidence).
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Signal:
    direction: int       # +1 long, -1 short, 0 flat
    strength: float      # 0.0 – 1.0 (used for position sizing)
    raw_pred: np.ndarray # raw model output (probability)


def generate_signal(pred: np.ndarray, cfg) -> Signal:
    """
    pred: (1,) array containing sigmoid probability P(1h up)
    cfg:  TradingConfig — thresholds are in probability space [0, 1]
    """
    prob = float(pred[0])

    if prob >= cfg.long_entry_threshold:
        direction = 1
        strength  = min((prob - cfg.long_entry_threshold) /
                        (1.0 - cfg.long_entry_threshold), 1.0)
    elif prob <= cfg.short_entry_threshold:
        direction = -1
        strength  = min((cfg.short_entry_threshold - prob) /
                        cfg.short_entry_threshold, 1.0)
    else:
        direction = 0
        strength  = 0.0

    return Signal(direction=direction, strength=strength, raw_pred=pred)


def kelly_position_size(
    strength: float,
    portfolio_value: float,
    price: float,
    cfg,
    volatility: float = 0.02,   # unused, kept for interface compatibility
) -> float:
    """Scale position by signal confidence, capped at max_position_pct."""
    fraction = min(cfg.kelly_fraction * strength, cfg.max_position_pct)
    return max(portfolio_value * fraction / price, 0.0)


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
    Returns:   (B, 1) numpy array of sigmoid probabilities P(1h up)
    """
    model.eval()
    model_device = next(model.parameters()).device
    sequences = sequences.to(model_device)
    if coin_idx is not None:
        coin_idx = coin_idx.to(model_device)
    logits = model(sequences, coin_idx=coin_idx)
    probs  = torch.sigmoid(logits)
    return probs.cpu().numpy().reshape(-1, 1)
