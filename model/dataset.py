"""
PyTorch Dataset for crypto time-series.

Each sample = (feature_sequence, coin_idx, target_returns)
  feature_sequence: (seq_len, n_features)  — normalised with rolling z-score
  coin_idx:         ()  int64              — stable symbol ID for coin embedding
  target_returns:   (n_horizons,)           — raw log-returns (un-scaled)

Key design decision: the full per-symbol DataFrame is stored in ONE
CryptoSequenceDataset; train/val/test splits are Subset *index views*,
not separate sliced DataFrames.  This means val/test samples can look back
into the training period for their rolling z-score statistics, eliminating
the cold-start normalisation artifact that deflates early val loss.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

from data.features import FEATURE_COLS, TARGET_COLS


class CryptoSequenceDataset(Dataset):
    """
    Sliding-window dataset for one symbol.

    Normalization: each feature is z-scored over the *preceding* `norm_window`
    timesteps to avoid look-ahead.  A minimum std of 1e-8 prevents division by zero.

    coin_idx is a stable integer ID for this symbol so the model's coin
    embedding layer can learn symbol-specific behaviour (volatility regime,
    BTC correlation, sector clustering, etc.).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 168,
        norm_window: int = 336,    # 2× sequence length for stable statistics
        coin_idx: int = 0,
    ):
        self.seq_len      = seq_len
        self.norm_window  = norm_window
        self.start_offset = max(seq_len, norm_window)
        self.coin_idx     = coin_idx

        self.features = df[FEATURE_COLS].values.astype(np.float32)
        self.targets  = df[TARGET_COLS].values.astype(np.float32)
        self.n        = len(df)

    def __len__(self):
        return max(0, self.n - self.start_offset)

    def __getitem__(self, idx):
        end   = self.start_offset + idx   # exclusive end (label row)
        start = end - self.seq_len

        seq = self.features[start:end].copy()   # (T, F)

        # Rolling z-score: stats computed over norm_window before sequence end.
        # Because splits are Subset index views (not sliced DataFrames), this
        # window reaches back into earlier data for val/test samples.
        norm_start = max(0, end - self.norm_window)
        norm_slice = self.features[norm_start:end]
        mean = norm_slice.mean(axis=0, keepdims=True)
        std  = norm_slice.std(axis=0, keepdims=True).clip(min=1e-8)
        seq  = (seq - mean) / std

        # Target is the log-return computed from the last candle in the sequence
        target = self.targets[end - 1]          # (n_horizons,)

        coin   = torch.tensor(self.coin_idx, dtype=torch.long)
        return torch.from_numpy(seq), coin, torch.from_numpy(target)


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[range, range, range]:
    """Chronological index ranges — never shuffles time."""
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return range(0, t1), range(t1, t2), range(t2, n)


def build_datasets(
    symbol_dfs: dict[str, pd.DataFrame],
    seq_len: int = 168,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """
    Build combined train / val / test datasets across all symbols.

    Returns (train_ds, val_ds, test_ds, test_dfs, num_coins):
      - train_ds / val_ds / test_ds: ConcatDataset (or None if empty)
      - test_dfs:  {symbol: raw DataFrame slice} for backtesting
      - num_coins: total number of distinct symbols (for embedding layer)
    """
    norm_window  = seq_len * 2   # keep in sync with CryptoSequenceDataset default
    symbol_index = {sym: i for i, sym in enumerate(symbol_dfs)}   # stable ordering

    train_sets, val_sets, test_sets = [], [], []
    test_dfs: dict[str, pd.DataFrame] = {}

    for symbol, df in symbol_dfs.items():
        coin_idx = symbol_index[symbol]
        full_ds  = CryptoSequenceDataset(df, seq_len, norm_window, coin_idx)
        n = len(full_ds)   # number of valid sliding windows

        if n < seq_len * 4:
            continue

        tr_idx, val_idx, te_idx = _split_indices(n, train_ratio, val_ratio)

        if len(tr_idx)  > 0: train_sets.append(Subset(full_ds, tr_idx))
        if len(val_idx) > 0: val_sets.append(Subset(full_ds, val_idx))
        if len(te_idx)  > 0: test_sets.append(Subset(full_ds, te_idx))

        # Backtest slice: start from te_idx.start (window index into full_ds),
        # NOT full_ds.start_offset + te_idx.start.  The backtest engine
        # instantiates a fresh CryptoSequenceDataset which applies its own
        # start_offset, so we need exactly start_offset rows of lookback
        # before the first real test bar.  te_idx.start rows of df provides
        # that lookback, and the engine's start_offset then lands precisely
        # on the first test sample.  Storing from (start_offset + te_idx.start)
        # would skip start_offset test rows and use future test data for
        # rolling z-score normalisation.
        test_dfs[symbol] = df.iloc[te_idx.start:].reset_index(drop=True)

    train_ds = ConcatDataset(train_sets) if train_sets else None
    val_ds   = ConcatDataset(val_sets)   if val_sets   else None
    test_ds  = ConcatDataset(test_sets)  if test_sets  else None
    num_coins = len(symbol_index)

    return train_ds, val_ds, test_ds, test_dfs, num_coins
