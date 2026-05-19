"""
PyTorch Dataset for crypto time-series.

Each sample = (feature_sequence, coin_idx, target_direction)
  feature_sequence: (seq_len, n_features)  — normalised with rolling z-score
  coin_idx:         ()  int64              — symbol ID (unused by baseline model)
  target_direction: ()  float32            — 1.0 if 1h return > 0, else 0.0

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
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 168,
        norm_window: int = 336,
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
        end   = self.start_offset + idx
        start = end - self.seq_len

        seq = self.features[start:end].copy()

        norm_start = max(0, end - self.norm_window)
        norm_slice = self.features[norm_start:end]
        mean = norm_slice.mean(axis=0, keepdims=True)
        std  = norm_slice.std(axis=0, keepdims=True).clip(min=1e-8)
        seq  = (seq - mean) / std

        # Binary direction target: 24h return > 0 (more persistent signal than 1h)
        raw_24h = self.targets[end - 1, 3]
        target  = np.float32(raw_24h > 0)

        coin = torch.tensor(self.coin_idx, dtype=torch.long)
        return torch.from_numpy(seq), coin, torch.tensor(target)


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[range, range, range]:
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

    Returns (train_ds, val_ds, test_ds, test_dfs, num_coins).
    """
    norm_window  = seq_len * 2
    symbol_index = {sym: i for i, sym in enumerate(symbol_dfs)}

    train_sets, val_sets, test_sets = [], [], []
    test_dfs: dict[str, pd.DataFrame] = {}

    for symbol, df in symbol_dfs.items():
        coin_idx = symbol_index[symbol]
        full_ds  = CryptoSequenceDataset(df, seq_len, norm_window, coin_idx)
        n = len(full_ds)

        if n < seq_len * 4:
            continue

        tr_idx, val_idx, te_idx = _split_indices(n, train_ratio, val_ratio)

        if len(tr_idx)  > 0: train_sets.append(Subset(full_ds, tr_idx))
        if len(val_idx) > 0: val_sets.append(Subset(full_ds, val_idx))
        if len(te_idx)  > 0: test_sets.append(Subset(full_ds, te_idx))

        # Backtest slice: start from te_idx.start so the engine's own
        # start_offset lands precisely on the first test sample.
        test_dfs[symbol] = df.iloc[te_idx.start:].reset_index(drop=True)

    train_ds  = ConcatDataset(train_sets) if train_sets else None
    val_ds    = ConcatDataset(val_sets)   if val_sets   else None
    test_ds   = ConcatDataset(test_sets)  if test_sets  else None
    num_coins = len(symbol_index)

    return train_ds, val_ds, test_ds, test_dfs, num_coins
