#!/usr/bin/env python3
"""
Grid-search over TradingConfig parameters without re-running model inference.

Steps:
  1. Load model + test_dfs once
  2. Pre-compute all predictions (the slow part)
  3. Sweep parameter grid — only the fast simulation loop runs each time
  4. Print sorted results table

Usage:
    python tune_backtest.py [--checkpoint models/best_model.pt]
"""

import argparse
import copy
import itertools
import logging
import os
import pickle
import sys

import numpy as np
import torch

logging.basicConfig(
    level=logging.WARNING,   # suppress info noise during sweep
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/best_model.pt')
    return p.parse_args()


# ── Patch BacktestEngine to accept pre-computed sym_data ─────────────────────

def _run_precomputed(engine, sym_data: dict) -> dict:
    """Like engine.run() but skips the inference step."""
    timestamps = sorted({ts for d in sym_data.values() for ts in d['timestamps']})
    return engine._simulate(sym_data, timestamps)


def main():
    args = parse_args()

    from config import ModelConfig, TradingConfig
    from data.features import FEATURE_COLS
    from model.architecture import CryptoTransformer
    from trading.backtest import BacktestEngine

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(os.path.dirname(args.checkpoint), 'model_meta.pkl')
    num_coins = 0
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            num_coins = pickle.load(f).get('num_coins', 0)

    model_cfg = ModelConfig()
    model = CryptoTransformer(
        n_features=len(FEATURE_COLS),
        tcn_channels=model_cfg.tcn_channels,
        kernel_size=model_cfg.kernel_size,
        d_model=model_cfg.transformer_d_model,
        nhead=model_cfg.transformer_nhead,
        n_transformer_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
        output_dim=model_cfg.output_dim,
        num_coins=num_coins,
        patch_size=model_cfg.patch_size,
        drop_path_rate=model_cfg.drop_path_rate,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)
    print(f'Loaded model from {args.checkpoint}')

    # ── Load test data ─────────────────────────────────────────────────────────
    with open('models/test_dfs.pkl', 'rb') as f:
        test_dfs: dict = pickle.load(f)
    print(f'Loaded test data: {len(test_dfs)} symbols')

    # ── Pre-compute predictions (once) ────────────────────────────────────────
    base_cfg = TradingConfig()
    engine   = BacktestEngine(model, device, base_cfg)

    print('Running model inference on test set (once)…')
    symbol_list = list(test_dfs.keys())
    coin_index  = {sym: i for i, sym in enumerate(symbol_list)}

    sym_data: dict = {}
    for symbol, df in test_dfs.items():
        arr = engine._make_predictions(df, symbol, coin_idx=coin_index[symbol])
        if arr is not None:
            sym_data[symbol] = arr
    print(f'Predictions cached for {len(sym_data)} symbols\n')

    # ── Parameter grid ────────────────────────────────────────────────────────
    # entry_threshold: symmetric long/short (|short| = long)
    thresholds     = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
    max_pos_pcts   = [0.04, 0.06, 0.08, 0.12]
    sl_tp_pairs    = [(0.02, 0.04), (0.03, 0.06), (0.03, 0.09), (0.04, 0.08)]
    allow_shorts   = [True, False]

    grid = list(itertools.product(thresholds, max_pos_pcts, sl_tp_pairs, allow_shorts))
    print(f'Sweeping {len(grid)} parameter combinations…\n')

    results = []

    for threshold, max_pos, (sl, tp), shorts in grid:
        cfg = copy.copy(base_cfg)
        cfg.long_entry_threshold  =  threshold
        cfg.short_entry_threshold = -threshold
        cfg.max_position_pct      = max_pos
        cfg.stop_loss_pct         = sl
        cfg.take_profit_pct       = tp

        # Patch engine config
        engine.cfg = cfg

        # trading.backtest imports generate_signal via
        #   from trading.strategy import ..., generate_signal
        # so patching trading.strategy has no effect — we must patch the name
        # as it is bound inside trading.backtest.
        import trading.backtest as bt_mod
        from trading.strategy import generate_signal as _orig_gen, Signal

        if not shorts:
            def _long_only(pred, cfg_inner):
                sig = _orig_gen(pred, cfg_inner)
                return Signal(direction=0, strength=0.0, raw_pred=sig.raw_pred) \
                       if sig.direction == -1 else sig
            bt_mod.generate_signal = _long_only
        else:
            bt_mod.generate_signal = _orig_gen

        try:
            res    = _run_precomputed(engine, sym_data)
            stats  = res['stats']

            results.append({
                'threshold':  threshold,
                'max_pos':    max_pos,
                'sl':         sl,
                'tp':         tp,
                'shorts':     shorts,
                'sharpe':     stats['sharpe'],
                'sortino':    stats['sortino'],
                'return':     stats['total_return'],
                'mdd':        stats['max_drawdown'],
                'win_rate':   stats['win_rate'],
                'pf':         stats['profit_factor'],
                'n_trades':   stats['n_trades'],
                'calmar':     stats['calmar'],
            })
        except Exception:
            pass

        finally:
            # Always restore original so the next iteration starts clean
            bt_mod.generate_signal = _orig_gen

    # ── Sort by Sharpe, print top results ─────────────────────────────────────
    results.sort(key=lambda r: r['sharpe'], reverse=True)

    print(f'{"Thr":>5} {"MaxPos":>6} {"SL":>4} {"TP":>4} {"Shorts":>6} '
          f'{"Sharpe":>7} {"Sortino":>7} {"Return":>7} {"MDD":>7} '
          f'{"WinR":>5} {"PF":>5} {"N":>5}')
    print('─' * 90)

    for r in results[:25]:
        print(
            f'{r["threshold"]:>5.3f} {r["max_pos"]:>6.2f} {r["sl"]:>4.2f} {r["tp"]:>4.2f} '
            f'{"Y" if r["shorts"] else "N":>6} '
            f'{r["sharpe"]:>7.3f} {r["sortino"]:>7.3f} {r["return"]:>7.2%} '
            f'{r["mdd"]:>7.2%} {r["win_rate"]:>5.1%} {r["pf"]:>5.3f} {r["n_trades"]:>5}'
        )

    print('\n── Top config ───────────────────────────────────────────────────────')
    best = results[0]
    print(f'  long_entry_threshold  = {best["threshold"]}')
    print(f'  short_entry_threshold = {-best["threshold"]}')
    print(f'  max_position_pct      = {best["max_pos"]}')
    print(f'  stop_loss_pct         = {best["sl"]}')
    print(f'  take_profit_pct       = {best["tp"]}')
    print(f'  allow_shorts          = {best["shorts"]}')
    print(f'  → Sharpe {best["sharpe"]:.3f}  Return {best["return"]:.2%}  '
          f'MDD {best["mdd"]:.2%}  Win {best["win_rate"]:.1%}  '
          f'Trades {best["n_trades"]}')


if __name__ == '__main__':
    main()
