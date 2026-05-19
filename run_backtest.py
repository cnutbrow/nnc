#!/usr/bin/env python3
"""
Run backtest using the trained model on the held-out test set.

Usage:
    python run_backtest.py [--checkpoint models/best_model.pt]
                           [--plot]
                           [--output results/backtest.html]
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/best_model.pt')
    p.add_argument('--plot',   action='store_true', help='Show equity curve plot')
    p.add_argument('--output', default='results/backtest.html',
                   help='Save interactive HTML report')
    return p.parse_args()


def buy_and_hold_equity(symbol_dfs: dict, initial_capital: float) -> np.ndarray:
    sym    = next((s for s in symbol_dfs if 'BTC' in s), next(iter(symbol_dfs)))
    df     = symbol_dfs[sym]
    prices = df['close'].values
    units  = (initial_capital / len(symbol_dfs)) / prices[0]
    return np.array([units * p * len(symbol_dfs) for p in prices])


def main():
    args = parse_args()

    from config import ModelConfig, TradingConfig
    from data.features import FEATURE_COLS
    from model.architecture import CryptoGRU
    from trading.backtest import BacktestEngine

    if not os.path.exists(args.checkpoint):
        logger.error(f'Checkpoint not found: {args.checkpoint}')
        logger.error('Run train.py first.')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')

    model_cfg = ModelConfig()
    trade_cfg = TradingConfig()

    model = CryptoGRU(
        n_features=len(FEATURE_COLS),
        hidden_dim=model_cfg.hidden_dim,
        n_layers=model_cfg.n_layers,
        dropout=model_cfg.dropout,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)
    logger.info(f'Loaded model from {args.checkpoint}')

    pkl_path = 'models/test_dfs.pkl'
    if not os.path.exists(pkl_path):
        logger.error(f'{pkl_path} not found. Run train.py first.')
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        test_dfs: dict = pickle.load(f)

    logger.info(f'Running backtest on {len(test_dfs)} symbols…')

    engine  = BacktestEngine(model, device, trade_cfg)
    results = engine.run(test_dfs)

    stats  = results['stats']
    eq     = results['equity_curve']
    trades = results['trades']

    bh_eq  = buy_and_hold_equity(test_dfs, trade_cfg.initial_capital)
    bh_ret = float(bh_eq[-1] / bh_eq[0] - 1) if len(bh_eq) > 1 else 0.0

    print('\n' + '=' * 60)
    print('  BACKTEST RESULTS')
    print('=' * 60)
    print(f"  Initial capital   : ${trade_cfg.initial_capital:>12,.2f}")
    print(f"  Final equity      : ${stats['final_equity']:>12,.2f}")
    print(f"  Total return      : {stats['total_return']:>12.2%}  "
          f"(B&H BTC: {bh_ret:+.2%})")
    print(f"  CAGR              : {stats['cagr']:>12.2%}")
    print(f"  Sharpe ratio      : {stats['sharpe']:>12.3f}")
    print(f"  Sortino ratio     : {stats['sortino']:>12.3f}")
    print(f"  Max drawdown      : {stats['max_drawdown']:>12.2%}")
    print(f"  Calmar ratio      : {stats['calmar']:>12.3f}")
    print(f"  # trades          : {stats['n_trades']:>12,}")
    print(f"  Win rate          : {stats['win_rate']:>12.2%}")
    print(f"  Profit factor     : {stats['profit_factor']:>12.3f}")
    print(f"  IC (1h)           : {results['ic_1h']:>12.4f}")
    print(f"  IC (24h)          : {results['ic_24h']:>12.4f}")
    print('=' * 60)

    if trades:
        trade_df = pd.DataFrame(trades)
        print('\nTop 5 trades by PnL:')
        print(trade_df.nlargest(5, 'pnl')[['symbol', 'direction', 'entry_price',
                                            'exit_price', 'pnl', 'exit_reason']].to_string(index=False))

    if args.plot or args.output:
        _plot(eq, trade_cfg.initial_capital, args.plot, args.output)


def _plot(eq: np.ndarray, initial: float, show: bool, save_path: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        subplot_titles=('Portfolio Equity', 'Drawdown'),
    )

    x = list(range(len(eq)))

    fig.add_trace(go.Scatter(x=x, y=eq, name='Strategy',
                             line=dict(color='#00d4aa', width=2)), row=1, col=1)
    fig.add_hline(y=initial, line_dash='dash', line_color='gray', row=1, col=1)

    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak == 0, 1, peak) * 100
    fig.add_trace(go.Scatter(x=x, y=dd, name='Drawdown %',
                             fill='tozeroy', line=dict(color='#ff4d4d', width=1)),
                  row=2, col=1)

    fig.update_layout(
        title='Neural Network Crypto Trading Backtest',
        template='plotly_dark',
        height=600,
        showlegend=True,
    )
    fig.update_yaxes(title_text='USD', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown %', row=2, col=1)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.write_html(save_path)
        logger.info(f'Interactive chart saved → {save_path}')

    if show:
        fig.show()


if __name__ == '__main__':
    main()
