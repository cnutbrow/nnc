"""
Event-driven backtest engine.

Simulates a portfolio trading multiple assets on 1-hour bars with:
  - Transaction costs (taker fee + slippage)
  - Stop-loss and take-profit orders
  - Max concurrent open positions
  - Kelly-based position sizing

Usage:
    engine = BacktestEngine(model, device, cfg_trading)
    results = engine.run(symbol_dfs)   # {symbol: featured_df with targets}
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from data.features import FEATURE_COLS, TARGET_COLS, compute_features, make_targets
from model.dataset import CryptoSequenceDataset
from trading.strategy import Signal, generate_signal, kelly_position_size, predict_batch
from trading.metrics import summarise

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: int          # +1 long, -1 short
    entry_price: float
    entry_idx: int
    units: float            # positive quantity
    stop_loss: float
    take_profit: float
    entry_cost: float       # total transaction cost paid on entry


@dataclass
class ClosedTrade:
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    units: float
    entry_idx: int
    exit_idx: int
    pnl: float
    exit_reason: str


class BacktestEngine:
    def __init__(self, model, device, cfg):
        self.model  = model
        self.device = device
        self.cfg    = cfg

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, symbol_dfs: dict[str, pd.DataFrame]) -> dict:
        """
        symbol_dfs: {symbol: raw OHLCV DataFrame (already feature-engineered + targets)}
        Returns dict with equity_curve, trades, per_symbol stats.
        """
        # Build per-symbol price arrays and prediction arrays
        # Assign stable coin indices matching the order used during training
        symbol_list = list(symbol_dfs.keys())
        coin_index  = {sym: i for i, sym in enumerate(symbol_list)}

        sym_data: dict[str, dict] = {}
        for symbol, df in symbol_dfs.items():
            arr = self._make_predictions(df, symbol, coin_idx=coin_index[symbol])
            if arr is not None:
                sym_data[symbol] = arr

        if not sym_data:
            raise RuntimeError('No valid symbol data for backtesting')

        # ── Information coefficient (cross-sectional Spearman IC) ─────────────
        # Computed on the full test-set predictions before simulation so it
        # reflects pure model quality independent of the trading rules.
        try:
            from scipy.stats import spearmanr as _spearmanr
            _all_p = np.concatenate([d['preds'][:, 0]   for d in sym_data.values()])
            _all_t = np.vstack([d['targets'] for d in sym_data.values()])
            # IC: Spearman(P(up), actual 1h return) and Spearman(P(up), actual 24h return)
            ic_1h  = float(_spearmanr(_all_p, _all_t[:, 0]).statistic)
            ic_24h = float(_spearmanr(_all_p, _all_t[:, -1]).statistic)
            logger.info(f'IC(1h): {ic_1h:.4f}   IC(24h): {ic_24h:.4f}')
        except Exception:
            ic_1h = ic_24h = float('nan')

        # Align all symbols to common time index
        timestamps = sorted({ts for d in sym_data.values() for ts in d['timestamps']})
        result = self._simulate(sym_data, timestamps)
        result['ic_1h']  = ic_1h
        result['ic_24h'] = ic_24h
        return result

    # ── Build predictions for one symbol ─────────────────────────────────────

    def _make_predictions(self, df: pd.DataFrame, symbol: str,
                          coin_idx: int = 0) -> Optional[dict]:
        from config import FeatureConfig
        seq_len = FeatureConfig().sequence_length

        ds = CryptoSequenceDataset(df, seq_len=seq_len, coin_idx=coin_idx)
        if len(ds) == 0:
            logger.warning(f'Skipping {symbol}: insufficient data')
            return None

        loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        all_preds = []
        for batch in loader:
            x, coin, _ = batch   # (x, coin_idx, target)
            preds = predict_batch(self.model, x, coin_idx=coin)
            all_preds.append(preds)

        preds_arr = np.vstack(all_preds)   # (N, 4)

        # Align preds to df rows.
        # ds.start_offset = max(seq_len, norm_window) — the first sample's label row
        # in df is at index start_offset.  See model/dataset.py for the matching
        # slice convention in build_datasets.
        offset = ds.start_offset
        aligned_df = df.iloc[offset: offset + len(preds_arr)].reset_index(drop=True)

        if len(aligned_df) != len(preds_arr):
            min_len = min(len(aligned_df), len(preds_arr))
            aligned_df  = aligned_df.iloc[:min_len]
            preds_arr   = preds_arr[:min_len]

        return {
            'timestamps': aligned_df['timestamp'].tolist(),
            'closes':     aligned_df['close'].values,
            'preds':      preds_arr,
            'df':         aligned_df,
            'targets':    aligned_df[TARGET_COLS].values,   # actual returns for IC
        }

    # ── Core simulation loop ──────────────────────────────────────────────────

    def _simulate(self, sym_data: dict[str, dict], timestamps: list) -> dict:
        cfg = self.cfg
        cash          = cfg.initial_capital
        positions: dict[str, Position] = {}
        closed_trades: list[ClosedTrade] = []
        equity_curve  = [cash]

        # Build lookup: symbol → {timestamp → (close, pred)}
        lookup: dict[str, dict] = {}
        for sym, d in sym_data.items():
            lookup[sym] = {
                ts: (close, pred)
                for ts, close, pred in zip(d['timestamps'], d['closes'], d['preds'])
            }

        # ── Regime filter: index BTC's trend + ADX by timestamp ──────────────
        # btc_trend  = BTC ema200_ratio: >0 means price above 200h EMA (uptrend)
        # btc_adx    = BTC ADX / 100: >0.20 means a real directional trend exists
        # These columns are already in the BTC feature DataFrame (FEATURE_COLS).
        btc_regime: dict = {}   # ts → (btc_trend, btc_adx)
        if cfg.use_regime_filter and 'BTC/USDT' in sym_data:
            btc_d = sym_data['BTC/USDT']
            btc_df_regime = btc_d['df']
            if 'btc_trend' in btc_df_regime.columns and 'btc_adx' in btc_df_regime.columns:
                for ts, trend, adx_val in zip(
                    btc_d['timestamps'],
                    btc_df_regime['btc_trend'].values,
                    btc_df_regime['btc_adx'].values,
                ):
                    btc_regime[ts] = (float(trend), float(adx_val))

        portfolio_value = cash

        for ts in timestamps:
            # ── Mark-to-market all open positions ─────────────────────────────
            unrealised = 0.0
            for sym, pos in list(positions.items()):
                if ts not in lookup[sym]:
                    continue
                price, _ = lookup[sym][ts]
                unrealised += pos.direction * pos.units * (price - pos.entry_price)

            portfolio_value = cash + unrealised

            # ── Check stop-loss / take-profit ─────────────────────────────────
            to_close: list[tuple[str, str]] = []
            for sym, pos in positions.items():
                if ts not in lookup[sym]:
                    continue
                price, _ = lookup[sym][ts]

                if pos.direction == 1:
                    if price <= pos.stop_loss:
                        to_close.append((sym, 'stop_loss'))
                    elif price >= pos.take_profit:
                        to_close.append((sym, 'take_profit'))
                else:
                    if price >= pos.stop_loss:
                        to_close.append((sym, 'stop_loss'))
                    elif price <= pos.take_profit:
                        to_close.append((sym, 'take_profit'))

            for sym, reason in to_close:
                if sym not in lookup or ts not in lookup[sym]:
                    continue
                price, _ = lookup[sym][ts]
                cash = self._close_position(sym, price, reason, positions, closed_trades, cash)

            # ── Generate new signals ──────────────────────────────────────────
            for sym, d in sym_data.items():
                if ts not in lookup[sym]:
                    continue
                price, pred = lookup[sym][ts]
                signal = generate_signal(pred, cfg)

                # Close opposing position
                if sym in positions and positions[sym].direction != signal.direction:
                    cash = self._close_position(sym, price, 'signal_reversal',
                                                positions, closed_trades, cash)

                # ── Regime filter ─────────────────────────────────────────────
                # Only open when BTC has a real trend (ADX threshold) and the
                # signal direction isn't fighting a severe BTC macro move.
                if signal.direction != 0 and btc_regime:
                    trend_val, adx_val = btc_regime.get(ts, (0.0, 1.0))
                    # Skip if market is ranging / directionless
                    if adx_val < cfg.regime_adx_min:
                        signal = Signal(direction=0, strength=0.0, raw_pred=signal.raw_pred)
                    # Skip longs when BTC is in severe downtrend
                    elif signal.direction == 1 and trend_val < cfg.regime_trend_min:
                        signal = Signal(direction=0, strength=0.0, raw_pred=signal.raw_pred)
                    # Skip shorts when BTC is in severe uptrend
                    elif signal.direction == -1 and trend_val > -cfg.regime_trend_min:
                        pass   # shorts are fine in neutral-to-down market
                    # (Don't suppress shorts in uptrend — model may be right)

                # Open new position
                if (signal.direction != 0
                        and sym not in positions
                        and len(positions) < cfg.max_open_positions):
                    volatility = float(np.std(d['preds'][:, 0]) + 1e-9)
                    units = kelly_position_size(
                        signal.strength, portfolio_value, price, cfg, volatility
                    )
                    if units * price < 1.0:   # skip micro-trades
                        continue

                    cost = units * price * (cfg.transaction_cost + cfg.slippage)
                    cash -= cost

                    if signal.direction == 1:
                        sl = price * (1 - cfg.stop_loss_pct)
                        tp = price * (1 + cfg.take_profit_pct)
                    else:
                        sl = price * (1 + cfg.stop_loss_pct)
                        tp = price * (1 - cfg.take_profit_pct)

                    positions[sym] = Position(
                        symbol=sym,
                        direction=signal.direction,
                        entry_price=price,
                        entry_idx=0,
                        units=units,
                        stop_loss=sl,
                        take_profit=tp,
                        entry_cost=cost,
                    )

            equity_curve.append(portfolio_value)

        # Close all remaining positions at last available price
        for sym in list(positions.keys()):
            if sym in lookup:
                last_ts = max(lookup[sym].keys(), key=lambda t: t)
                price, _ = lookup[sym][last_ts]
                cash = self._close_position(sym, price, 'end_of_backtest',
                                            positions, closed_trades, cash)

        equity_curve.append(cash)
        eq = np.array(equity_curve, dtype=np.float64)
        stats = summarise(eq, [t.__dict__ for t in closed_trades])

        return {
            'equity_curve': eq,
            'trades':       [t.__dict__ for t in closed_trades],
            'stats':        stats,
        }

    def _close_position(
        self,
        symbol: str,
        price: float,
        reason: str,
        positions: dict,
        closed_trades: list,
        cash: float,
    ) -> float:
        pos  = positions.pop(symbol)
        cfg  = self.cfg
        cost = pos.units * price * (cfg.transaction_cost + cfg.slippage)
        proceeds = pos.direction * pos.units * (price - pos.entry_price) - cost - pos.entry_cost
        cash += proceeds   # realised PnL; capital was never deducted on open (margin model)

        closed_trades.append(ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            units=pos.units,
            entry_idx=pos.entry_idx,
            exit_idx=0,
            pnl=proceeds,
            exit_reason=reason,
        ))
        return cash
