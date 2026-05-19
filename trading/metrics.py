"""
Performance metrics for the backtest engine.
All metrics are annualised to 365-day crypto calendar.
"""

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 365 * 24


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Annualised Sortino ratio (downside std only)."""
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-9
    if downside_std == 0:
        return 0.0
    return float((returns.mean() / downside_std) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction (0 → 1)."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak == 0, 1, peak)
    return float(drawdown.min())


def calmar_ratio(equity: np.ndarray, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Annualised return / |max drawdown|."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    total_periods = len(equity)
    total_return  = equity[-1] / equity[0] - 1
    annual_return = (1 + total_return) ** (periods_per_year / total_periods) - 1
    return float(annual_return / mdd)


def cagr(equity: np.ndarray, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Compound annual growth rate."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    total_return  = equity[-1] / equity[0]
    annual_factor = periods_per_year / len(equity)
    return float(total_return ** annual_factor - 1)


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return wins / len(trades)


def profit_factor(trades: list[dict]) -> float:
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss   = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    if gross_loss == 0:
        return float('inf')
    return gross_profit / gross_loss


def summarise(equity_curve: np.ndarray, trades: list[dict]) -> dict:
    returns = np.diff(equity_curve) / equity_curve[:-1]
    return {
        'final_equity':   float(equity_curve[-1]),
        'total_return':   float(equity_curve[-1] / equity_curve[0] - 1),
        'cagr':           cagr(equity_curve),
        'sharpe':         sharpe_ratio(returns),
        'sortino':        sortino_ratio(returns),
        'max_drawdown':   max_drawdown(equity_curve),
        'calmar':         calmar_ratio(equity_curve),
        'n_trades':       len(trades),
        'win_rate':       win_rate(trades),
        'profit_factor':  profit_factor(trades),
    }
