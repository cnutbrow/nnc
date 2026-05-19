"""
Fetch and cache perpetual futures derivatives data:
  - Funding rates  (every 8 hours, forward-filled to 1h)
  - Open interest  (hourly)

Data sources: Bybit (primary) → OKX (fallback).
Both are free, no API key required.

Funding rate interpretation:
  positive → longs pay shorts → market over-leveraged long → bearish mean-reversion signal
  negative → shorts pay longs → market crowded short → bullish contrarian signal
  extremes (|rate| > 0.05%) historically precede sharp reversals.

Open interest interpretation:
  rising OI + rising price  → trend continuation (new money entering long)
  rising OI + falling price → capitulation risk (leveraged shorts building)
  falling OI               → de-leveraging (trend exhaustion)
"""

import logging
import os
import time
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DERIV_EXCHANGES = ['bybit', 'okx']
_PAGE            = 200


def _spot_to_perp(symbol: str) -> str:
    """'BTC/USDT' → 'BTC/USDT:USDT'  (CCXT unified perpetual format)"""
    base, quote = symbol.split('/')
    return f'{base}/{quote}:{quote}'


def _exchange(name: str):
    cls = getattr(ccxt, name)
    return cls({'enableRateLimit': True})


def _fetch_funding(exch, perp_sym: str, since_ms: int) -> pd.DataFrame:
    rows, since = [], since_ms
    while True:
        try:
            batch = exch.fetch_funding_rate_history(perp_sym, since=since, limit=_PAGE)
        except Exception as e:
            logger.debug(f'{exch.id} funding {perp_sym}: {e}')
            break
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < _PAGE:
            break
        since = batch[-1]['timestamp'] + 1
        time.sleep(exch.rateLimit / 1000)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([
        {'timestamp': r['timestamp'], 'funding_rate': float(r['fundingRate'])}
        for r in rows if r.get('fundingRate') is not None
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def _fetch_oi(exch, perp_sym: str, since_ms: int) -> pd.DataFrame:
    rows, since = [], since_ms
    while True:
        try:
            batch = exch.fetch_open_interest_history(perp_sym, '1h', since=since, limit=_PAGE)
        except Exception as e:
            logger.debug(f'{exch.id} OI {perp_sym}: {e}')
            break
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < _PAGE:
            break
        since = batch[-1]['timestamp'] + 1
        time.sleep(exch.rateLimit / 1000)

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        ts = r.get('timestamp')
        # prefer USD-denominated value; fall back to base-currency amount
        oi = r.get('openInterestValue') or r.get('openInterestAmount')
        if ts and oi:
            records.append({'timestamp': ts, 'oi_usd': float(oi)})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def fetch_symbol_derivatives(
    symbol: str,
    since_days: int = 400,
    cache_dir: str = 'data/funding',
) -> Optional[pd.DataFrame]:
    """
    Return an hourly DataFrame with columns [timestamp, funding_rate, oi_usd]
    for the given spot symbol, or None if no perpetual market exists.

    Results are cached to CSV; cache is reused if younger than 2 hours.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe = symbol.replace('/', '_').replace(':', '_')
    cache = os.path.join(cache_dir, f'{safe}.csv')

    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < 7200:
        df = pd.read_csv(cache, parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df

    since_ms  = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=since_days)).timestamp() * 1000)
    perp_sym  = _spot_to_perp(symbol)
    funding   = pd.DataFrame()
    oi        = pd.DataFrame()

    for name in _DERIV_EXCHANGES:
        if not funding.empty and not oi.empty:
            break
        exch = _exchange(name)
        logger.info(f'  {symbol}: derivatives ← {name}')
        if funding.empty:
            funding = _fetch_funding(exch, perp_sym, since_ms)
        if oi.empty:
            oi = _fetch_oi(exch, perp_sym, since_ms)

    if funding.empty and oi.empty:
        logger.warning(f'  {symbol}: no perpetual market found — skipping derivatives')
        return None

    # Resample funding (8h) → 1h via forward-fill
    if not funding.empty:
        funding = (funding.set_index('timestamp')
                         .resample('1h').last()
                         .ffill())
    # OI is already ~1h; ensure regularity
    if not oi.empty:
        oi = (oi.set_index('timestamp')
                .resample('1h').last()
                .ffill())

    if not funding.empty and not oi.empty:
        result = funding.join(oi, how='outer').ffill()
    elif not funding.empty:
        result = funding
        result['oi_usd'] = np.nan
    else:
        result = oi
        result['funding_rate'] = np.nan

    result = result.reset_index()
    result.to_csv(cache, index=False)
    return result


def fetch_all_derivatives(
    symbols: list[str],
    since_days: int = 400,
    cache_dir: str = 'data/funding',
) -> dict[str, pd.DataFrame]:
    """Fetch derivatives for every symbol. Returns {symbol: hourly DataFrame}."""
    out = {}
    for sym in symbols:
        df = fetch_symbol_derivatives(sym, since_days, cache_dir)
        if df is not None and not df.empty:
            out[sym] = df
        time.sleep(0.05)
    logger.info(f'Derivatives available for {len(out)}/{len(symbols)} symbols')
    return out
