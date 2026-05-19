"""
OHLCV data collector using publicly accessible REST APIs — no API key required.

Source priority per symbol:
  1. CryptoCompare  — supports all major coins, up to 2000 candles/call, global access
  2. Kraken direct  — fallback for CryptoCompare failures or missing symbols
"""

import os
import time
import logging
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_CRYPTOCOMPARE_BASE = 'https://min-api.cryptocompare.com/data/v2/histohour'
_KRAKEN_BASE        = 'https://api.kraken.com/0/public/OHLC'

# CryptoCompare: derive fsym/tsym automatically from any "BASE/QUOTE" symbol.
# Override only the handful of coins whose CC ticker differs from convention.
_CC_OVERRIDES: dict[str, tuple[str, str]] = {
    # none currently needed — all major coins use their standard ticker on CC
}

def _cc_symbol(symbol: str) -> tuple[str, str] | None:
    """Return (fsym, tsym) for CryptoCompare, or None if not a supported format."""
    if '/' not in symbol:
        return None
    base, quote = symbol.split('/', 1)
    return _CC_OVERRIDES.get(symbol, (base, quote))


# Kraken uses non-standard tickers for a few coins; map only those.
_KRAKEN_TICKER: dict[str, str] = {
    'BTC': 'XBT',   # Kraken calls Bitcoin XBT
}

def _kraken_pair(symbol: str) -> str | None:
    """Return Kraken pair string, e.g. 'XBTUSDT', or None if unsupported format."""
    if '/' not in symbol:
        return None
    base, quote = symbol.split('/', 1)
    kraken_base = _KRAKEN_TICKER.get(base, base)
    return f'{kraken_base}{quote}'


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get_json(url: str, retries: int = 3) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            logger.debug(f'HTTP {e.code} for {url}')
            return None
        except Exception as e:
            wait = 3 * attempt
            logger.debug(f'Request error (attempt {attempt}/{retries}): {e}. Waiting {wait}s.')
            if attempt < retries:
                time.sleep(wait)
    return None


# ── CryptoCompare source ──────────────────────────────────────────────────────

def _fetch_cryptocompare(
    symbol: str,
    since_days: int,
) -> list[dict]:
    """Paginate CryptoCompare histohour backwards until since_days covered."""
    cc = _cc_symbol(symbol)
    if cc is None:
        logger.debug(f'CryptoCompare: cannot parse symbol {symbol}')
        return []
    fsym, tsym = cc
    since_ts = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp())

    all_rows: list[dict] = []
    to_ts = None   # None → use current time on first call

    while True:
        url = f'{_CRYPTOCOMPARE_BASE}?fsym={fsym}&tsym={tsym}&limit=2000'
        if to_ts is not None:
            url += f'&toTs={to_ts}'

        data = _get_json(url)
        if data is None or data.get('Response') != 'Success':
            logger.debug(f'CryptoCompare bad response for {symbol}: {data}')
            break

        candles = data['Data']['Data']
        if not candles:
            break

        # Filter rows within our window
        valid = [c for c in candles if c['time'] >= since_ts and c['volumeto'] > 0]
        all_rows.extend(valid)

        earliest = candles[0]['time']
        if earliest <= since_ts:
            break   # covered the full window

        to_ts = earliest - 1
        time.sleep(0.3)   # polite pacing

    return all_rows


def _cryptocompare_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={'time': 'timestamp', 'volumefrom': 'volume'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


# ── Kraken source ─────────────────────────────────────────────────────────────

def _fetch_kraken(symbol: str, since_days: int) -> list:
    """Paginate Kraken OHLC (max 720 rows/call) until since_days covered."""
    pair = _kraken_pair(symbol)
    if pair is None:
        logger.debug(f'Kraken: cannot parse symbol {symbol}')
        return []
    since_ts = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp())

    all_candles: list = []
    cursor = since_ts

    while True:
        url = f'{_KRAKEN_BASE}?pair={pair}&interval=60&since={cursor}'
        data = _get_json(url)
        if data is None or data.get('error'):
            logger.debug(f'Kraken error for {symbol}: {data}')
            break

        result = data.get('result', {})
        candles = result.get(pair, result.get(list(result.keys() - {'last'})[0] if result.keys() - {'last'} else pair, []))
        if not candles:
            break

        all_candles.extend(candles)
        last = data['result'].get('last', 0)
        if not last or last <= cursor:
            break
        cursor = last
        time.sleep(0.5)

    return all_candles


def _kraken_to_df(candles: list) -> pd.DataFrame:
    # Kraken format: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','vwap','volume','count'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
    df = df[['timestamp','open','high','low','close','volume']].astype(
        {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    )
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    timeframe: str = '1h',
    since_days: int = 730,
    exchanges: list[str] | None = None,   # unused, kept for API compat
    save_dir: str = 'data/raw',
) -> pd.DataFrame:
    """Fetch OHLCV for one symbol, trying CryptoCompare then Kraken."""
    os.makedirs(save_dir, exist_ok=True)
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(save_dir, f'{safe_symbol}_{timeframe}.csv')

    # ── Try CryptoCompare ─────────────────────────────────────────────────────
    logger.info(f'  CryptoCompare → {symbol}…')
    rows = _fetch_cryptocompare(symbol, since_days)
    if rows:
        df = _cryptocompare_to_df(rows)
        min_rows = min(200, since_days * 12)   # at least 50% of expected rows
        if len(df) >= min_rows:
            df.to_csv(filepath, index=False)
            logger.info(f'  CryptoCompare: {len(df):,} rows for {symbol} → {filepath}')
            return df
        logger.debug(f'  CryptoCompare: only {len(df)} rows — trying Kraken')

    # ── Try Kraken ────────────────────────────────────────────────────────────
    logger.info(f'  Kraken → {symbol}…')
    candles = _fetch_kraken(symbol, since_days)
    if candles:
        df = _kraken_to_df(candles)
        min_rows = min(200, since_days * 12)
        if len(df) >= min_rows:
            df.to_csv(filepath, index=False)
            logger.info(f'  Kraken: {len(df):,} rows for {symbol} → {filepath}')
            return df

    logger.warning(f'  No data retrieved for {symbol}')
    return pd.DataFrame()


def load_or_fetch(
    symbol: str,
    timeframe: str = '1h',
    since_days: int = 730,
    exchanges: list[str] | None = None,
    save_dir: str = 'data/raw',
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached CSV if present, otherwise fetch."""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(save_dir, f'{safe_symbol}_{timeframe}.csv')

    if not force_refresh and os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        logger.info(f'  Loaded {len(df):,} cached rows for {symbol}')
        return df

    return fetch_ohlcv(symbol, timeframe, since_days, exchanges, save_dir)


def fetch_all_symbols(
    symbols: list[str],
    timeframe: str = '1h',
    since_days: int = 730,
    exchange_id: str | None = None,   # ignored, kept for API compat
    exchanges: list[str] | None = None,
    save_dir: str = 'data/raw',
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all symbols. Returns {symbol: DataFrame}."""
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        logger.info(f'Fetching {symbol}…')
        try:
            df = load_or_fetch(symbol, timeframe, since_days, exchanges, save_dir, force_refresh)
            if not df.empty:
                data[symbol] = df
        except Exception as e:
            logger.error(f'  Unexpected error for {symbol}: {e}')
    logger.info(f'Fetched data for {len(data)}/{len(symbols)} symbols')
    return data
