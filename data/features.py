"""
Feature engineering: computes ~40 technical indicators on raw OHLCV data,
plus 8 cross-market features derived from BTC and ETH as leading indicators.

All indicators are computed causally (no look-ahead).
"""

import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with columns [timestamp, open, high, low, close, volume]
    Output: DataFrame with all features, NaN rows dropped.
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    c, h, l, v, o = df['close'], df['high'], df['low'], df['volume'], df['open']

    # ── Price-based features ──────────────────────────────────────────────────
    df['log_return'] = np.log(c / c.shift(1))
    df['hl_range']   = (h - l) / c                         # High-low normalized range
    df['body']       = (c - o) / c                         # Candle body
    df['upper_wick'] = (h - c.clip(upper=o)) / c           # Upper shadow
    df['lower_wick'] = (c.clip(upper=o) - l) / c           # Lower shadow

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma20 = v.rolling(20).mean()
    df['vol_ratio']  = v / vol_ma20.replace(0, np.nan)     # Volume vs 20-period MA

    # ── Trend indicators ──────────────────────────────────────────────────────
    for span in [9, 21, 50, 200]:
        ema = c.ewm(span=span, adjust=False).mean()
        df[f'ema{span}_ratio'] = c / ema - 1               # % deviation from EMA

    df['sma20_ratio'] = c / c.rolling(20).mean() - 1
    df['sma50_ratio'] = c / c.rolling(50).mean() - 1

    # MACD
    macd_ind = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df['macd']        = macd_ind.macd() / c
    df['macd_signal'] = macd_ind.macd_signal() / c
    df['macd_hist']   = macd_ind.macd_diff() / c

    # ADX / DI
    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    df['adx']      = adx_ind.adx() / 100
    df['di_plus']  = adx_ind.adx_pos() / 100
    df['di_minus'] = adx_ind.adx_neg() / 100

    # ── Momentum indicators ───────────────────────────────────────────────────
    df['rsi14'] = ta.momentum.RSIIndicator(c, window=14).rsi() / 100
    df['rsi7']  = ta.momentum.RSIIndicator(c, window=7).rsi() / 100

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch() / 100
    df['stoch_d'] = stoch.stoch_signal() / 100

    df['williams_r'] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r() / -100

    df['roc10'] = ta.momentum.ROCIndicator(c, window=10).roc() / 100

    # ── Volatility indicators ─────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['bb_pct']   = bb.bollinger_pband()                  # Position within bands [0,1]

    df['atr14'] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c

    df['kc_pct'] = ta.volatility.KeltnerChannel(h, l, c, window=20).keltner_channel_pband()

    # ── Volume / money-flow indicators ────────────────────────────────────────
    df['mfi14'] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index() / 100

    df['cmf20'] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()

    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_ma = obv.rolling(20).mean()
    df['obv_ratio'] = (obv - obv_ma) / (obv_ma.abs().replace(0, np.nan))

    # ── Time-of-day / day-of-week cyclical features ───────────────────────────
    dt = pd.to_datetime(df['timestamp'])
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow / 7)

    # ── Drop NaN rows (warm-up period from longest indicator, ~200 candles) ───
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


FEATURE_COLS = [
    # ── Per-coin features (35) ──────────────────────────────────────────────────
    'log_return', 'hl_range', 'body', 'upper_wick', 'lower_wick', 'vol_ratio',
    'ema9_ratio', 'ema21_ratio', 'ema50_ratio', 'ema200_ratio',
    'sma20_ratio', 'sma50_ratio',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'di_plus', 'di_minus',
    'rsi14', 'rsi7',
    'stoch_k', 'stoch_d',
    'williams_r', 'roc10',
    'bb_width', 'bb_pct',
    'atr14', 'kc_pct',
    'mfi14', 'cmf20', 'obv_ratio',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # ── Multi-timeframe features (6) — added by add_multitf_features() ──────────
    # 4-hour aggregated: capture intraday session momentum
    'rsi14_4h', 'atr14_4h', 'vol_ratio_4h', 'macd_signal_4h',
    # Daily aggregated: macro trend context
    'rsi14_1d', 'atr14_1d',
    # ── Market-context features (9) — added by add_market_features() ────────────
    'btc_ret_1h', 'btc_ret_4h', 'btc_rsi14', 'btc_adx',
    'btc_trend', 'btc_vol_spike', 'eth_ret_1h', 'btc_corr24',
    'eth_btc_ratio',     # ETH/ETH-BTC_MA - 1: altcoin risk-appetite signal
    # ── Market breadth (1) — added by add_market_breadth() in train.py ──────────
    'market_breadth',    # fraction of universe with positive 1h return [0, 1]
]


# ── Multi-timeframe feature injection ────────────────────────────────────────

def add_multitf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4-hour and daily aggregated technical features.

    Uses right-closed, right-labeled resampling so that a bar labeled at time t
    covers (t - period, t] — all constituent hourly bars are in the past at t.
    The resampled values are forward-filled to every hourly bar, ensuring strict
    causality: at hour t you see the most recently *completed* 4h / 1d bar.

    Adds 6 features:
      4h: rsi14_4h, atr14_4h, vol_ratio_4h, macd_signal_4h
      1d: rsi14_1d, atr14_1d
    """
    df = df.copy()
    # Build a DatetimeIndex version for resampling
    ts = pd.to_datetime(df['timestamp'])
    df_ts = df.set_index(ts).sort_index()

    ohlcv_agg = {'open': 'first', 'high': 'max', 'low': 'min',
                 'close': 'last', 'volume': 'sum'}

    def _resample_features(period: str, prefix: str) -> pd.DataFrame:
        # closed='right', label='right': bar labeled at t covers (t-period, t]
        rs = df_ts[['open', 'high', 'low', 'close', 'volume']].resample(
            period, closed='right', label='right'
        ).agg(ohlcv_agg).dropna()

        c, h, l, v = rs['close'], rs['high'], rs['low'], rs['volume']
        out = pd.DataFrame(index=rs.index)
        out[f'rsi14_{prefix}']       = ta.momentum.RSIIndicator(c, window=14).rsi() / 100
        out[f'atr14_{prefix}']       = (
            ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c
        )
        vol_ma = v.rolling(20).mean()
        out[f'vol_ratio_{prefix}']   = v / vol_ma.replace(0, np.nan)
        macd_obj = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        out[f'macd_signal_{prefix}'] = macd_obj.macd_signal() / c
        return out

    feats_4h = _resample_features('4h', '4h')
    feats_1d = _resample_features('1D', '1d')

    # Forward-fill resampled features onto every hourly bar
    hourly_idx = df_ts.index
    for feat_df, cols in [
        (feats_4h, ['rsi14_4h', 'atr14_4h', 'vol_ratio_4h', 'macd_signal_4h']),
        (feats_1d, ['rsi14_1d', 'atr14_1d']),
    ]:
        reindexed = feat_df[cols].reindex(hourly_idx, method='ffill')
        for col in cols:
            df_ts[col] = reindexed[col].values

    df_out = df_ts.reset_index(drop=True)
    df_out['timestamp'] = ts.values

    mt_cols = ['rsi14_4h', 'atr14_4h', 'vol_ratio_4h', 'macd_signal_4h',
               'rsi14_1d', 'atr14_1d']
    df_out[mt_cols] = df_out[mt_cols].ffill().fillna(0)
    return df_out


# ── Cross-market feature injection ────────────────────────────────────────────

def add_market_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge BTC and ETH market-context features into a per-coin DataFrame.

    BTC acts as the market leader — its momentum, trend, volatility, and
    RSI give every coin a read on macro regime without requiring the model
    to infer it from any single coin's own history.

    ETH's 1h return captures the large-cap altcoin direction independently
    of BTC (important when ETH/BTC diverges).

    btc_corr24 — rolling 24h Pearson correlation of the coin's returns with
    BTC — tells the model how "beta" this coin currently is to the market,
    helping it weight the market context appropriately.

    All joins are left joins on timestamp so no future data can leak.
    Gaps (exchange outages, weekend thinness) are forward-filled then zeroed.
    """
    df = df.copy()

    # ── BTC: select and rename columns we want ───────────────────────────────
    btc_sel = btc_df[['timestamp', 'log_return', 'rsi14', 'adx',
                       'ema200_ratio', 'vol_ratio']].copy()
    btc_sel = btc_sel.rename(columns={
        'log_return':   'btc_ret_1h',
        'rsi14':        'btc_rsi14',
        'adx':          'btc_adx',
        'ema200_ratio': 'btc_trend',      # positive = above 200h EMA (uptrend)
        'vol_ratio':    'btc_vol_spike',  # volume relative to 20-period MA
    })

    # BTC 4-hour return (shift on the BTC df before merging to avoid look-ahead)
    btc_close = btc_df.set_index('timestamp')['close']
    btc_ret_4h = np.log(btc_close / btc_close.shift(4)).rename('btc_ret_4h')
    btc_sel = btc_sel.merge(
        btc_ret_4h.reset_index(), on='timestamp', how='left'
    )

    # ── ETH: 1h return ───────────────────────────────────────────────────────
    eth_sel = eth_df[['timestamp', 'log_return']].copy()
    eth_sel = eth_sel.rename(columns={'log_return': 'eth_ret_1h'})

    # ── Merge into target coin df ─────────────────────────────────────────────
    df = df.merge(
        btc_sel[['timestamp', 'btc_ret_1h', 'btc_ret_4h',
                 'btc_rsi14', 'btc_adx', 'btc_trend', 'btc_vol_spike']],
        on='timestamp', how='left',
    )
    df = df.merge(
        eth_sel[['timestamp', 'eth_ret_1h']],
        on='timestamp', how='left',
    )

    # ── Rolling 24h correlation with BTC ─────────────────────────────────────
    # Uses only past data (rolling window closes at current bar) — causal.
    df['btc_corr24'] = (
        df['log_return'].rolling(24).corr(df['btc_ret_1h'])
    )

    # ── ETH/BTC ratio: altcoin risk-appetite indicator ────────────────────────
    # When ETH outperforms BTC, alt-season dynamics are usually in play.
    # Expressed as deviation from the rolling 168h mean of the ratio.
    btc_close = btc_df[['timestamp', 'close']].rename(columns={'close': '_btc_c'})
    eth_close = eth_df[['timestamp', 'close']].rename(columns={'close': '_eth_c'})
    df = df.merge(btc_close, on='timestamp', how='left')
    df = df.merge(eth_close, on='timestamp', how='left')
    eth_btc_raw  = df['_eth_c'] / df['_btc_c']
    eth_btc_ma   = eth_btc_raw.rolling(168, min_periods=1).mean()
    df['eth_btc_ratio'] = (eth_btc_raw / eth_btc_ma - 1)
    df = df.drop(columns=['_btc_c', '_eth_c'])

    # ── Fill gaps ─────────────────────────────────────────────────────────────
    mkt_cols = ['btc_ret_1h', 'btc_ret_4h', 'btc_rsi14', 'btc_adx',
                'btc_trend', 'btc_vol_spike', 'eth_ret_1h', 'btc_corr24',
                'eth_btc_ratio']
    df[mkt_cols] = df[mkt_cols].ffill().fillna(0)

    return df


def add_market_breadth(
    symbol_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Compute market breadth = fraction of all symbols with a positive 1h return
    at each timestamp, then add it as 'market_breadth' to every DataFrame.

    Must be called AFTER all per-symbol features are computed (needs log_return).
    Returns the same dict with the new column added to each DataFrame.
    """
    # Collect log_return series indexed by timestamp for all symbols
    ret_map = {}
    for sym, df in symbol_dfs.items():
        s = df.set_index('timestamp')['log_return']
        ret_map[sym] = s

    breadth_df = pd.DataFrame(ret_map)
    # At each timestamp: fraction of symbols with return > 0
    breadth = (breadth_df > 0).mean(axis=1).rename('market_breadth')

    out = {}
    for sym, df in symbol_dfs.items():
        df2 = df.merge(breadth.reset_index(), on='timestamp', how='left')
        df2['market_breadth'] = df2['market_breadth'].ffill().fillna(0.5)
        out[sym] = df2

    return out


def make_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Compute future log-returns at each horizon as regression targets.
    Rows where any target is NaN (end of series) are dropped.
    """
    df = df.copy()
    for h in horizons:
        df[f'target_{h}h'] = np.log(df['close'].shift(-h) / df['close'])
    df = df.dropna(subset=[f'target_{h}h' for h in horizons]).reset_index(drop=True)
    return df


TARGET_COLS = ['target_1h', 'target_4h', 'target_12h', 'target_24h']
