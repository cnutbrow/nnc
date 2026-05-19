#!/usr/bin/env python3
"""
Rebuild test_dfs.pkl and model_meta.pkl from cached OHLCV data.

Run this after a training run was interrupted before the pickle files
were written, or after feature engineering changes that require the
test DataFrames to be re-generated.

Does NOT retrain the model — just re-runs feature engineering and
re-splits the data identically to train.py.

Usage:
    python rebuild_test_data.py
"""

import logging
import os
import pickle
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    from config import DataConfig, FeatureConfig, TrainingConfig
    from data.collector import fetch_all_symbols
    from data.features import (compute_features, add_multitf_features,
                               add_market_features, add_market_breadth,
                               make_targets, FEATURE_COLS)
    from model.dataset import build_datasets

    data_cfg  = DataConfig()
    feat_cfg  = FeatureConfig()
    train_cfg = TrainingConfig()

    # ── Load from cache (no network requests) ────────────────────────────────
    logger.info(f'Loading cached data for {len(data_cfg.symbols)} symbols…')
    raw_data = fetch_all_symbols(
        data_cfg.symbols,
        timeframe=data_cfg.timeframe,
        since_days=data_cfg.since_days,
        exchanges=data_cfg.exchanges,
        save_dir=data_cfg.data_dir,
        force_refresh=False,   # use cache only
    )
    logger.info(f'Loaded {len(raw_data)} symbols from cache')

    if not raw_data:
        logger.error('No cached data found. Run train.py --force-refresh first.')
        sys.exit(1)

    # ── Feature engineering (mirrors train.py exactly) ────────────────────────
    logger.info('Engineering features…')

    btc_feat_df = None
    eth_feat_df = None
    for sym, anchor in [('BTC/USDT', 'btc'), ('ETH/USDT', 'eth')]:
        if sym in raw_data:
            try:
                _f = compute_features(raw_data[sym])
                _f = add_multitf_features(_f)
                if anchor == 'btc':
                    btc_feat_df = _f
                else:
                    eth_feat_df = _f
            except Exception as e:
                logger.warning(f'  {sym}: feature engineering failed — {e}')

    have_market_ctx = btc_feat_df is not None and eth_feat_df is not None
    if not have_market_ctx:
        logger.warning('BTC or ETH unavailable — market-context features will be zero-filled')

    featured: dict = {}
    for symbol, df in raw_data.items():
        try:
            feat_df = compute_features(df)
            feat_df = add_multitf_features(feat_df)
            if have_market_ctx:
                feat_df = add_market_features(feat_df, btc_feat_df, eth_feat_df)
            else:
                for col in ['btc_ret_1h', 'btc_ret_4h', 'btc_rsi14', 'btc_adx',
                            'btc_trend', 'btc_vol_spike', 'eth_ret_1h', 'btc_corr24',
                            'eth_btc_ratio']:
                    feat_df[col] = 0.0
            feat_df = make_targets(feat_df, feat_cfg.target_horizons)
            if len(feat_df) > feat_cfg.sequence_length * 4:
                featured[symbol] = feat_df
                logger.info(f'  {symbol}: {len(feat_df)} rows, {len(FEATURE_COLS)} features')
        except Exception as e:
            logger.warning(f'  {symbol}: feature engineering failed — {e}')

    logger.info('Computing market breadth…')
    featured = add_market_breadth(featured)

    if not featured:
        logger.error('No valid feature data.')
        sys.exit(1)

    # ── Rebuild datasets (same split ratios as training) ──────────────────────
    logger.info('Rebuilding datasets…')
    _, _, _, test_dfs, num_coins = build_datasets(
        featured,
        seq_len=feat_cfg.sequence_length,
        train_ratio=train_cfg.train_ratio,
        val_ratio=train_cfg.val_ratio,
    )
    logger.info(f'Test set: {len(test_dfs)} symbols, {num_coins} coins total')

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    with open('models/test_dfs.pkl', 'wb') as f:
        pickle.dump(test_dfs, f)
    with open('models/model_meta.pkl', 'wb') as f:
        pickle.dump({'num_coins': num_coins}, f)

    logger.info('Saved models/test_dfs.pkl and models/model_meta.pkl')
    logger.info('You can now run: python run_backtest.py --plot')


if __name__ == '__main__':
    main()
