#!/usr/bin/env python3
"""
Main training script.

Steps:
  1. Download / load OHLCV data for all configured symbols
  2. Engineer features + compute targets
  3. Build train / val / test datasets
  4. Train CryptoTransformer
  5. Report test-set metrics

Run:
    python train.py [--force-refresh] [--epochs N] [--batch-size N]
"""

import argparse
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force-refresh', action='store_true',
                   help='Re-download all OHLCV data even if cached')
    p.add_argument('--epochs',     type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr',         type=float, default=None)
    p.add_argument('--symbols',    nargs='+', default=None,
                   help='Subset of symbols to train on, e.g. BTC/USDT ETH/USDT')
    return p.parse_args()


def main():
    args = parse_args()

    from config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
    from data.collector import fetch_all_symbols
    from data.features import (compute_features, add_multitf_features,
                               add_market_features, add_market_breadth,
                               make_targets, FEATURE_COLS, TARGET_COLS)
    from model.architecture import CryptoTransformer
    from model.dataset import build_datasets
    from training.trainer import train

    data_cfg  = DataConfig()
    feat_cfg  = FeatureConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if args.epochs:     train_cfg.epochs     = args.epochs
    if args.batch_size: train_cfg.batch_size = args.batch_size
    if args.lr:         train_cfg.learning_rate = args.lr
    if args.symbols:    data_cfg.symbols     = args.symbols

    # ── 1. Data download ──────────────────────────────────────────────────────
    logger.info(f'Fetching data for {len(data_cfg.symbols)} symbols…')
    raw_data = fetch_all_symbols(
        data_cfg.symbols,
        timeframe=data_cfg.timeframe,
        since_days=data_cfg.since_days,
        exchanges=data_cfg.exchanges,
        save_dir=data_cfg.data_dir,
        force_refresh=args.force_refresh,
    )
    logger.info(f'Got data for {len(raw_data)} symbols')

    if not raw_data:
        logger.error('No data retrieved — check your internet connection.')
        sys.exit(1)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    logger.info('Engineering features…')

    # Pre-compute BTC and ETH features — used as market context for every coin.
    # If either is missing, market features fall back to zeros (graceful degradation).
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
                logger.warning(f'  {sym}: base feature engineering failed — {e}')

    have_market_ctx = btc_feat_df is not None and eth_feat_df is not None
    if not have_market_ctx:
        logger.warning('BTC or ETH data unavailable — market-context features will be zero-filled')

    featured: dict = {}
    for symbol, df in raw_data.items():
        try:
            feat_df = compute_features(df)
            feat_df = add_multitf_features(feat_df)         # +6 multi-timeframe features
            if have_market_ctx:
                feat_df = add_market_features(feat_df, btc_feat_df, eth_feat_df)
            else:
                # Zero-fill all market-context columns so FEATURE_COLS stays consistent
                for col in ['btc_ret_1h', 'btc_ret_4h', 'btc_rsi14', 'btc_adx',
                            'btc_trend', 'btc_vol_spike', 'eth_ret_1h', 'btc_corr24',
                            'eth_btc_ratio']:
                    feat_df[col] = 0.0
            feat_df = make_targets(feat_df, feat_cfg.target_horizons)
            if len(feat_df) > feat_cfg.sequence_length * 4:
                featured[symbol] = feat_df
        except Exception as e:
            logger.warning(f'  {symbol}: feature engineering failed — {e}')

    # ── Market breadth: fraction of universe with positive 1h return ──────────
    # Computed cross-sectionally after all coins are processed.
    logger.info('Computing market breadth…')
    featured = add_market_breadth(featured)   # adds 'market_breadth' column to each df
    for symbol, feat_df in featured.items():
        logger.info(f'  {symbol}: {len(feat_df)} rows, {len(FEATURE_COLS)} features')

    if not featured:
        logger.error('No valid feature data.')
        sys.exit(1)

    # ── 3. Build datasets ─────────────────────────────────────────────────────
    logger.info('Building datasets…')
    train_ds, val_ds, test_ds, test_dfs, num_coins = build_datasets(
        featured,
        seq_len=feat_cfg.sequence_length,
        train_ratio=train_cfg.train_ratio,
        val_ratio=train_cfg.val_ratio,
    )
    logger.info(f'  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}  Coins: {num_coins}')

    # ── 4. Build model ────────────────────────────────────────────────────────
    n_features = len(FEATURE_COLS)
    model = CryptoTransformer(
        n_features=n_features,
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model: {n_params:,} parameters')

    # ── 5. Train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    history = train(model, train_ds, val_ds, train_cfg)
    logger.info(f'Training finished in {(time.time() - t0) / 60:.1f} min')

    # ── 6. Test-set evaluation ────────────────────────────────────────────────
    import numpy as np
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    model.eval().to(device)

    nw_test = 4 if device.type == 'cuda' else 0   # MPS/CPU: no worker processes
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw_test)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, coin, y in test_loader:
            pred = model(x.to(device), coin_idx=coin.to(device)).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    from scipy.stats import spearmanr

    mse  = float(np.mean((preds - targets) ** 2))
    mae  = float(np.mean(np.abs(preds - targets)))
    # Directional accuracy on 1h horizon
    dir_acc = float(np.mean(np.sign(preds[:, 0]) == np.sign(targets[:, 0])))
    # Information coefficient: Spearman rank correlation of predictions vs actuals.
    # IC > 0.03 is tradeable; IC > 0.05 is good. More meaningful than hit-rate alone
    # because it rewards ranking assets correctly, not just individual direction calls.
    ic_1h  = float(spearmanr(preds[:, 0],  targets[:, 0]).statistic)
    ic_24h = float(spearmanr(preds[:, -1], targets[:, -1]).statistic)

    logger.info(f'Test  MSE: {mse:.6f}  MAE: {mae:.6f}  '
                f'Dir-Acc(1h): {dir_acc:.3f}  IC(1h): {ic_1h:.4f}  IC(24h): {ic_24h:.4f}')

    # Save test_dfs and model metadata for backtest script
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/test_dfs.pkl', 'wb') as f:
        pickle.dump(test_dfs, f)
    with open('models/model_meta.pkl', 'wb') as f:
        pickle.dump({'num_coins': num_coins}, f)
    logger.info('Saved test_dfs.pkl and model_meta.pkl for backtest')

    print('\n=== Training complete ===')
    print(f'Test directional accuracy (1h): {dir_acc:.1%}')
    print(f'Information coefficient  (1h): {ic_1h:.4f}  (24h): {ic_24h:.4f}')
    print(f'Best validation loss: {min(history["val_loss"]):.6f}')
    print(f'Checkpoint: models/{train_cfg.checkpoint_name}')


if __name__ == '__main__':
    main()
