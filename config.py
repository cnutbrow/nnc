from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: [
        # Original 15 — large-cap anchors
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
        'XRP/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT',
        'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'NEAR/USDT',
        # Legacy L1s / high-liquidity alts
        'TRX/USDT', 'BCH/USDT', 'ETC/USDT', 'XLM/USDT', 'VET/USDT',
        'EOS/USDT', 'ZEC/USDT', 'FTM/USDT', 'THETA/USDT', 'APT/USDT',
        # DeFi blue-chips
        'AAVE/USDT', 'MKR/USDT', 'GRT/USDT', 'CRV/USDT', 'YFI/USDT',
        'LDO/USDT',
        # Exchange / infrastructure tokens
        'CRO/USDT', 'QNT/USDT', 'IMX/USDT', 'ANKR/USDT',
        # Metaverse / gaming
        'SAND/USDT', 'MANA/USDT',
        # AI / data
        'FET/USDT', 'RNDR/USDT', 'OCEAN/USDT',
        # Utility / browser
        'BAT/USDT',
    ])
    timeframe: str = '1h'
    since_days: int = 1095  # 3 years of hourly data
    data_dir: str = 'data/raw'
    # Tried in order; first exchange that returns ≥500 rows wins per symbol.
    # Binance excluded — returns 451 in geo-restricted regions.
    exchanges: List[str] = field(default_factory=lambda: [
        'bybit', 'kucoin', 'okx', 'kraken', 'gate', 'mexc'
    ])


@dataclass
class FeatureConfig:
    sequence_length: int = 168       # 7 days of hourly context
    target_horizons: List[int] = field(default_factory=lambda: [1, 4, 12, 24])
    n_features: int = 51             # from data.features.FEATURE_COLS


@dataclass
class ModelConfig:
    tcn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_layers: int = 3
    transformer_ff_dim: int = 256
    dropout: float = 0.25
    output_dim: int = 4              # one per target horizon
    patch_size: int = 12             # PatchTST: TCN outputs grouped into 12-bar patches
                                     # seq_len=168 → 14 patch tokens for Transformer
    drop_path_rate: float = 0.10     # max stochastic-depth drop prob (linearly scaled per TCN block)


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    epochs: int = 150
    patience: int = 25               # early stopping
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test is remainder (0.15)
    model_dir: str = 'models'
    checkpoint_name: str = 'best_model.pt'


@dataclass
class TradingConfig:
    # Signal thresholds (predicted log-return)
    # Grid-searched; 0.006 filters noise while keeping ample trade frequency.
    long_entry_threshold: float = 0.006    # +0.6% predicted composite return to go long
    short_entry_threshold: float = -0.006  # -0.6% predicted composite return to go short
    close_threshold: float = 0.002         # Close position when signal reverses past ±0.2%

    # Risk management
    # 4% SL / 8% TP (1:2 R:R) maximises Sharpe vs tighter 3/6 ratios.
    stop_loss_pct: float = 0.04            # 4% stop-loss per trade
    take_profit_pct: float = 0.08          # 8% take-profit per trade
    max_position_pct: float = 0.12         # Max 12% of portfolio per asset
    max_open_positions: int = 5            # Across all assets

    # Execution
    transaction_cost: float = 0.001        # 0.1% taker fee
    slippage: float = 0.0005               # 0.05% estimated slippage
    initial_capital: float = 100_000.0

    # Kelly fraction (0 = equal-weight, 1 = full Kelly)
    kelly_fraction: float = 0.25           # quarter-Kelly for safety

    # Regime filter — only open new positions when BTC confirms the macro environment
    use_regime_filter: bool = True
    regime_adx_min: float = 0.18          # BTC ADX/100 must exceed this (real trend exists)
    regime_trend_min: float = -0.05       # BTC must be above this × 200h EMA for longs
    #   e.g. -0.05 = allow longs until BTC is >5% below its 200h EMA
