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
    exchanges: List[str] = field(default_factory=lambda: [
        'bybit', 'kucoin', 'okx', 'kraken', 'gate', 'mexc'
    ])


@dataclass
class FeatureConfig:
    sequence_length: int = 168       # 7 days of hourly context
    target_horizons: List[int] = field(default_factory=lambda: [1, 4, 12, 24])
    n_features: int = 57             # len(data.features.FEATURE_COLS)


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    n_layers: int = 1
    dropout: float = 0.30


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    epochs: int = 150
    patience: int = 30
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    model_dir: str = 'models'
    checkpoint_name: str = 'best_model.pt'


@dataclass
class TradingConfig:
    # Thresholds in probability space [0, 1]
    # prob > 0.55 → long, prob < 0.45 → short
    long_entry_threshold: float = 0.55
    short_entry_threshold: float = 0.45
    close_threshold: float = 0.50

    # Risk management
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.08
    max_position_pct: float = 0.12
    max_open_positions: int = 5

    # Execution
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    initial_capital: float = 100_000.0

    kelly_fraction: float = 0.25

    # Regime filter
    use_regime_filter: bool = True
    regime_adx_min: float = 0.18
    regime_trend_min: float = -0.05
