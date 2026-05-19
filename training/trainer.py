"""
Training loop with:
  - AdamW optimizer + cosine-annealing LR schedule with linear warmup
  - Gradient clipping
  - Early stopping on validation loss
  - Model checkpointing
  - Mixed-precision (AMP) when CUDA is available
"""

import os
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WeightedHorizonLoss(nn.Module):
    """
    Composite loss for multi-horizon return prediction:

      (1 - dir_weight) × weighted Huber   — magnitude, equalised across horizons
    +       dir_weight × directional BCE  — explicitly rewards correct sign

    The Huber component is weighted by 1/sqrt(horizon) so that the 1h target
    contributes equally to the 24h target instead of being 28× drowned out.

    The directional BCE treats pred/delta as a logit for "return > 0", giving
    the model a direct gradient signal for direction even when magnitude is tiny.

    Default horizons [1, 4, 12, 24] → magnitude weights ≈ [0.50, 0.25, 0.14, 0.10]
    """

    def __init__(
        self,
        horizons: list[int] = None,
        delta: float = 0.01,
        dir_weight: float = 0.4,     # raised 0.3→0.4: direction is the key trading signal
        sigma_1h: float = 0.005,     # typical 1-hour log-return std (~0.5%)
        clip_sigma: float = 3.0,     # clip targets at ±clip_sigma × per-horizon σ
    ):
        super().__init__()
        if horizons is None:
            horizons = [1, 4, 12, 24]
        h = np.array(horizons, dtype=np.float32)

        # Magnitude weights: 1/sqrt(h), normalised
        w = 1.0 / np.sqrt(h)
        w = w / w.sum()
        self.register_buffer('weights', torch.tensor(w, dtype=torch.float32))

        # Per-horizon volatility scale: returns scale with sqrt(time)
        # This bounds the directional logit to a comparable range across horizons
        sigma = (sigma_1h * np.sqrt(h)).astype(np.float32)
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))

        self.huber      = nn.HuberLoss(reduction='none', delta=delta)
        self.dir_weight = dir_weight
        self.clip_sigma = clip_sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ── Target clipping at ±3σ per horizon ───────────────────────────────
        # Crypto has frequent >5σ return events that dominate MSE gradients and
        # push the model to learn spike-prediction instead of direction.
        # Clipping at ±3σ eliminates outlier dominance without discarding direction.
        clip = self.sigma * self.clip_sigma   # (H,)
        target = target.clamp(-clip, clip)

        # ── Magnitude loss (weighted Huber) ───────────────────────────────────
        per_horizon = self.huber(pred, target)          # (N, H)
        mag_loss = (per_horizon * self.weights).sum(dim=1).mean()

        # ── Directional loss (soft-sign agreement, bounded [0, 1]) ───────────
        # soft_sign = tanh(pred / σ_h) ∈ (-1, 1)  — bounded soft indicator of direction
        # direction = ±1 based on target sign
        # loss = (1 - soft_sign * direction) / 2
        #        → 0 when confidently correct, 0.5 when pred≈0, 1 when confidently wrong
        #
        # Unlike BCE, this stays bounded even when the model outputs large
        # wrong-direction predictions at initialisation, so training doesn't
        # explode.  Gradient at pred=0 is -direction/(2σ) — always points the
        # right way, so the model has signal even before it predicts anything.
        soft_sign = torch.tanh(pred / self.sigma)           # (N, H) ∈ (-1, 1)
        direction = (target > 0).float() * 2 - 1            # (N, H) ∈ {-1, +1}
        dir_loss  = ((1 - soft_sign * direction) / 2).mean()

        return (1 - self.dir_weight) * mag_loss + self.dir_weight * dir_loss


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
    grad_clip: float = 1.0,
) -> float:
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False, desc='train' if train else 'val'):
            # Batch is (x, coin_idx, y) when coin embeddings are used,
            # or (x, y) for backwards compatibility.
            if len(batch) == 3:
                x, coin_idx, y = batch
                coin_idx = coin_idx.to(device, non_blocking=True)
            else:
                x, y = batch
                coin_idx = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                # ── Gaussian feature noise ────────────────────────────────────
                # Small perturbation forces robust representations.
                x = x + torch.randn_like(x) * 0.02

                # ── Time masking (SpecAugment-style) ─────────────────────────
                # Randomly zero out a contiguous block of ~15% of time steps.
                # Prevents the model from relying on any specific bar position
                # and is the most effective augmentation for financial OHLCV series.
                if torch.rand(1).item() < 0.5:
                    T = x.size(1)
                    n_mask = max(1, int(T * 0.15))
                    mask_start = torch.randint(0, T - n_mask + 1, (1,)).item()
                    x[:, mask_start:mask_start + n_mask, :] = 0.0

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred = model(x, coin_idx=coin_idx)
                loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


class EarlyStopping:
    def __init__(self, patience: int, model_path: str):
        self.patience   = patience
        self.model_path = model_path
        self.best_loss  = float('inf')
        self.counter    = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.model_path)
            return False   # continue
        self.counter += 1
        return self.counter >= self.patience   # True → stop


def train(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg,                 # TrainingConfig
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps'  if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    model = model.to(device)

    # pin_memory is only beneficial on CUDA; MPS doesn't support it.
    # num_workers > 0 on MPS spawns macOS subprocesses that add noise and
    # can stall; keep workers for CUDA/CPU only.
    pin  = device.type == 'cuda'
    nw   = 4 if device.type != 'mps' else 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps   = len(train_loader) * cfg.epochs
    warmup_steps  = len(train_loader) * cfg.warmup_epochs
    scheduler     = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    from config import FeatureConfig
    horizons  = FeatureConfig().target_horizons
    criterion = WeightedHorizonLoss(horizons=horizons, delta=0.01).to(device)

    os.makedirs(cfg.model_dir, exist_ok=True)
    checkpoint   = os.path.join(cfg.model_dir, cfg.checkpoint_name)
    resume_path  = checkpoint.replace('.pt', '_resume.pt')
    stopper      = EarlyStopping(cfg.patience, checkpoint)
    history      = {'train_loss': [], 'val_loss': []}
    start_epoch  = 1

    # ── Auto-resume if a prior run was interrupted ────────────────────────────
    if os.path.exists(resume_path):
        logger.info(f'Resume checkpoint found — loading {resume_path}')
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch          = state['epoch'] + 1
        stopper.best_loss    = state['best_loss']
        stopper.counter      = state['patience_counter']
        history              = state['history']
        logger.info(f'Resuming from epoch {start_epoch}  '
                    f'(best val {stopper.best_loss:.6f}, '
                    f'patience {stopper.counter}/{cfg.patience})')
    else:
        logger.info('No resume checkpoint — starting from scratch')

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, scheduler, scaler,
                            criterion, device, train=True,  grad_clip=cfg.grad_clip)
        va_loss = run_epoch(model, val_loader,   optimizer, scheduler, scaler,
                            criterion, device, train=False, grad_clip=cfg.grad_clip)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        logger.info(f'Epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}')

        # ── Save full resume state every 5 epochs ────────────────────────────
        # Saving every epoch spikes MPS memory: torch.save serialises MPS
        # tensors by copying them to CPU, temporarily doubling memory usage.
        # Every 5 epochs loses at most ~2.5 h of progress if killed; the best-
        # model checkpoint (written by EarlyStopping above) is always current.
        if epoch % 5 == 0:
            if device.type == 'mps':
                torch.mps.synchronize()   # drain pending GPU ops before serialising
                torch.mps.empty_cache()   # free fragmented MPS allocations
            torch.save({
                'model':            model.state_dict(),
                'optimizer':        optimizer.state_dict(),
                'scheduler':        scheduler.state_dict(),
                'scaler':           scaler.state_dict(),
                'epoch':            epoch,
                'best_loss':        stopper.best_loss,
                'patience_counter': stopper.counter,
                'history':          history,
            }, resume_path)

        if device.type == 'mps':
            torch.mps.empty_cache()

        if stopper(va_loss, model):
            logger.info(f'Early stopping at epoch {epoch}. Best val: {stopper.best_loss:.6f}')
            break

    # Restore best weights
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.info(f'Loaded best checkpoint from {checkpoint}')

    return history
