"""
Training loop:
  - AdamW optimizer + cosine LR schedule with linear warmup
  - Gradient clipping
  - Early stopping on validation loss
  - Model checkpointing + auto-resume
  - Mixed-precision AMP on CUDA
"""

import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LabelSmoothBCE(nn.Module):
    """BCE with label smoothing to prevent overconfident predictions."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(logits, targets)


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
            if len(batch) == 3:
                x, coin_idx, y = batch
                coin_idx = coin_idx.to(device, non_blocking=True)
            else:
                x, y = batch
                coin_idx = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                # Gaussian feature noise for robustness
                x = x + torch.randn_like(x) * 0.01

                # Time masking: zero out ~20% of timesteps
                if torch.rand(1).item() < 0.5:
                    T = x.size(1)
                    n_mask = max(1, int(T * 0.20))
                    mask_start = torch.randint(0, T - n_mask + 1, (1,)).item()
                    x[:, mask_start:mask_start + n_mask, :] = 0.0

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(x, coin_idx=coin_idx)
                loss   = criterion(logits, y)

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
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps'  if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    model = model.to(device)

    pin = device.type == 'cuda'
    nw  = 4 if device.type != 'mps' else 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    optimizer     = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                      weight_decay=cfg.weight_decay)
    total_steps   = len(train_loader) * cfg.epochs
    warmup_steps  = len(train_loader) * cfg.warmup_epochs
    scheduler     = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion     = LabelSmoothBCE(smoothing=0.10).to(device)

    os.makedirs(cfg.model_dir, exist_ok=True)
    checkpoint  = os.path.join(cfg.model_dir, cfg.checkpoint_name)
    resume_path = checkpoint.replace('.pt', '_resume.pt')
    stopper     = EarlyStopping(cfg.patience, checkpoint)
    history     = {'train_loss': [], 'val_loss': []}
    start_epoch = 1

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

        if epoch % 5 == 0:
            if device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
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

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.info(f'Loaded best checkpoint from {checkpoint}')

    return history
