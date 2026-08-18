# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Training and evaluation loops shared by every model.

Model contract
--------------
Every registered model is an ``nn.Module`` whose forward signature is::

    forward(undersampled_kspace: complex tensor (B, 1, H, W),
            mask: real tensor (1, 1, H, W)) -> real image (B, 1, H, W)

Passing the mask into ``forward`` (rather than baking it in as a buffer, as the
original notebook did) lets the same model run under different undersampling
patterns without rebuilding, which is what the acceleration/mask sweeps need.
Models that do not need the mask (e.g. a plain U-Net on the zero-filled image)
simply ignore it.

The forward MRI model lives here in one place: a ground-truth image batch is
transformed to k-space, undersampled with the mask, and handed to the model.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from .losses import build_loss  # noqa: F401  (re-exported for convenience)
from .metrics import DATA_RANGE, MetricAccumulator, compute_metrics
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Instantiate a model from ``cfg['model']`` via the registry."""
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    return MODEL_REGISTRY.build(name, **mcfg).to(device)


def build_optimizer(model: torch.nn.Module, train_cfg: Dict) -> torch.optim.Optimizer:
    """Build an optimizer from the ``train`` config block."""
    name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer '{name}'")


def undersample(label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the forward MRI model: complex image -> k-space -> masked k-space.

    ``label`` is a 2-channel (real, imag) image ``(B, 2, H, W)``; it is combined into a
    complex image before the FFT so phase is preserved through the forward model.
    """
    return mask * fft2c(chan_to_complex(label))


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    mask: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> float:
    """Run one training epoch and return the sample-weighted mean loss."""
    model.train()
    total, count = 0.0, 0
    for label in loader:
        label = label.to(device)
        y = undersample(label, mask)

        optimizer.zero_grad()
        recon = model(y, mask)
        loss = loss_fn(recon, label)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = label.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    mask: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    data_range: float = DATA_RANGE,
) -> Dict[str, float]:
    """Evaluate over a loader; return loss plus dataset-wide metric mean/std.

    Images are normalized so the maximum magnitude is 1, hence the real and imaginary
    channels lie in ``[-1, 1]`` (a data range of 2.0). Reconstructions are clamped to
    this range so PSNR and SSIM are well defined; the loss is on the raw model output.
    """
    model.eval()
    acc = MetricAccumulator()
    total, count = 0.0, 0
    for label in loader:
        label = label.to(device)
        y = undersample(label, mask)
        recon = model(y, mask)

        bs = label.size(0)
        total += loss_fn(recon, label).item() * bs
        count += bs

        acc.update(
            compute_metrics(recon.clamp(-1, 1), label.clamp(-1, 1), data_range)
        )

    summary = acc.summary()
    summary["loss"] = total / max(count, 1)
    return summary
