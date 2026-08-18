# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Loss functions, registered by name and selected (and combined) via config.

A registered loss has signature ``loss(pred, target, **params) -> scalar tensor``.
``build_loss`` binds the config's params to the function so the engine can call a plain
``loss_fn(pred, target)``.

Losses are composable: the ``composite`` loss sums any set of other registered losses
with per-term weights, so an experiment can mix image-domain, structural, and
k-space-domain objectives purely from YAML, e.g.::

    loss:
      name: composite
      terms: {mse: 1.0, ssim: 0.1, kspace: 0.05}

New losses only require a ``@register_loss`` decorated function here.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .registry import LOSS_REGISTRY, register_loss
from .utils import chan_to_complex, fft2c


# ---------------------------------------------------------------------------
# Image-domain losses
# ---------------------------------------------------------------------------


@register_loss("mse")
def mse_loss(pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    return F.mse_loss(pred, target)


@register_loss("l1")
def l1_loss(pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    return F.l1_loss(pred, target)


@register_loss("mse_l1")
def mse_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                mse_w: float = 1.0, l1_w: float = 0.1, **_) -> torch.Tensor:
    """Combined MSE + L1 loss (the notebook's training objective)."""
    return mse_w * F.mse_loss(pred, target) + l1_w * F.l1_loss(pred, target)


@register_loss("charbonnier")
def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3, **_) -> torch.Tensor:
    """Charbonnier (smooth-L1) loss: robust to outliers, differentiable at 0."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


# ---------------------------------------------------------------------------
# Structural similarity (differentiable, image-domain)
# ---------------------------------------------------------------------------


def _gaussian_window(win_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d[None, None]


def ssim_index(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0,
               win_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM over a batch of images (differentiable).

    Multi-channel inputs (e.g. the 2-channel real/imag MRI image) are folded into the
    batch so SSIM is computed per channel and averaged.
    """
    b, c, h, w = pred.shape
    pred = pred.reshape(b * c, 1, h, w)
    target = target.reshape(b * c, 1, h, w)
    window = _gaussian_window(win_size, sigma, pred.device, pred.dtype)
    pad = win_size // 2
    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


@register_loss("ssim")
def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0,
              win_size: int = 7, sigma: float = 1.5, **_) -> torch.Tensor:
    """1 - SSIM, so that minimizing the loss maximizes structural similarity."""
    return 1.0 - ssim_index(pred, target, data_range, win_size, sigma)


# ---------------------------------------------------------------------------
# k-space (frequency) domain loss
# ---------------------------------------------------------------------------


@register_loss("kspace")
def kspace_loss(pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    """MSE between the full k-space of the reconstruction and the target.

    Penalizes errors directly in the frequency domain (real + imaginary parts),
    complementing image-domain losses with a measurement-consistency objective.
    """
    kp = torch.view_as_real(fft2c(chan_to_complex(pred)))
    kt = torch.view_as_real(fft2c(chan_to_complex(target)))
    return F.mse_loss(kp, kt)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@register_loss("composite")
def composite_loss(pred: torch.Tensor, target: torch.Tensor,
                   terms: Dict[str, float] | None = None, **_) -> torch.Tensor:
    """Weighted sum of other registered losses, selected by name from ``terms``."""
    terms = terms or {"mse": 1.0}
    total = pred.new_zeros(())
    for name, weight in terms.items():
        if name == "composite":
            raise ValueError("composite loss cannot reference itself")
        total = total + float(weight) * LOSS_REGISTRY.get(name)(pred, target)
    return total


def build_loss(loss_cfg: Dict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a ``loss_fn(pred, target)`` from a config dict with a ``name`` key."""
    params = {k: v for k, v in loss_cfg.items() if k != "name"}
    fn = LOSS_REGISTRY.get(loss_cfg["name"])
    return functools.partial(fn, **params)
