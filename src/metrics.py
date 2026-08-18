# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Reconstruction quality metrics for complex-valued MRI: PSNR and SSIM.

The MRI course brief requires, for complex-valued images, computing PSNR and SSIM on
the **real and imaginary components separately**. Images are represented as 2-channel
real tensors ``(B, 2, H, W)`` = (real, imag), normalized so the maximum magnitude is 1
(so each channel lies in ``[-1, 1]`` -> data range 2.0).

Reading the absolute numbers
---------------------------
``DATA_RANGE = 2.0`` is the width of the interval the channels are guaranteed to lie in,
not the amplitude of the signal actually present: the measured RMS magnitude of a central
brain slice in this dataset is about 0.2. The peak in "peak signal-to-noise ratio" is
therefore roughly 10x the typical signal, which makes every PSNR in this project higher
than one computed with a per-image peak. The choice is applied identically to every
method, split and sampling ratio, so all *comparisons* are fair, but the absolute dB
values are not directly comparable to published numbers. ``convert_psnr`` gives the exact
conversion, and the report quotes both conventions.

The same reasoning explains why PSNR on the imaginary channel comes out several dB above
PSNR on the real channel: the imaginary component carries a smaller amplitude, so a fixed
peak of 2.0 flatters it. It is not evidence that the phase is reconstructed better.

All functions return per-image values so downstream code can (a) aggregate into
dataset-wide mean/std via ``MetricAccumulator`` and (b) log per-sample values for the
required scatter plots / Pearson correlations and the qualitative example selection.
An extra complex NMSE is also provided (not required by the brief, reported as a bonus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

DATA_RANGE = 2.0  # real/imag channels lie in [-1, 1] after magnitude normalization


def convert_psnr(psnr_db: float, from_range: float = DATA_RANGE,
                 to_range: float = 1.0) -> float:
    """Restate a PSNR value under a different assumed data range.

    PSNR is ``20*log10(peak) - 10*log10(mse)``, so changing only the assumed peak shifts
    every value by the same constant: switching from a range of 2.0 to the more common
    "peak = maximum magnitude = 1.0" costs exactly 20*log10(2) = 6.02 dB.
    """
    return psnr_db + 20.0 * math.log10(to_range / from_range)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Reshape a single-channel batch ``(B, H, W)`` to ``(B, H*W)``."""
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def psnr_channel(pred: torch.Tensor, target: torch.Tensor, data_range: float = DATA_RANGE) -> torch.Tensor:
    """Per-image PSNR (dB) for one channel ``(B, H, W)`` -> ``(B,)``."""
    p, t = _flatten(pred.float()), _flatten(target.float())
    mse = torch.mean((p - t) ** 2, dim=1).clamp_min(1e-12)
    peak = torch.tensor(float(data_range), device=p.device)
    return 20.0 * torch.log10(peak) - 10.0 * torch.log10(mse)


@torch.no_grad()
def ssim_channel(pred: torch.Tensor, target: torch.Tensor, data_range: float = DATA_RANGE) -> torch.Tensor:
    """Per-image SSIM for one channel ``(B, H, W)`` via scikit-image -> ``(B,)``."""
    from skimage.metrics import structural_similarity

    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    vals = [
        structural_similarity(t[i], p[i], data_range=data_range)
        for i in range(p.shape[0])
    ]
    return torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def nmse_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image normalized MSE over both channels ``(B, 2, H, W)`` -> ``(B,)`` (bonus)."""
    p = pred.float().reshape(pred.shape[0], -1)
    t = target.float().reshape(target.shape[0], -1)
    num = torch.sum((p - t) ** 2, dim=1)
    den = torch.sum(t ** 2, dim=1).clamp_min(1e-12)
    return num / den


def _split_channels(x: torch.Tensor):
    """Return (real, imag) single-channel batches from ``(B, 2, H, W)``."""
    if x.dim() != 4 or x.shape[1] != 2:
        raise ValueError(f"expected a (B, 2, H, W) complex-as-channels tensor, got {tuple(x.shape)}")
    return x[:, 0], x[:, 1]


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = DATA_RANGE,
) -> Dict[str, torch.Tensor]:
    """Per-image PSNR/SSIM on real & imaginary channels (+ complex NMSE) for a batch."""
    pr, pi = _split_channels(pred)
    tr, ti = _split_channels(target)
    return {
        "psnr_real": psnr_channel(pr, tr, data_range),
        "psnr_imag": psnr_channel(pi, ti, data_range),
        "ssim_real": ssim_channel(pr, tr, data_range),
        "ssim_imag": ssim_channel(pi, ti, data_range),
        "nmse": nmse_complex(pred, target),
    }


@dataclass
class MetricAccumulator:
    """Collect per-image metric values across batches into a dataset summary."""

    values: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, batch_metrics: Dict[str, torch.Tensor]) -> None:
        for key, tensor in batch_metrics.items():
            self.values.setdefault(key, []).extend(
                tensor.detach().cpu().reshape(-1).tolist()
            )

    def summary(self) -> Dict[str, float]:
        """Return ``{metric_mean, metric_std}`` over all accumulated images."""
        out: Dict[str, float] = {}
        for key, vals in self.values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(arr.mean()) if arr.size else float("nan")
            out[f"{key}_std"] = float(arr.std()) if arr.size else float("nan")
        out["n_images"] = len(next(iter(self.values.values()))) if self.values else 0
        return out
