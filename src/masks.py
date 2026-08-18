# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Undersampling masks for k-space, registered by name.

Each registered function describes a *centered* sampling pattern (the fully sampled
low-frequency region sits in the middle of the array), which is the intuitive way to
reason about MRI undersampling. ``build_mask`` then converts the centered pattern into
the non-centered layout expected by ``torch.fft.fft2`` (DC at the corner) and returns
both layouts: the non-centered tensor for use during training/inference, and the
centered array for visualization.

New mask types only need a ``@register_mask`` decorated function with the signature
``(shape, **params) -> np.ndarray`` returning a centered 0/1 mask.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .registry import MASK_REGISTRY, register_mask


@register_mask("cartesian")
def cartesian_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """1D Cartesian undersampling: keep full phase-encode columns.

    A central band of columns is always sampled (low frequencies), and the
    remaining columns are sampled randomly so that the overall sampling rate is
    approximately ``1 / acceleration``.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float32)

    num_center = int(round(cols * center_fraction))
    start = cols // 2 - num_center // 2
    mask[:, start : start + num_center] = 1.0

    target = cols / acceleration
    remaining = max(target - num_center, 0.0)
    outer = cols - num_center
    prob = remaining / outer if outer > 0 else 0.0
    draws = rng.random(cols) < prob
    mask[:, draws] = 1.0
    return mask


@register_mask("gaussian1d")
def gaussian1d_mask(
    shape: Tuple[int, int],
    sampling_ratio: float = 0.3,
    seed: int = 0,
    std_scale: float = 0.25,
    center_lines: int = 0,
    **_: object,
) -> np.ndarray:
    """1D variable-density undersampling, per the MRI course brief.

    A fixed *fraction* (``sampling_ratio`` in {0.2, 0.3, 0.5}) of the phase-encode
    lines (rows) is retained. Row indices are drawn from a **normal distribution
    centered on the middle of k-space**, so the low frequencies -- which carry most
    of the image energy -- are densely sampled while high frequencies are sparse.
    Rows are drawn **without replacement** (rejection sampling on a set) and the
    pattern is fully determined by ``seed`` for reproducibility.

    This is a 1D pattern: whole rows are kept across all columns, matching a
    Cartesian phase-encode acquisition where entire readout lines are skipped.

    Args:
        sampling_ratio: fraction of rows to keep (0 < r <= 1).
        std_scale: std of the sampling normal as a fraction of the number of rows.
        center_lines: number of central rows to force-sample (0 = pure normal draw;
            a small value can be used to guarantee the DC line is acquired).
    """
    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError(f"sampling_ratio must be in (0, 1], got {sampling_ratio}")

    rng = np.random.default_rng(seed)
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float32)

    num_keep = int(round(sampling_ratio * rows))
    center = (rows - 1) / 2.0
    std = max(std_scale * rows, 1.0)

    chosen: set[int] = set()

    if center_lines > 0:
        start = int(round(center - center_lines / 2.0))
        for i in range(start, start + center_lines):
            if 0 <= i < rows:
                chosen.add(i)

    # Rejection sampling: draw row indices from N(center, std), keep unique in-range
    # draws until exactly num_keep rows are selected -> sampling without replacement.
    while len(chosen) < min(num_keep, rows):
        idx = int(round(rng.normal(loc=center, scale=std)))
        if 0 <= idx < rows:
            chosen.add(idx)

    mask[sorted(chosen), :] = 1.0
    return mask


@register_mask("radial")
def radial_mask(
    shape: Tuple[int, int],
    num_spokes: int = 30,
    seed: int = 0,
    **_: object,
) -> np.ndarray:
    """Pseudo-radial sampling: union of straight spokes through the center."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float32)
    cy, cx = rows / 2.0, cols / 2.0
    radius = np.hypot(rows, cols)
    angles = np.sort(rng.uniform(0, np.pi, size=num_spokes))
    t = np.linspace(-radius, radius, int(2 * radius))
    for theta in angles:
        ys = np.clip(np.round(cy + t * np.sin(theta)).astype(int), 0, rows - 1)
        xs = np.clip(np.round(cx + t * np.cos(theta)).astype(int), 0, cols - 1)
        mask[ys, xs] = 1.0
    return mask


@register_mask("poisson")
def poisson_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Variable-density random 2D sampling with a fully sampled center disk."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    cy, cx = rows / 2.0, cols / 2.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist = np.hypot(yy - cy, xx - cx) / np.hypot(cy, cx)

    density = (1.0 - dist) ** 2
    density = density / density.sum()
    prob = density * (rows * cols / acceleration)
    mask = (rng.random((rows, cols)) < prob).astype(np.float32)

    center_radius = center_fraction * np.hypot(cy, cx)
    mask[np.hypot(yy - cy, xx - cx) <= center_radius] = 1.0
    return mask


def build_mask(
    name: str,
    shape: Tuple[int, int],
    device: torch.device | str | None = None,
    **params: object,
):
    """Build a mask by name.

    Returns:
        mask_tensor: float32 tensor of shape ``(1, 1, H, W)`` in the non-centered
            (FFT) layout, ready to multiply with ``torch.fft.fft2`` output.
        mask_centered: the centered numpy array for visualization.
    """
    centered = MASK_REGISTRY.build(name, shape=shape, **params)
    mask_np = np.fft.ifftshift(centered)
    tensor = torch.from_numpy(np.ascontiguousarray(mask_np)).float()[None, None]
    if device is not None:
        tensor = tensor.to(device)
    return tensor, centered


def sampling_rate(mask_centered: np.ndarray) -> float:
    """Fraction of sampled k-space locations, in [0, 1]."""
    return float(mask_centered.mean())
