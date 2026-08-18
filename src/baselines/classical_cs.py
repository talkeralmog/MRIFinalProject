# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Classical compressed-sensing baselines: sparsity prior + k-space data consistency.

Two registered variants share the same iterative scheme -- apply a sparsity-promoting
prior in the image domain, then enforce agreement with the acquired k-space lines
(POCS) -- and differ only in how strong that prior is:

``classical_cs``
    A single-level Haar wavelet soft-threshold. This is the naive textbook form and it
    is kept unchanged for the report's prior ablation: a one-level Haar transform only
    exposes the finest-scale detail coefficients, so the shrinkage is little more than a
    mild local smoothing and the iteration converges essentially back onto the
    zero-filled solution (measured: +0.13 dB on a Shepp-Logan phantom, and no value of
    ``lam`` over four orders of magnitude does better).

``classical_cs_tv``
    The properly regularized form used as the report's baseline: a *multi-level* wavelet
    soft-threshold combined with a total-variation proximal step. TV is the standard
    prior for compressed-sensing MRI (Lustig et al., 2007) and it is anatomically
    motivated -- brain tissue is piecewise smooth with sharp boundaries between grey
    matter, white matter and CSF, so the image gradient is sparse, whereas the
    undersampling aliasing is spread over the whole field of view.

Both are training-free: the transforms use fixed (non-learnable) filters, so the module
has no trainable parameters and the training loop treats it as evaluation-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from ..utils import chan_to_complex, complex_to_chan, fft2c, ifft2c

# Operating point of the TV baseline, selected on the VALIDATION split by
# configs/experiments/baseline_tuning.yaml. Re-run that sweep and update these two
# numbers if the data or the pre-processing changes; the comparison sweep deliberately
# does not override them, so the tuned values live in exactly one place.
TUNED_LAM = 0.03
TUNED_TV_WEIGHT = 0.02


def _haar_filters() -> torch.Tensor:
    """Return the 4 orthonormal single-level Haar analysis filters, shape (4,1,2,2)."""
    half = 0.5
    ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    return half * torch.stack([ll, lh, hl, hh]).unsqueeze(1)


def _soft_threshold(x: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.sign(x) * torch.relu(torch.abs(x) - thr)


def _neg_divergence(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Negative divergence of the dual field ``p`` (backward differences)."""
    d = -(px + py)
    d[..., 1:, :] += px[..., :-1, :]
    d[..., 1:] += py[..., :-1]
    return d


def tv_denoise(x: torch.Tensor, weight: float, num_iters: int = 20) -> torch.Tensor:
    """Proximal operator of the isotropic total-variation penalty (Chambolle, 2004).

    Solves ``argmin_u 0.5*||u - x||^2 + weight*TV(u)`` with the classic dual fixed-point
    iteration, applied independently to every ``(H, W)`` plane of ``x``. This mirrors
    ``skimage.restoration.denoise_tv_chambolle`` but stays in torch, so the baseline runs
    on the GPU alongside the learned models and needs no extra dependency.
    """
    if weight <= 0:
        return x

    tau = 0.25  # 1 / (2 * ndim) for 2D
    px = torch.zeros_like(x)
    py = torch.zeros_like(x)
    out = x
    for i in range(num_iters):
        if i > 0:
            out = x + _neg_divergence(px, py)
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(x)
        gx[..., :-1, :] = out[..., 1:, :] - out[..., :-1, :]
        gy[..., :-1] = out[..., 1:] - out[..., :-1]
        norm = torch.sqrt(gx ** 2 + gy ** 2) * (tau / weight) + 1.0
        px = (px - tau * gx) / norm
        py = (py - tau * gy) / norm
    return x + _neg_divergence(px, py)


@register_model("classical_cs")
class ClassicalCS(nn.Module):
    """Iterative CS reconstruction: sparsity prior + POCS data consistency.

    Args:
        num_iters: number of prior/data-consistency alternations.
        lam: soft-threshold applied to the wavelet detail coefficients.
        wavelet_levels: depth of the Haar decomposition. 1 reproduces the naive
            single-scale variant; 3 exposes coarser scales where brain structure is
            genuinely sparse.
        tv_weight: strength of the additional total-variation proximal step
            (0 disables it).
        tv_iters: inner iterations of the TV solver.
    """

    def __init__(self, num_iters: int = 50, lam: float = 0.02,
                 wavelet_levels: int = 1, tv_weight: float = 0.0,
                 tv_iters: int = 20, **_):
        super().__init__()
        self.num_iters = int(num_iters)
        self.lam = float(lam)
        self.wavelet_levels = int(wavelet_levels)
        self.tv_weight = float(tv_weight)
        self.tv_iters = int(tv_iters)
        self.register_buffer("haar", _haar_filters())

    def _effective_levels(self, height: int, width: int) -> int:
        """Cap the decomposition depth at what the image size actually supports."""
        levels = self.wavelet_levels
        while levels > 1 and (height % (2 ** levels) or width % (2 ** levels)):
            levels -= 1
        return max(levels, 1)

    def _wavelet_shrink(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-level Haar soft-thresholding of the detail sub-bands.

        The real and imaginary channels are folded into the batch dimension so the
        single-channel Haar filters apply to both alike.
        """
        b, c, h, w = x.shape
        levels = self._effective_levels(h, w)
        cur = x.reshape(b * c, 1, h, w)

        details = []
        for _ in range(levels):
            coeffs = F.conv2d(cur, self.haar, stride=2)
            details.append(_soft_threshold(coeffs[:, 1:], self.lam))
            cur = coeffs[:, :1]
        for detail in reversed(details):
            cur = F.conv_transpose2d(torch.cat([cur, detail], dim=1), self.haar, stride=2)
        return cur.reshape(b, c, h, w)

    def _data_consistency(self, x: torch.Tensor, y: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """POCS: restore the measured k-space lines, keep the estimate elsewhere."""
        kspace = fft2c(chan_to_complex(x))
        kspace = y + (1.0 - mask) * kspace
        return complex_to_chan(ifft2c(kspace))

    @torch.no_grad()
    def forward(self, undersampled_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = undersampled_kspace
        x = complex_to_chan(ifft2c(y))  # (B, 2, H, W): real, imag
        for _ in range(self.num_iters):
            x = self._wavelet_shrink(x)
            x = tv_denoise(x, self.tv_weight, self.tv_iters)
            x = self._data_consistency(x, y, mask)
        return x


@register_model("classical_cs_tv")
class ClassicalCSWaveletTV(ClassicalCS):
    """The report's classical baseline: multi-level wavelet + TV, tuned on validation."""

    def __init__(self, num_iters: int = 50, lam: float = TUNED_LAM,
                 wavelet_levels: int = 3, tv_weight: float = TUNED_TV_WEIGHT,
                 tv_iters: int = 20, **_):
        super().__init__(num_iters=num_iters, lam=lam, wavelet_levels=wavelet_levels,
                         tv_weight=tv_weight, tv_iters=tv_iters)
