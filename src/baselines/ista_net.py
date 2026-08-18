# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""ISTA-Net baseline: a second deep-unfolding network for comparison.

ISTA-Net (Zhang & Ghanem, CVPR 2018) unrolls the Iterative Shrinkage-Thresholding
Algorithm. Each phase performs a gradient/data-consistency step followed by a learned
nonlinear proximal operator built from a forward transform, soft-thresholding, and a
backward transform. Comparing ISTA-Net to our ADMM-Net contrasts two different
unrolling designs (ISTA vs ADMM) under the same training and evaluation harness.

This is a compact implementation of the core phase; it omits the auxiliary symmetry
constraint loss of the original paper so it plugs into the project's standard losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from ..utils import chan_to_complex, complex_to_chan, fft2c, ifft2c


class ISTANetPhase(nn.Module):
    """One ISTA-Net phase: data-consistency gradient step + learned proximal."""

    def __init__(self, channels: int = 32, step_init: float = 0.1, thr_init: float = 0.01):
        super().__init__()
        self.step = nn.Parameter(torch.tensor(float(step_init)))
        self.threshold = nn.Parameter(torch.tensor(float(thr_init)))
        self.lift = nn.Conv2d(2, channels, 3, padding=1)  # (real, imag) input
        self.forward_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.backward_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.project = nn.Conv2d(channels, 2, 3, padding=1)  # (real, imag) output

    def forward(self, x, y, mask):
        # Gradient step on the data term ||M F x - y||^2 (y are masked measurements).
        # x is a 2-channel (real, imag) image; the gradient is taken in the complex
        # domain and mapped back to the 2-channel representation.
        residual_k = mask * fft2c(chan_to_complex(x)) - y
        grad = complex_to_chan(ifft2c(residual_k))
        r = x - self.step * grad

        # Learned proximal operator with soft-thresholding in the transform domain.
        feat = self.lift(r)
        transformed = self.forward_transform(feat)
        thr = F.softplus(self.threshold)
        thresholded = torch.sign(transformed) * torch.relu(torch.abs(transformed) - thr)
        back = self.backward_transform(thresholded)
        return r + self.project(back)


@register_model("ista_net")
class ISTANet(nn.Module):
    def __init__(self, num_stages: int = 8, channels: int = 32,
                 share_weights: bool = False, step_init: float = 0.1,
                 thr_init: float = 0.01, **_):
        super().__init__()
        self.num_stages = num_stages

        def make_phase():
            return ISTANetPhase(channels, step_init, thr_init)

        if share_weights:
            shared = make_phase()
            self.phases = nn.ModuleList([shared] * num_stages)
        else:
            self.phases = nn.ModuleList([make_phase() for _ in range(num_stages)])

    def forward(self, undersampled_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = undersampled_kspace
        x = complex_to_chan(ifft2c(y))  # (B, 2, H, W): real, imag
        for phase in self.phases:
            x = phase(x, y, mask)
        return x
