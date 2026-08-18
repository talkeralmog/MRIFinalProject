# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Unrolled ADMM-Net for compressed-sensing MRI (soft-threshold variant).

This is the project's modernized take on the ADMM-Net of Yang et al. (NIPS 2016):
the iterative ADMM updates are unrolled into a fixed number of identical stages, and
the proximal/regularization operators are made learnable. Two registered variants
share the same unrolling and data-consistency logic, differing only in the Z-update
nonlinearity:

* ``admmnet_softthresh`` -- a learnable channel-wise soft-threshold (modernized) with
  residual + BatchNorm convolution blocks for stable training.
* ``admmnet_pwl`` -- the paper-faithful learnable piecewise-linear nonlinearity
  (``PiecewiseLinear``) mirroring the original ``LinearLabel = -1:0.02:1`` knots.

All components are configurable from YAML so the model is a moving target for
experiments: ``num_stages``, ``channels``, ``share_weights``, ``use_bn``,
``use_residual`` and ``rho_init`` are all constructor arguments populated from the
``model`` config block.

Forward contract (shared by every model in the project)::

    forward(undersampled_kspace: (B, 1, H, W) complex,
            mask: (1, 1, H, W) real) -> (B, 2, H, W) real  (real, imag channels)

The image is carried as a 2-channel real tensor (real part, imaginary part) so the
network processes the complex MRI signal end to end; the data-consistency step
converts back to a genuine complex k-space for the FFT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .utils import chan_to_complex, complex_to_chan, fft2c, ifft2c


class LearnableSoftThreshold(nn.Module):
    """Channel-wise soft-thresholding with a learnable, strictly positive threshold.

    Implements the proximal operator of the L1 norm used in the ADMM Z-update. A
    softplus keeps each per-channel threshold positive throughout training.
    """

    def __init__(self, channels: int, init: float = 0.01):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = F.softplus(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - t)


class RobustCNNBlock(nn.Module):
    """Two 3x3 convolutions with optional BatchNorm and a residual connection.

    The residual is only added when input and output channel counts match (otherwise
    there is nothing to add); this mirrors the behavior of the original notebook block
    while making BatchNorm and the residual independently switchable.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_bn: bool = True, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else 0.0
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PiecewiseLinear(nn.Module):
    """Learnable piecewise-linear nonlinearity, faithful to the original ADMM-Net.

    The paper's nonlinear transform layer is a piecewise-linear function defined by
    control points on a fixed grid (the original ``LinearLabel = -1:0.02:1``, i.e. 101
    knots). Knot *positions* are fixed; the function *values* at the knots are learned,
    independently per channel. Inputs outside the grid are linearly extrapolated using
    the boundary segment's slope. The values are initialized to the identity so each
    stage starts as a pass-through.
    """

    def __init__(self, channels: int, num_knots: int = 101, knot_range: float = 1.0):
        super().__init__()
        knots = torch.linspace(-knot_range, knot_range, num_knots)
        self.register_buffer("knots", knots)
        self.p0 = float(knots[0])
        self.dp = float(knots[1] - knots[0])
        self.num_knots = num_knots
        # learnable per-channel values, initialized to identity (q_i = p_i)
        self.values = nn.Parameter(knots.clone().unsqueeze(0).repeat(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        pos = (x - self.p0) / self.dp
        idx = torch.floor(pos).long().clamp(0, self.num_knots - 2)
        t = pos - idx.float()  # not clamped: t<0 or t>1 yields linear extrapolation
        chan = torch.arange(channels, device=x.device).view(1, channels, 1, 1).expand_as(idx)
        q_left = self.values[chan, idx]
        q_right = self.values[chan, idx + 1]
        return q_left + t * (q_right - q_left)


class CustomADMMStage(nn.Module):
    """One unrolled ADMM iteration with an injected nonlinearity.

    The stage performs a feature-domain regularization step (analysis -> nonlinearity
    -> dual update -> synthesis) to produce a prior, then enforces consistency with the
    measured k-space (the X-update / data-consistency step) with a learnable, strictly
    positive penalty ``rho``. The Z-update nonlinearity is injected so the same stage
    serves both the soft-threshold and piecewise-linear model variants.
    """

    def __init__(self, channels: int, nonlinearity: nn.Module, use_bn: bool = True,
                 use_residual: bool = True, rho_init: float = 0.1):
        super().__init__()
        self.analysis = RobustCNNBlock(2, channels, use_bn, use_residual)
        self.nonlinearity = nonlinearity
        self.synthesis = RobustCNNBlock(channels, 2, use_bn, use_residual)
        self.rho = nn.Parameter(torch.tensor(float(rho_init)))

    def data_consistency(self, x_prior: torch.Tensor, y: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """X-update: blend the prior's k-space with the measurements where sampled.

        Operates on the complex k-space of the (real, imag) image and returns the
        updated image back in the 2-channel representation.
        """
        x_kspace = fft2c(chan_to_complex(x_prior))
        rho = F.softplus(self.rho)
        blended = mask * ((y + rho * x_kspace) / (1.0 + rho)) + (1.0 - mask) * x_kspace
        return complex_to_chan(ifft2c(blended))

    def forward(self, x, z, m, y, mask):
        c = self.analysis(x)              # C-update: feature extraction
        z_new = self.nonlinearity(c + m)  # Z-update: learnable proximal
        m_new = m + c - z_new             # M-update: dual ascent
        correction = self.synthesis(z_new - m_new)
        x_prior = x + correction
        x_new = self.data_consistency(x_prior, y, mask)  # X-update
        return x_new, z_new, m_new


class _UnrolledADMMNet(nn.Module):
    """Shared unrolling logic; subclasses only choose the per-stage nonlinearity.

    ``make_stage`` is a callable returning a fresh ``CustomADMMStage``. When
    ``share_weights`` is true a single stage instance is reused across all iterations
    (true weight tying); otherwise each iteration gets independent parameters.
    """

    def __init__(self, num_stages: int, channels: int, share_weights: bool, make_stage):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels
        self.share_weights = share_weights
        if share_weights:
            shared = make_stage()
            self.stages = nn.ModuleList([shared] * num_stages)
        else:
            self.stages = nn.ModuleList([make_stage() for _ in range(num_stages)])

    def forward(self, undersampled_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = undersampled_kspace
        x = complex_to_chan(ifft2c(y))  # (B, 2, H, W): real, imag
        b, _, h, w = x.shape
        z = torch.zeros(b, self.channels, h, w, device=x.device, dtype=x.dtype)
        m = torch.zeros_like(z)
        for stage in self.stages:
            x, z, m = stage(x, z, m, y, mask)
        return x


@register_model("admmnet_softthresh")
class ADMMNet(_UnrolledADMMNet):
    """Unrolled ADMM-Net with a learnable channel-wise soft-threshold nonlinearity."""

    def __init__(self, num_stages: int = 8, channels: int = 64,
                 share_weights: bool = False, use_bn: bool = True,
                 use_residual: bool = True, rho_init: float = 0.1, **_):
        def make_stage():
            return CustomADMMStage(
                channels, LearnableSoftThreshold(channels), use_bn, use_residual, rho_init
            )

        super().__init__(num_stages, channels, share_weights, make_stage)


@register_model("admmnet_pwl")
class ADMMNetPWL(_UnrolledADMMNet):
    """Paper-faithful ADMM-Net with learnable piecewise-linear nonlinearities."""

    def __init__(self, num_stages: int = 8, channels: int = 64,
                 share_weights: bool = False, use_bn: bool = True,
                 use_residual: bool = True, rho_init: float = 0.1,
                 num_knots: int = 101, knot_range: float = 1.0, **_):
        def make_stage():
            return CustomADMMStage(
                channels, PiecewiseLinear(channels, num_knots, knot_range),
                use_bn, use_residual, rho_init,
            )

        super().__init__(num_stages, channels, share_weights, make_stage)
