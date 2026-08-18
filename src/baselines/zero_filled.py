# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Zero-filled reconstruction baseline (no learning).

The trivial reconstruction: take the inverse FFT of the zero-filled undersampled
k-space, keeping both real and imaginary parts (returned as a 2-channel image). It is
the aliased model input and the lower bound every other method should beat; it has no
trainable parameters, so the training loop treats it as evaluation-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_model
from ..utils import complex_to_chan, ifft2c


@register_model("zero_filled")
class ZeroFilled(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, undersampled_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return complex_to_chan(ifft2c(undersampled_kspace))
