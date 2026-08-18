# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""U-Net baseline: a non-unrolled CNN that maps the zero-filled image to a clean one.

This is the "deep but not model-based" reference. Unlike ADMM-Net / ISTA-Net it has no
unrolled optimization and no explicit data-consistency step; it simply learns a direct
image-to-image mapping from the zero-filled reconstruction. Comparing it to the unrolled
networks isolates the value of the model-based structure at a similar parameter budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_model
from ..utils import complex_to_chan, ifft2c


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        layers += [nn.BatchNorm2d(out_ch)] if use_bn else []
        layers += [nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        layers += [nn.BatchNorm2d(out_ch)] if use_bn else []
        layers += [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


@register_model("unet")
class UNet(nn.Module):
    """A compact 2-level U-Net operating on the zero-filled image (mask unused)."""

    def __init__(self, channels: int = 32, depth: int = 2, use_bn: bool = True, **_):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = 2  # complex image as (real, imag) channels
        feats = channels
        for _i in range(depth):
            self.downs.append(_DoubleConv(ch, feats, use_bn))
            ch = feats
            feats *= 2

        self.bottleneck = _DoubleConv(ch, feats, use_bn)

        for _i in range(depth):
            self.up_convs.append(nn.ConvTranspose2d(feats, feats // 2, 2, stride=2))
            self.ups.append(_DoubleConv(feats, feats // 2, use_bn))
            feats //= 2

        self.head = nn.Conv2d(feats, 2, 1)  # (real, imag) output

    def forward(self, undersampled_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = complex_to_chan(ifft2c(undersampled_kspace))

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        return self.head(x)
