# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Typeset the report's display equations as images, using LaTeX notation.

The report body is rendered by ReportLab and python-docx, neither of which typesets
mathematics. Equations written as inline markup (``T<sub>scan</sub> = ...``) are legible but
plainly not typeset, and the more involved ones -- the ADMM data-consistency update, the
PSNR definition -- do not survive it at all.

Matplotlib's mathtext understands a large subset of LaTeX and needs no TeX installation, so
each display equation is rendered once here to a tight, transparent PNG at high resolution
and embedded in the report as a small centred figure. Inline symbols stay as markup: mixing
image fragments into a line of prose would wreck the baseline.

Every equation the report displays is defined in ``EQUATIONS`` below, so the notation has a
single source and cannot drift between the PDF and the DOCX.

Usage::

    python -m src.figures_equations
    python -m src.figures_equations --which forward_model psnr
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

#: name -> LaTeX source for every equation the report displays.
EQUATIONS: Dict[str, str] = {
    # The acquisition: what undersampling actually buys.
    "scan_time":
        r"$T_{\mathrm{scan}} \;=\; N_{\mathrm{PE}} \times \mathrm{TR} \times \mathrm{NSA}$",

    # Spatial encoding: the gradient makes the Larmor frequency position-dependent.
    "larmor":
        r"$\nu(\mathbf{r}) \;=\; \gamma\,\left(B_0 + \mathbf{G}\cdot\mathbf{r}\right)$",

    # The forward model we invert.
    "forward_model":
        r"$\mathbf{y} \;=\; \mathbf{M}\odot\mathcal{F}\{x\}\,,\qquad$"
        r"$\mathbf{M}\in\{0,1\}^{N\times N}$",

    # Field of view in the unprotected direction.
    "fov":
        r"$\mathrm{FOV}_y \;=\; 1/\Delta k_y$",

    # Why fewer lines cost SNR.
    "snr":
        r"$\mathrm{SNR} \;\propto\; \sqrt{N_{\mathrm{samples}}}$",

    # Hermitian symmetry of a real-valued image, which partial-Fourier exploits.
    "hermitian":
        r"$F(-\mathbf{k}) \;=\; F^{*}(\mathbf{k})$",

    # The reconstruction problem both methods solve, in its regularized form.
    "objective":
        r"$\hat{x} \;=\; \arg\min_{x}\;"
        r"\frac{1}{2}\| \mathbf{M}\odot\mathcal{F}\{x\} - \mathbf{y}\|_2^2"
        r"\;+\;\lambda\| \Psi x\|_1 \;+\;\tau\,\mathrm{TV}(x)$",

    # The classical baseline's data-consistency projection (hard POCS).
    "pocs":
        r"$\mathcal{F}\{x^{(k+1)}\} \;=\; \mathbf{M}\odot\mathbf{y}"
        r"\;+\;(1-\mathbf{M})\odot\mathcal{F}\{x^{(k)}\}$",

    # Our model's learned X-update: the same projection, softened by a learned rho.
    "data_consistency":
        r"$\mathcal{F}\{x_{\mathrm{new}}\} \;=\; \mathbf{M}\odot"
        r"\frac{\mathbf{y} + \rho\,\mathcal{F}\{x\}}{1+\rho}"
        r"\;+\;(1-\mathbf{M})\odot\mathcal{F}\{x\}$",

    # The learned proximal operator that replaces wavelet shrinkage.
    "soft_threshold":
        r"$\mathrm{prox}_{t}(x) \;=\; \mathrm{sign}(x)\,\max\left(|x| - t,\;0\right)$",

    # How the required metrics are computed.
    "psnr":
        r"$\mathrm{PSNR} \;=\; 20\log_{10}(\mathrm{peak}) \;-\; 10\log_{10}(\mathrm{MSE})$",

    "magnitude":
        r"$|x| \;=\; \sqrt{\mathrm{Re}(x)^2 + \mathrm{Im}(x)^2}$",

    # The two MRI-native measures we added beyond PSNR/SSIM.
    "contrast":
        r"$C \;=\; \dfrac{S_{\mathrm{WM}} - S_{\mathrm{GM}}}{S_{\mathrm{WM}}}"
        r"\,,\qquad \mathrm{CNR} \;=\; "
        r"\dfrac{S_{\mathrm{WM}} - S_{\mathrm{GM}}}{\sigma_{\mathrm{background}}}$",
}


def render(latex: str, out_path: str, fontsize: int = 17, dpi: int = 320) -> str:
    """Render one LaTeX display equation to a tight, transparent PNG."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, latex, fontsize=fontsize, color="#1a1a1a")
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=os.path.join("results", "figures", "equations"))
    p.add_argument("--which", nargs="*", default=sorted(EQUATIONS),
                   choices=sorted(EQUATIONS))
    p.add_argument("--fontsize", type=int, default=17)
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    written: List[str] = []
    for name in args.which:
        path = os.path.join(args.out_dir, f"eq_{name}.png")
        render(EQUATIONS[name], path, args.fontsize)
        written.append(path)
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
