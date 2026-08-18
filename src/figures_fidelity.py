# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Direct comparison of a reconstruction against the fully sampled ground truth.

The qualitative panel in ``src/make_qualitative.py`` shows whole images side by side,
which answers "does it look right". This module answers the sharper question -- *where*
and *in what way* the reconstruction differs from the truth -- with four views that a
single-number metric cannot give:

``zoom``
    A magnified region over the ventricles and cortex, where the grey/white boundary and
    the fine sulcal detail live, with the error map on the same scale.
``profile``
    An intensity profile along one line through the brain. This is the readout of
    resolution and contrast: blurring shows as rounded edges, a wrong signal level shows
    as a vertical offset, and residual aliasing shows as ripple in the background.
``spectrum``
    The k-space magnitude spectrum of truth and reconstruction along the phase-encode
    axis, with the acquired lines marked. This separates the two things a reconstruction
    can do -- reproduce what was measured, and infer what was not -- and shows the
    frequency at which the inference stops working.
``joint``
    The voxel-wise joint histogram of reconstruction against truth. A perfect
    reconstruction is the identity line; blurring pulls the cloud towards the mean,
    and any systematic gain or bias shows as a rotation or shift.

Usage::

    python -m src.figures_fidelity --ratios 0.2 0.3 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from . import model as _model  # noqa: F401
from .baselines import classical_cs as _cs  # noqa: F401
from .config import load_config
from .display import to_display
from .make_qualitative import _find_checkpoint, load_cached_test_split
from .masks import build_mask
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c, ifft2c, load_checkpoint

METHODS = (
    ("zero-filled input", "0.45"),
    ("baseline: classical CS", "tab:orange"),
    ("our model: ADMM-Net", "tab:green"),
)


@torch.no_grad()
def reconstruct(checkpoint_path: str, sample_index: int, baseline: str = "classical_cs_tv",
                cache_root: str = "cache", device: str = "cpu"):
    """Ground truth plus the three reconstructions of one slice, in viewing orientation."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    base_model = MODEL_REGISTRY.build(baseline).to(device).eval()

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    slices, _ = load_cached_test_split(cache_root)
    label = slices[sample_index : sample_index + 1].to(device)
    y = mask * fft2c(chan_to_complex(label))

    def magnitude(x: torch.Tensor) -> np.ndarray:
        return to_display(torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).cpu().numpy())

    zf = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1)
    images = {
        "truth": magnitude(label),
        "zero-filled input": magnitude(zf),
        "baseline: classical CS": magnitude(base_model(y, mask).clamp(-1, 1)),
        "our model: ADMM-Net": magnitude(model(y, mask).clamp(-1, 1)),
    }
    row_mask = mask[0, 0, :, 0].cpu().numpy()
    return images, row_mask, float(cfg["mask"]["sampling_ratio"])


def _ky_spectrum(image: np.ndarray) -> np.ndarray:
    """Mean |F| per phase-encode line, centred. The image is in viewing orientation, so
    the phase-encode axis is the *second* one (see src/display.py)."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.abs(spectrum).mean(axis=0)


def fidelity_figure(images: Dict[str, np.ndarray], row_mask: np.ndarray, ratio: float,
                    sample_index: int, zoom: Tuple[int, int, int, int] = (38, 30, 56, 62)
                    ) -> Tuple[plt.Figure, List[Dict]]:
    """The four comparison views for one slice."""
    truth = images["truth"]
    vmax = float(truth.max()) or 1.0
    y0, x0, h, w = zoom
    profile_row = y0 + h // 2

    fig = plt.figure(figsize=(16.0, 9.4))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.42, wspace=0.24)

    # --- row 1: the images, with the zoom box and the profile line marked ----
    order = ["truth"] + [name for name, _ in METHODS]
    titles = {"truth": "ground truth (fully sampled)"}
    for name, _ in METHODS:
        titles[name] = name
    for col, name in enumerate(order):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(images[name], cmap="gray", vmin=0, vmax=vmax)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="tab:cyan",
                               linewidth=1.2))
        ax.axhline(profile_row, color="tab:red", linewidth=0.9, alpha=0.85)
        ax.axis("off")
        if name == "truth":
            ax.set_title(titles[name], fontsize=9.5)
        else:
            err = float(np.sqrt(((images[name] - truth) ** 2).mean()))
            ax.set_title(f"{titles[name]}\nRMSE {err:.4f}", fontsize=9.5)

    # --- row 2 left: zoomed detail -------------------------------------------
    ax = fig.add_subplot(grid[1, 0])
    crop = truth[y0 : y0 + h, x0 : x0 + w]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title("zoom: ground truth", fontsize=9)
    ax.axis("off")
    for col, (name, _) in enumerate(METHODS, start=1):
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(images[name][y0 : y0 + h, x0 : x0 + w], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"zoom: {name.split(':')[-1].strip()}", fontsize=9)
        ax.axis("off")

    # --- row 3 left: intensity profile ---------------------------------------
    ax = fig.add_subplot(grid[2, :2])
    xs = np.arange(truth.shape[1])
    ax.plot(xs, truth[profile_row], color="black", linewidth=2.0, label="ground truth")
    for name, colour in METHODS:
        ax.plot(xs, images[name][profile_row], color=colour, linewidth=1.2, alpha=0.9,
                label=name)
    ax.axvspan(x0, x0 + w, color="tab:cyan", alpha=0.12)
    ax.set_xlabel("position along the red line (pixels, phase-encode direction)")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Intensity profile through the slice: edge sharpness, signal level and "
                 "background ripple", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)

    # --- row 3 right: k-space spectrum ---------------------------------------
    ax = fig.add_subplot(grid[2, 2:])
    n = truth.shape[1]
    offsets = np.arange(n) - n // 2
    acquired = np.fft.fftshift(row_mask) > 0
    ax.semilogy(offsets, _ky_spectrum(truth), color="black", linewidth=2.0,
                label="ground truth")
    for name, colour in METHODS:
        ax.semilogy(offsets, _ky_spectrum(images[name]), color=colour, linewidth=1.1,
                    alpha=0.9, label=name)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(offsets, ymin, ymax, where=acquired, color="0.85", alpha=0.55,
                    step="mid", zorder=0, label="acquired lines")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$k_y$ relative to the centre of k-space")
    ax.set_ylabel("mean |F| per line (log)")
    ax.set_title("k-space spectrum: on the shaded lines the data was measured; "
                 "elsewhere it was inferred", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"Reconstruction against ground truth, test slice #{sample_index} at "
                 f"{int(round(ratio * 100))}% of phase-encode lines", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # --- the numbers behind the spectrum panel, split by acquired / inferred --
    truth_spec = _ky_spectrum(truth)
    rows: List[Dict] = []
    for name, _ in METHODS:
        spec = _ky_spectrum(images[name])
        for label, sel in (("acquired", acquired), ("inferred", ~acquired)):
            rel = np.abs(spec[sel] - truth_spec[sel]).sum() / truth_spec[sel].sum()
            rows.append({"sampling ratio": ratio, "slice": sample_index, "method": name,
                         "lines": label, "relative spectrum error": round(float(rel), 4)})
    return fig, rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--ratios", nargs="*", type=float, default=[0.3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slice-index", type=int, default=151)
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    written: List[str] = []
    all_rows: List[Dict] = []
    for ratio in args.ratios:
        ckpt = _find_checkpoint(results_root, ratio, args.seed)
        if ckpt is None:
            print(f"skipping ratio {ratio}: no checkpoint")
            continue
        images, row_mask, actual = reconstruct(ckpt, args.slice_index,
                                               cache_root=args.cache_root,
                                               device=args.device)
        fig, rows = fidelity_figure(images, row_mask, actual, args.slice_index)
        all_rows.extend(rows)
        stem = ("mri_fidelity" if len(args.ratios) == 1
                else f"mri_fidelity_r{int(round(actual * 100))}")
        path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}")
        for r in rows:
            print(f"   {r['method']:<24} {r['lines']:<9} "
                  f"spectrum error {r['relative spectrum error']:.3f}")

    if all_rows:
        path = os.path.join(out_dir, "mri_fidelity.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
