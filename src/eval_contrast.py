# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Does the reconstruction preserve *tissue contrast*, not just pixel fidelity?

PSNR and SSIM -- the two metrics the brief requires -- are generic image-similarity
measures. A reconstruction can score well on both and still be clinically useless if it
has flattened the grey-matter/white-matter difference, because that difference is the
contrast a radiologist actually reads. The course makes the same distinction between SNR
and CNR: a high-SNR image with low CNR shows nothing.

This module measures, on every test slice, two MRI-native quantities and compares them
between the fully sampled reference, the classical baseline and our model:

``contrast``
    The relative white-matter/grey-matter contrast,
    ``C = (S_WM - S_GM) / S_WM``, using tissue masks derived from the *reference* image
    by intensity percentile (a threshold-based segmentation, as in the course's
    intensity-histogram lecture) so the same voxels are compared in all three images.
``CNR``
    ``(S_WM - S_GM) / sigma_background``, with the noise level estimated in the air
    outside the head. This is the quantity that decides whether the two tissues are
    actually distinguishable.

Reported as a ratio to the reference, so 1.0 means the reconstruction preserved the
contrast exactly and < 1.0 means it washed it out.

Usage::

    python -m src.eval_contrast --ratios 0.2 0.3 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import model as _model  # noqa: F401
from .baselines import classical_cs as _cs  # noqa: F401
from .config import load_config
from .display import to_display
from .make_qualitative import _find_checkpoint, load_cached_test_split
from .masks import build_mask
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c, ifft2c, load_checkpoint

# Percentile bands used to define the tissue masks on the reference image. In a
# T1-weighted slice the brightest in-brain voxels are white matter and the mid band is
# cortical/deep grey matter; the bands are deliberately narrow so partial-volume voxels
# at the tissue boundary are excluded from both.
BRAIN_THRESHOLD = 0.08          # in-brain voxels (the head fills far less than the FOV)
GM_BAND = (35.0, 55.0)          # percentiles of the in-brain intensity distribution
WM_BAND = (88.0, 99.0)


def tissue_masks(reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grey-matter, white-matter and background masks derived from the reference slice."""
    brain = reference > BRAIN_THRESHOLD
    inside = reference[brain]
    gm_lo, gm_hi = np.percentile(inside, GM_BAND)
    wm_lo, wm_hi = np.percentile(inside, WM_BAND)
    gm = brain & (reference >= gm_lo) & (reference <= gm_hi)
    wm = brain & (reference >= wm_lo) & (reference <= wm_hi)
    background = reference < 0.02
    return gm, wm, background


def contrast_and_cnr(image: np.ndarray, gm: np.ndarray, wm: np.ndarray,
                     background: np.ndarray) -> Tuple[float, float]:
    """Relative WM/GM contrast and WM-GM contrast-to-noise ratio for one image."""
    s_gm = float(image[gm].mean())
    s_wm = float(image[wm].mean())
    noise = float(image[background].std()) if background.sum() > 32 else float("nan")
    contrast = (s_wm - s_gm) / s_wm if s_wm != 0 else float("nan")
    cnr = (s_wm - s_gm) / noise if noise and np.isfinite(noise) and noise > 0 else float("nan")
    return contrast, cnr


@torch.no_grad()
def evaluate(checkpoint_path: str, baseline: str = "classical_cs_tv",
             cache_root: str = "cache", device: str = "cpu") -> Tuple[List[Dict], float]:
    """Per-slice contrast/CNR for the reference, the baseline and our model."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    ratio = float(cfg["mask"]["sampling_ratio"])

    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    base_model = MODEL_REGISTRY.build(baseline).to(device).eval()

    size = cfg["data"]["image_size"]
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    mask_params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **mask_params)

    slices, _ = load_cached_test_split(cache_root)

    def magnitude(x: torch.Tensor) -> np.ndarray:
        return to_display(torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).cpu().numpy())

    records: List[Dict] = []
    for i in range(slices.shape[0]):
        label = slices[i : i + 1].to(device)
        truth = magnitude(label)
        gm, wm, background = tissue_masks(truth)
        if gm.sum() < 64 or wm.sum() < 64 or background.sum() < 64:
            continue

        y = mask * fft2c(chan_to_complex(label))
        zf = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1)
        images = {
            "reference": truth,
            "zero_filled": magnitude(zf),
            "baseline": magnitude(base_model(y, mask).clamp(-1, 1)),
            "model": magnitude(model(y, mask).clamp(-1, 1)),
        }
        row: Dict[str, object] = {"index": i}
        for name, img in images.items():
            contrast, cnr = contrast_and_cnr(img, gm, wm, background)
            row[f"{name}_contrast"] = contrast
            row[f"{name}_cnr"] = cnr
        records.append(row)
    return records, ratio


def summarize(records: List[Dict], ratio: float) -> List[Dict]:
    """Mean contrast/CNR and their ratio to the fully sampled reference."""
    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in records], dtype=float)

    reference_contrast = np.nanmean(col("reference_contrast"))
    reference_cnr = np.nanmean(col("reference_cnr"))

    out: List[Dict] = []
    for name, label in (("reference", "fully sampled reference"),
                        ("zero_filled", "zero-filled input"),
                        ("baseline", "baseline: classical CS"),
                        ("model", "our model: ADMM-Net")):
        contrast = col(f"{name}_contrast")
        cnr = col(f"{name}_cnr")
        out.append({
            "sampling ratio": ratio,
            "method": label,
            "WM/GM contrast": round(float(np.nanmean(contrast)), 4),
            "contrast retained": round(float(np.nanmean(contrast) / reference_contrast), 3),
            "WM-GM CNR": round(float(np.nanmean(cnr)), 2),
            "CNR retained": round(float(np.nanmean(cnr) / reference_cnr), 3),
            "n slices": int(np.isfinite(contrast).sum()),
        })
    return out


def figure(all_rows: List[Dict]) -> plt.Figure:
    """Contrast and CNR retention against the reference, per sampling ratio."""
    ratios = sorted({r["sampling ratio"] for r in all_rows})
    methods = [("zero-filled input", "0.45"),
               ("baseline: classical CS", "tab:orange"),
               ("our model: ADMM-Net", "tab:green")]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
    width = 0.8 / len(methods)
    positions = np.arange(len(ratios))

    for ax, (key, title, ylabel) in zip(axes, (
            ("contrast retained", "(a) White-matter / grey-matter contrast retained",
             "reconstruction contrast / reference contrast"),
            ("CNR retained", "(b) White-matter to grey-matter CNR retained",
             "reconstruction CNR / reference CNR"))):
        for i, (label, colour) in enumerate(methods):
            values = [next((r[key] for r in all_rows
                            if r["method"] == label and r["sampling ratio"] == ratio), np.nan)
                      for ratio in ratios]
            bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width,
                          color=colour, edgecolor="black", label=label)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                            f"{value:.2f}", ha="center", fontsize=8)
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.3,
                   label="fully sampled reference")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{int(r * 100)}% of lines" for r in ratios])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle("Beyond PSNR/SSIM: is the tissue contrast a radiologist reads still "
                 "there?\n(tissue masks taken from the reference slice, so the same "
                 "voxels are compared in every image)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--ratios", nargs="*", type=float, default=[0.2, 0.3, 0.5])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default="classical_cs_tv")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> List[Dict]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict] = []
    for ratio in args.ratios:
        ckpt = _find_checkpoint(results_root, ratio, args.seed)
        if ckpt is None:
            print(f"skipping ratio {ratio}: no comparison ADMM-Net checkpoint")
            continue
        records, actual = evaluate(ckpt, args.baseline, args.cache_root, args.device)
        rows = summarize(records, actual)
        all_rows.extend(rows)
        for row in rows:
            print(f"  ratio={row['sampling ratio']} {row['method']:26s} "
                  f"contrast={row['WM/GM contrast']:.4f} ({row['contrast retained']:.2f}x)  "
                  f"CNR={row['WM-GM CNR']:7.2f} ({row['CNR retained']:.2f}x)  "
                  f"n={row['n slices']}")

    if all_rows:
        path = os.path.join(out_dir, "mri_contrast_retention.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {path}")
        fig = figure(all_rows)
        png = os.path.join(out_dir, "mri_contrast_retention.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")
    return all_rows


if __name__ == "__main__":
    main()
