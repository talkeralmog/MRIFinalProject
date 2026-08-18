# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""PSNR and SSIM on the magnitude image, alongside the per-channel numbers.

The per-channel convention comes from the complex-valued case: when an MR image carries
real phase, the real and imaginary parts are separate physical quantities and averaging
them hides which one failed. This dataset is not that case. The volumes are reconstructed
magnitude images, so the reference imaginary channel is identically zero and the reference
magnitude equals its real channel exactly (the real channel is non-negative).

That makes the magnitude the natural quantity to score here, for two reasons:

* it is what is actually displayed and read. A radiologist looks at |x|, not at Re(x);
* it is the only summary that charges a method for the spurious imaginary component the
  non-Hermitian mask injects. Scoring Re(x) alone silently ignores it; scoring Im(x)
  against a zero reference over-rewards any method free to output zero there. The
  magnitude combines both without either distortion.

So this module recomputes the headline comparison as
``|x| = sqrt(Re^2 + Im^2)`` versus the magnitude reference, over the full cached test split
at every sampling ratio and seed, and reports it next to the per-channel numbers so the
two conventions can be compared directly.

Usage::

    python -m src.eval_magnitude --ratios 0.2 0.3 0.5 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import model as _model  # noqa: F401
from .baselines import classical_cs as _cs  # noqa: F401
from .config import load_config
from .make_qualitative import load_cached_test_split
from .masks import build_mask
from .metrics import DATA_RANGE, psnr_channel, ssim_channel
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c, ifft2c, load_checkpoint

RATIOS = (0.2, 0.3, 0.5)
SEEDS = (0, 1, 2)

METHODS = (
    ("zero_filled", "Zero-filled input (no reconstruction)"),
    ("classical_cs", "Naive CS (single-level wavelet)"),
    ("classical_cs_tv", "Baseline: classical CS (wavelet + TV)"),
    ("admmnet_softthresh", "Our model: unrolled ADMM-Net"),
)


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """|x| from a 2-channel (real, imaginary) tensor, as ``(B, H, W)``."""
    return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)


@torch.no_grad()
def evaluate_run(checkpoint_path: str, cache_root: str = "cache", device: str = "cpu",
                 batch: int = 64) -> Tuple[Dict[str, Dict[str, List[float]]], float, int]:
    """Per-slice magnitude and real-channel metrics for every method on one run."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    baselines = {name: MODEL_REGISTRY.build(name).to(device).eval()
                 for name in ("classical_cs", "classical_cs_tv")}

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    slices, _ = load_cached_test_split(cache_root)
    acc: Dict[str, Dict[str, List[float]]] = {
        key: {"psnr_mag": [], "ssim_mag": [], "psnr_real": [], "ssim_real": []}
        for key, _ in METHODS}

    for start in range(0, slices.shape[0], batch):
        label = slices[start : start + batch].to(device)
        y = mask * fft2c(chan_to_complex(label))
        truth_mag = magnitude(label)

        outputs = {
            "zero_filled": torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1),
            "classical_cs": baselines["classical_cs"](y, mask).clamp(-1, 1),
            "classical_cs_tv": baselines["classical_cs_tv"](y, mask).clamp(-1, 1),
            "admmnet_softthresh": model(y, mask).clamp(-1, 1),
        }
        for key, out in outputs.items():
            acc[key]["psnr_mag"] += psnr_channel(magnitude(out), truth_mag).tolist()
            acc[key]["ssim_mag"] += ssim_channel(magnitude(out), truth_mag).tolist()
            acc[key]["psnr_real"] += psnr_channel(out[:, 0], label[:, 0]).tolist()
            acc[key]["ssim_real"] += ssim_channel(out[:, 0], label[:, 0]).tolist()

    return acc, float(cfg["mask"]["sampling_ratio"]), int(cfg["train"]["seed"])


def _find_checkpoint(results_root: str, ratio: float, seed: int) -> Optional[str]:
    for path in sorted(glob.glob(os.path.join(results_root, "*", "manifest.json"))):
        cfg = json.load(open(path))["config"]
        ckpt = os.path.join(os.path.dirname(path), "best.pth")
        if not os.path.exists(ckpt) or cfg.get("name") != "comparison":
            continue
        if not str(cfg.get("model", {}).get("name", "")).startswith("admmnet"):
            continue
        if int(cfg["train"]["seed"]) != seed:
            continue
        if abs(float(cfg["mask"]["sampling_ratio"]) - ratio) > 1e-9:
            continue
        return ckpt
    return None


def figure(rows: Sequence[Dict]) -> plt.Figure:
    """Magnitude metrics per ratio, and how they compare with the real-channel numbers."""
    ratios = sorted({float(r["sampling ratio"]) for r in rows})
    colours = {"zero_filled": "0.5", "classical_cs": "tab:purple",
               "classical_cs_tv": "tab:orange", "admmnet_softthresh": "tab:green"}

    def series(key: str, column: str) -> List[float]:
        return [statistics.fmean([float(r[column]) for r in rows
                                  if r["method key"] == key
                                  and float(r["sampling ratio"]) == ratio])
                for ratio in ratios]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

    for ax, column, label in ((axes[0], "psnr_mag mean", "PSNR on |x| (dB)"),
                              (axes[1], "ssim_mag mean", "SSIM on |x|")):
        for key, name in METHODS:
            ax.plot(ratios, series(key, column), marker="o", linewidth=2,
                    color=colours[key], label=name)
        ax.set_xlabel("fraction of phase-encode lines acquired")
        ax.set_ylabel(label)
        ax.set_xticks(ratios)
        ax.grid(alpha=0.3)
    axes[0].set_title("(a) PSNR on the magnitude image", fontsize=10.5)
    axes[1].set_title("(b) SSIM on the magnitude image", fontsize=10.5)
    axes[0].legend(fontsize=7.5)

    # (c) magnitude against real channel: how much the spurious phase costs each method
    ax = axes[2]
    width = 0.8 / len(METHODS)
    positions = np.arange(len(ratios))
    for i, (key, name) in enumerate(METHODS):
        delta = np.array(series(key, "psnr_mag mean")) - np.array(series(key, "psnr_real mean"))
        bars = ax.bar(positions + i * width - 0.4 + width / 2, delta, width,
                      color=colours[key], edgecolor="black", label=name)
        for bar, value in zip(bars, delta):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (0.12 if value >= 0 else -0.35), f"{value:+.1f}",
                    ha="center", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in ratios])
    ax.set_xlabel("fraction of phase-encode lines acquired")
    ax.set_ylabel("PSNR on |x| minus PSNR on Re(x)  (dB)")
    ax.set_title("(c) Cost of the spurious imaginary component\n"
                 "(negative = the magnitude metric charges for it)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Scoring the magnitude image, which is what is displayed and read, "
                 "rather than the real channel alone", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--ratios", nargs="*", type=float, default=list(RATIOS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> List[Dict]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    pooled: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
    for ratio in args.ratios:
        for seed in args.seeds:
            ckpt = _find_checkpoint(results_root, ratio, seed)
            if ckpt is None:
                print(f"skipping ratio {ratio} seed {seed}: no checkpoint")
                continue
            acc, actual, actual_seed = evaluate_run(ckpt, args.cache_root, args.device)
            for key, metrics in acc.items():
                bucket = pooled.setdefault((key, actual),
                                           {k: [] for k in metrics})
                for name, values in metrics.items():
                    bucket[name] += values
            print(f"  ratio={actual} seed={actual_seed}: "
                  f"ADMM-Net |x| PSNR "
                  f"{statistics.fmean(acc['admmnet_softthresh']['psnr_mag']):.2f} dB, "
                  f"SSIM {statistics.fmean(acc['admmnet_softthresh']['ssim_mag']):.4f}")

    rows: List[Dict] = []
    pretty = dict(METHODS)
    for (key, ratio), metrics in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row: Dict[str, object] = {"sampling ratio": ratio, "method key": key,
                                  "method": pretty[key], "n": len(metrics["psnr_mag"])}
        for name, values in metrics.items():
            row[f"{name} mean"] = round(statistics.fmean(values), 4)
            row[f"{name} std"] = round(statistics.stdev(values), 4)
        rows.append(row)

    if not rows:
        return rows
    path = os.path.join(out_dir, "mri_magnitude.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")

    fig = figure(rows)
    png = os.path.join(out_dir, "mri_magnitude.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}\n")

    print(f"{'ratio':>6} {'method':<40} {'PSNR |x|':>16} {'SSIM |x|':>18}")
    for row in rows:
        print(f"{row['sampling ratio']:>6} {row['method']:<40} "
              f"{row['psnr_mag mean']:>8.2f} ± {row['psnr_mag std']:<5.2f} "
              f"{row['ssim_mag mean']:>8.4f} ± {row['ssim_mag std']:<6.4f}")
    return rows


if __name__ == "__main__":
    main()
