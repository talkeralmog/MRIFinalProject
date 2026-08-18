# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Does the classical baseline do better at a larger matrix size?

The report's central caveat is that our classical compressed-sensing baseline recovers very
little because 1D Cartesian undersampling of a 128x128 image leaves the aliasing too
coherent for a hand-designed sparsity prior. That is an explanation, not a measurement, and
it makes a falsifiable prediction: at a larger matrix size the same image is *sparser*
relative to the sampling grid, so compressed-sensing recovery should improve.

This tests it. The baseline is training-free, so nothing has to be retrained: extract the
test split at 256x256 as well as 128x128, run the identical `classical_cs_tv` reconstruction
at both sizes and the same sampling ratios, and compare the gain each achieves over its own
zero-filled input. Comparing *gains* rather than absolute PSNR is the point, since the two
matrix sizes are different images and their absolute numbers are not comparable.

If the gain grows with matrix size, the report's explanation is confirmed. If it does not,
the explanation is wrong and we would rather know.

Run on the machine that has the dataset (the 256 cache has to be built from the volumes)::

    python -m src.eval_resolution --sizes 128 256 --ratios 0.2 0.3 0.5
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import model as _model  # noqa: F401
from .baselines import classical_cs as _cs  # noqa: F401
from .config import load_config
from .dataset import build_datasets
from .masks import build_mask
from .metrics import psnr_channel, ssim_channel
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c, ifft2c

RATIOS = (0.2, 0.3, 0.5)


@torch.no_grad()
def evaluate_size(cfg: Dict, size: int, ratios: Sequence[float], seed: int = 0,
                  baseline: str = "classical_cs_tv", limit: int = 120,
                  device: str = "cpu") -> List[Dict]:
    """Zero-filled and classical-CS metrics at one matrix size, over the test split."""
    cfg = {**cfg, "data": {**cfg["data"], "image_size": size}}
    datasets = build_datasets(cfg)          # builds/reuses a cache keyed on image_size
    testset = datasets["test"]
    n = min(limit, len(testset))
    model = MODEL_REGISTRY.build(baseline).to(device).eval()

    rows: List[Dict] = []
    for ratio in ratios:
        mask, _ = build_mask(cfg["mask"]["name"], (size, size), device,
                             sampling_ratio=ratio, std_scale=cfg["mask"]["std_scale"],
                             center_lines=cfg["mask"].get("center_lines", 0), seed=seed)
        acc = {"zf_psnr": [], "cs_psnr": [], "zf_ssim": [], "cs_ssim": []}
        for i in range(n):
            label = testset[i].unsqueeze(0).to(device)
            y = mask * fft2c(chan_to_complex(label))
            zf = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1)
            cs = model(y, mask).clamp(-1, 1)
            acc["zf_psnr"].append(psnr_channel(zf[:, 0], label[:, 0]).item())
            acc["cs_psnr"].append(psnr_channel(cs[:, 0], label[:, 0]).item())
            acc["zf_ssim"].append(ssim_channel(zf[:, 0], label[:, 0]).item())
            acc["cs_ssim"].append(ssim_channel(cs[:, 0], label[:, 0]).item())

        gain = statistics.fmean(acc["cs_psnr"]) - statistics.fmean(acc["zf_psnr"])
        ssim_gain = statistics.fmean(acc["cs_ssim"]) - statistics.fmean(acc["zf_ssim"])
        rows.append({
            "image size": size,
            "sampling ratio": ratio,
            "n slices": n,
            "zero-filled PSNR (dB)": round(statistics.fmean(acc["zf_psnr"]), 3),
            "classical CS PSNR (dB)": round(statistics.fmean(acc["cs_psnr"]), 3),
            "CS gain over zero-filling (dB)": round(gain, 3),
            "zero-filled SSIM": round(statistics.fmean(acc["zf_ssim"]), 4),
            "classical CS SSIM": round(statistics.fmean(acc["cs_ssim"]), 4),
            "CS SSIM gain": round(ssim_gain, 4),
        })
        print(f"  {size}x{size}, {int(ratio * 100):>2}% lines: "
              f"zero-filled {rows[-1]['zero-filled PSNR (dB)']:6.2f} dB -> "
              f"CS {rows[-1]['classical CS PSNR (dB)']:6.2f} dB "
              f"(gain {gain:+.2f} dB, SSIM {ssim_gain:+.4f})")
    return rows


def figure(rows: Sequence[Dict]) -> plt.Figure:
    """The CS gain over zero-filling, per matrix size, as a function of sampling ratio."""
    sizes = sorted({int(r["image size"]) for r in rows})
    ratios = sorted({float(r["sampling ratio"]) for r in rows})
    colours = dict(zip(sizes, ["tab:blue", "tab:green", "tab:purple", "tab:orange"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, key, label in ((axes[0], "CS gain over zero-filling (dB)",
                            "classical CS gain over its own zero-filled input (dB)"),
                           (axes[1], "CS SSIM gain", "classical CS SSIM gain")):
        for size in sizes:
            values = [next(float(r[key]) for r in rows
                           if int(r["image size"]) == size
                           and float(r["sampling ratio"]) == ratio)
                      for ratio in ratios]
            ax.plot(ratios, values, marker="o", linewidth=2, color=colours[size],
                    label=f"{size}x{size}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("fraction of phase-encode lines acquired")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(ratios)
        ax.legend(fontsize=8, title="matrix size")
        ax.grid(alpha=0.3)
    axes[0].set_title("(a) PSNR gain from the sparsity prior", fontsize=10.5)
    axes[1].set_title("(b) SSIM gain from the sparsity prior", fontsize=10.5)

    fig.suptitle("Testing the report's explanation: compressed sensing should help more at a "
                 "larger matrix size,\nwhere the same anatomy is sparser relative to the "
                 "sampling grid", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--sizes", nargs="*", type=int, default=[128, 256])
    p.add_argument("--ratios", nargs="*", type=float, default=list(RATIOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=120,
                   help="Test slices per cell. The TV solver is the slow part.")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> List[Dict]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    out_dir = os.path.join(cfg["paths"]["results_root"], "figures")
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict] = []
    for size in args.sizes:
        print(f"\n=== {size}x{size} ===")
        rows.extend(evaluate_size(cfg, size, args.ratios, args.seed,
                                  limit=args.limit, device=args.device))

    path = os.path.join(out_dir, "mri_resolution.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")

    fig = figure(rows)
    png = os.path.join(out_dir, "mri_resolution.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")

    print("\n--- verdict ---")
    for ratio in args.ratios:
        gains = {int(r["image size"]): float(r["CS gain over zero-filling (dB)"])
                 for r in rows if float(r["sampling ratio"]) == ratio}
        if len(gains) > 1:
            small, large = min(gains), max(gains)
            delta = gains[large] - gains[small]
            verdict = ("CONFIRMS the report's explanation" if delta > 0.3
                       else "does NOT confirm it" if delta < 0.1 else "is inconclusive")
            print(f"  {int(ratio * 100)}% lines: gain {gains[small]:+.2f} dB at {small} vs "
                  f"{gains[large]:+.2f} dB at {large} ({delta:+.2f} dB) -> {verdict}")
    return rows


if __name__ == "__main__":
    main()
