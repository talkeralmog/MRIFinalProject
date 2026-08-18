# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Two physics constraints the trained network does not enforce, applied afterwards.

Auditing our own model for the report turned up two places where its output violates
something we know about the acquisition. Both can be repaired by projecting the finished
reconstruction, with no retraining, so we measure the effect of each here rather than
leave the weakness unstated.

**1. Data consistency is soft, not exact.** The ADMM-Net X-update blends the estimate with
the measurement on the acquired lines as ``(y + rho*Fx) / (1 + rho)``. That is the correct
ADMM update for a noisy measurement, but it means the reconstruction does *not* reproduce
the lines the scanner actually acquired: with the values of ``rho`` the networks learned
(0.01 to 6.1 across stages), the relative error on the acquired lines reaches ~5%. The
classical baseline's POCS step, by contrast, is exact by construction. A single hard
projection at the end restores exactness. Since our forward model is noiseless -- we
generated the measurements ourselves with an FFT -- there is no reason to keep the
soft version at inference.

**2. The reconstruction is not constrained to be real.** The course volumes are
reconstructed magnitude images, so the target is real-valued and its k-space is
conjugate-symmetric. Nothing in the network knows this, and its output carries a non-zero
imaginary part. Projecting onto real images enforces the constraint. This is the
image-domain form of the Hermitian symmetry that partial-Fourier reconstruction exploits.

The two are applied in this order: enforce data consistency, then take the real part. The
second projection breaks the first slightly (discarding the imaginary part perturbs
k-space), so ``apply`` re-projects when both are requested.

Usage::

    python -m src.postprocess --ratios 0.2 0.3 0.5
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

from . import model as _model  # noqa: F401
from .baselines import classical_cs as _cs  # noqa: F401
from .config import load_config
from .make_qualitative import _find_checkpoint, load_cached_test_split
from .masks import build_mask
from .metrics import psnr_channel, ssim_channel
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, complex_to_chan, fft2c, ifft2c, load_checkpoint

RATIOS = (0.2, 0.3, 0.5)
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# The two projections
# ---------------------------------------------------------------------------


def enforce_data_consistency(image: torch.Tensor, measured_kspace: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """Replace the acquired k-space lines with the measured values (hard POCS).

    Leaves the unacquired lines -- everything the network actually had to infer --
    untouched, so this can only move the estimate towards the measurement.
    """
    kspace = fft2c(chan_to_complex(image))
    return complex_to_chan(ifft2c(mask * measured_kspace + (1.0 - mask) * kspace))


def enforce_real(image: torch.Tensor) -> torch.Tensor:
    """Project onto real-valued images by discarding the imaginary channel.

    Valid only when the target is known to be real, as it is for this dataset. In k-space
    this is exactly the statement that the spectrum should be conjugate-symmetric.
    """
    return torch.cat([image[:, 0:1], torch.zeros_like(image[:, 1:2])], dim=1)


def apply(image: torch.Tensor, measured_kspace: torch.Tensor, mask: torch.Tensor,
          data_consistency: bool = True, real_valued: bool = True) -> torch.Tensor:
    """Apply the requested projections in the order that leaves both (nearly) satisfied."""
    out = image
    if data_consistency:
        out = enforce_data_consistency(out, measured_kspace, mask)
    if real_valued:
        out = enforce_real(out)
        if data_consistency:
            # Taking the real part perturbs k-space, so restore the measured lines again.
            out = enforce_data_consistency(out, measured_kspace, mask)
    return out


def kspace_error(image: torch.Tensor, measured_kspace: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Relative L2 error on the acquired lines: 0 means the measurement is reproduced."""
    kspace = fft2c(chan_to_complex(image))
    num = (mask * (kspace - measured_kspace)).abs().pow(2).sum(dim=(1, 2, 3))
    den = (mask * measured_kspace).abs().pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return (num / den).sqrt()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VARIANTS = (
    ("as trained", dict(data_consistency=False, real_valued=False)),
    ("+ exact data consistency", dict(data_consistency=True, real_valued=False)),
    ("+ real-valued constraint", dict(data_consistency=True, real_valued=True)),
)


@torch.no_grad()
def evaluate_run(checkpoint_path: str, cache_root: str = "cache",
                 device: str = "cpu", batch: int = 64) -> Tuple[Dict[str, Dict], float, int]:
    """Metrics for each post-processing variant on one trained network."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    slices, _ = load_cached_test_split(cache_root)
    acc: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k in ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag", "kerr")}
        for name, _ in VARIANTS}

    for start in range(0, slices.shape[0], batch):
        label = slices[start : start + batch].to(device)
        y = mask * fft2c(chan_to_complex(label))
        raw = model(y, mask).clamp(-1, 1)
        for name, opts in VARIANTS:
            out = apply(raw, y, mask, **opts).clamp(-1, 1)
            acc[name]["psnr_real"] += psnr_channel(out[:, 0], label[:, 0]).tolist()
            acc[name]["psnr_imag"] += psnr_channel(out[:, 1], label[:, 1]).tolist()
            acc[name]["ssim_real"] += ssim_channel(out[:, 0], label[:, 0]).tolist()
            acc[name]["ssim_imag"] += ssim_channel(out[:, 1], label[:, 1]).tolist()
            acc[name]["kerr"] += kspace_error(out, y, mask).tolist()

    summary = {name: {k: float(np.mean(v)) for k, v in metrics.items()}
               for name, metrics in acc.items()}
    return summary, float(cfg["mask"]["sampling_ratio"]), int(cfg["train"]["seed"])


def figure(rows: Sequence[Dict]) -> plt.Figure:
    """What each projection changes, per sampling ratio."""
    ratios = sorted({r["sampling ratio"] for r in rows})
    names = [n for n, _ in VARIANTS]
    colours = ["0.55", "tab:blue", "tab:green"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
    positions = np.arange(len(ratios))
    width = 0.8 / len(names)

    def series(name: str, key: str) -> List[float]:
        return [float(np.mean([r[key] for r in rows
                               if r["variant"] == name and r["sampling ratio"] == ratio]))
                for ratio in ratios]

    # (a) consistency with the measurement -- the defect being repaired
    ax = axes[0]
    for i, (name, colour) in enumerate(zip(names, colours)):
        values = [max(v, 1e-9) for v in series(name, "kerr")]
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width,
                      color=colour, edgecolor="black", label=name)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.4,
                    f"{value:.1e}" if value < 1e-3 else f"{value:.1%}",
                    ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in ratios])
    ax.set_ylabel("relative error on the acquired lines (log)")
    ax.set_title("(a) Fidelity to the measured k-space lines\n(lower is better; the projection reaches machine precision)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3, which="both")

    # (b) effect on the real channel -- essentially none
    ax = axes[1]
    for i, (name, colour) in enumerate(zip(names, colours)):
        values = series(name, "psnr_real")
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width,
                      color=colour, edgecolor="black", label=name)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}",
                    ha="center", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in ratios])
    ax.set_ylabel("PSNR, real channel (dB)")
    ax.set_ylim(0, max(series(names[-1], "psnr_real")) * 1.25)
    ax.set_title("(b) Effect on the real channel\n(small, and always positive)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (c) effect on the imaginary channel -- the metric collapses
    ax = axes[2]
    for i, (name, colour) in enumerate(zip(names, colours)):
        values = series(name, "psnr_imag")
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width,
                      color=colour, edgecolor="black", label=name)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}",
                    ha="center", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in ratios])
    ax.set_ylabel("PSNR, imaginary channel (dB)")
    ax.set_title("(c) Effect on the imaginary channel\n(the metric saturates, the image does not improve)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Two constraints the network does not enforce, applied afterwards "
                 "without retraining", fontsize=12.5)
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

    rows: List[Dict] = []
    for ratio in args.ratios:
        for seed in args.seeds:
            ckpt = _find_checkpoint(results_root, ratio, seed)
            if ckpt is None:
                print(f"skipping ratio {ratio} seed {seed}: no checkpoint")
                continue
            summary, actual_ratio, actual_seed = evaluate_run(
                ckpt, args.cache_root, args.device)
            for name, metrics in summary.items():
                rows.append({"sampling ratio": actual_ratio, "seed": actual_seed,
                             "variant": name, **{k: round(v, 6) for k, v in metrics.items()}})
            base = summary["as trained"]
            fixed = summary["+ real-valued constraint"]
            print(f"  ratio={actual_ratio} seed={actual_seed}: "
                  f"PSNR_real {base['psnr_real']:.2f} -> {fixed['psnr_real']:.2f}, "
                  f"PSNR_imag {base['psnr_imag']:.2f} -> {fixed['psnr_imag']:.2f}, "
                  f"k-space error {base['kerr']:.2%} -> {fixed['kerr']:.1e}")

    if not rows:
        return rows
    path = os.path.join(out_dir, "mri_postprocess.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    fig = figure(rows)
    png = os.path.join(out_dir, "mri_postprocess.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    return rows


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Per-stage audit: which unrolled stages actually do anything
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_activity(checkpoint_path: str, cache_root: str = "cache", device: str = "cpu",
                   limit: int = 96, batch: int = 32) -> Tuple[List[Dict], float, int]:
    """How often each unrolled stage applies a correction, and what it contributes.

    The synthesis block of ``CustomADMMStage`` ends in a ReLU and has no residual path
    (its input and output channel counts differ), so the correction a stage adds to the
    image is non-negative by construction. A stage that would need to subtract therefore
    saturates to exactly zero and becomes an identity map. This measures how often that
    happens, per stage, so the report can quote it rather than assert it.
    """
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    slices, _ = load_cached_test_split(cache_root)
    n_slices = min(limit, slices.shape[0])
    n_stages = len(model.stages)
    dead = np.zeros(n_stages)
    gain = np.zeros(n_stages)

    for start in range(0, n_slices, batch):
        label = slices[start : start + batch].to(device)
        count = label.shape[0]
        y = mask * fft2c(chan_to_complex(label))
        x = complex_to_chan(ifft2c(y))
        z = torch.zeros(count, model.channels, size, size, device=device, dtype=x.dtype)
        m = torch.zeros_like(z)
        previous = psnr_channel(x[:, 0].clamp(-1, 1), label[:, 0])
        for index, stage in enumerate(model.stages):
            c = stage.analysis(x)
            z_new = stage.nonlinearity(c + m)
            m_new = m + c - z_new
            correction = stage.synthesis(z_new - m_new)
            x = stage.data_consistency(x + correction, y, mask)
            z, m = z_new, m_new
            dead[index] += float((correction.abs().amax(dim=(1, 2, 3)) == 0).sum())
            current = psnr_channel(x[:, 0].clamp(-1, 1), label[:, 0])
            gain[index] += float((current - previous).sum())
            previous = current

    rows = [{
        "sampling ratio": float(cfg["mask"]["sampling_ratio"]),
        "seed": int(cfg["train"]["seed"]),
        "stage": index + 1,
        "dead fraction": round(dead[index] / n_slices, 4),
        "mean PSNR gain (dB)": round(gain[index] / n_slices, 4),
        "n slices": n_slices,
    } for index in range(n_stages)]
    return rows, float(cfg["mask"]["sampling_ratio"]), int(cfg["train"]["seed"])


def stage_activity_main(argv=None) -> List[Dict]:
    """Write ``results/figures/mri_stage_activity.csv`` for every comparison checkpoint."""
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict] = []
    for ratio in args.ratios:
        for seed in args.seeds:
            ckpt = _find_checkpoint(results_root, ratio, seed)
            if ckpt is None:
                continue
            run_rows, actual_ratio, actual_seed = stage_activity(
                ckpt, args.cache_root, args.device)
            rows.extend(run_rows)
            dead = [r["dead fraction"] for r in run_rows]
            print(f"  ratio={actual_ratio} seed={actual_seed}: dead fraction per stage = "
                  + " ".join(f"{v:.2f}" for v in dead))

    if not rows:
        return rows
    path = os.path.join(out_dir, "mri_stage_activity.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    n_stages = max(r["stage"] for r in rows)
    print("\npooled over all runs:")
    for stage in range(1, n_stages + 1):
        subset = [r for r in rows if r["stage"] == stage]
        dead = 100 * float(np.mean([r["dead fraction"] for r in subset]))
        gain = float(np.mean([r["mean PSNR gain (dB)"] for r in subset]))
        print(f"  stage {stage}: dead on {dead:5.1f}% of slices, mean gain {gain:+.2f} dB")
    return rows
