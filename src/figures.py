# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Report figures that describe the *method and the data*, rather than the results.

``src/analysis.py`` turns the logged metrics into the required results plots. This module
covers the MRI-specific figures the report needs alongside them:

``masks``
    The 1D variable-density undersampling patterns at 20% / 30% / 50%, plus the
    row-sampling density, showing that low spatial frequencies are densely sampled.
``pipelines``
    Block diagrams of the two compared reconstruction pipelines.
``eda``
    Exploratory data analysis: example central slices, the age distribution of the three
    splits, and a fully sampled vs undersampled k-space / image comparison that makes the
    aliasing artifact visible.
``per_stage``
    The reconstruction as it evolves through the unrolled ADMM stages -- the clearest
    single illustration of what the unrolling actually does.

``masks`` and ``pipelines`` need no dataset and no checkpoint, so they can be produced on
any machine; ``eda`` needs the dataset and ``per_stage`` needs a trained checkpoint.

Usage::

    python -m src.figures --config configs/default.yaml
    python -m src.figures --config configs/default.yaml --which masks pipelines
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # write files without needing a display (HPC / headless)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import load_config
from .display import to_display
from .masks import build_mask, sampling_rate
from .utils import chan_to_complex, fft2c, ifft2c, load_checkpoint

RATIOS = (0.2, 0.3, 0.5)


def _figure_dir(cfg: Dict) -> str:
    path = os.path.join(cfg["paths"]["results_root"], "figures")
    os.makedirs(path, exist_ok=True)
    return path


def _magnitude(chan: torch.Tensor) -> np.ndarray:
    """Magnitude image in viewing orientation, from a 2-channel tensor ``(1, 2, H, W)``."""
    x = chan.detach().cpu()
    return to_display(torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).numpy())


# ---------------------------------------------------------------------------
# 1. Undersampling masks (no data required)
# ---------------------------------------------------------------------------


def mask_overview(cfg: Dict, ratios: Sequence[float] = RATIOS) -> plt.Figure:
    """Show the k-space sampling pattern and its density at each sampling ratio."""
    size = cfg["data"]["image_size"]
    name = cfg["mask"]["name"]
    params = {k: v for k, v in cfg["mask"].items() if k not in ("name", "sampling_ratio")}
    params.setdefault("seed", cfg["train"]["seed"])

    fig, axes = plt.subplots(2, len(ratios), figsize=(4 * len(ratios), 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, ratio in enumerate(ratios):
        _, centered = build_mask(name, (size, size), sampling_ratio=ratio, **params)

        # Shown in the same orientation as every brain image in the report (see
        # src/display.py), so the phase-encode axis is horizontal in both.
        axes[0, col].imshow(to_display(centered), cmap="gray", aspect="equal")
        axes[0, col].set_title(f"{int(round(ratio * 100))}% of phase-encode lines\n"
                               f"(actual {sampling_rate(centered) * 100:.1f}%)")
        axes[0, col].set_xlabel("$k_y$ (phase encode)")
        if col == 0:
            axes[0, col].set_ylabel("$k_x$ (readout)")

        # Per-row sampling probability, averaged over many mask realizations. A single
        # realization is too sparse to read; the average is what shows the design intent,
        # i.e. that low spatial frequencies are almost always acquired.
        offsets = np.arange(size) - (size - 1) / 2.0
        probability = np.zeros(size)
        realizations = 200
        for extra_seed in range(realizations):
            draw = {**params, "seed": extra_seed}
            _, one = build_mask(name, (size, size), sampling_ratio=ratio, **draw)
            probability += one[:, 0]
        probability /= realizations

        axes[1, col].fill_between(offsets, probability, color="0.75", edgecolor="0.3",
                                  linewidth=0.8, label=f"mean over {realizations} masks")
        axes[1, col].axhline(ratio, color="crimson", linestyle="--", linewidth=1.2,
                             label=f"uniform ({int(round(ratio * 100))}%)")
        axes[1, col].set_xlabel("$k_y$ relative to the centre of k-space")
        axes[1, col].set_ylim(0, 1.05)
        if col == 0:
            axes[1, col].set_ylabel("P(row sampled)")
        axes[1, col].legend(fontsize=7.5, loc="upper right")
        axes[1, col].grid(alpha=0.3)

    fig.suptitle("1D variable-density Cartesian undersampling: rows drawn from a normal "
                 "distribution centred on the middle of k-space", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Pipeline diagrams (no data required)
# ---------------------------------------------------------------------------


def _draw_flow(ax, boxes: List[str], title: str, loop_label: Optional[str] = None,
               loop_span: Optional[Sequence[int]] = None) -> None:
    """Draw a left-to-right chain of labelled boxes, optionally marking a repeated block."""
    n = len(boxes)
    width, height, gap = 1.0, 0.5, 0.42
    # Longer chains get a smaller font so the labels stay inside their boxes.
    fontsize = 8.5 if n <= 8 else 7.2
    for i, label in enumerate(boxes):
        x = i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, 0), width, height,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    linewidth=1.3, edgecolor="black", facecolor="0.93"))
        ax.text(x + width / 2, height / 2, label, ha="center", va="center",
                fontsize=fontsize, wrap=True)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + width, height / 2),
                                        (x + width + gap, height / 2),
                                        arrowstyle="-|>", mutation_scale=12,
                                        linewidth=1.1, color="black"))

    if loop_label and loop_span:
        lo, hi = loop_span
        x0 = lo * (width + gap) - 0.09
        x1 = hi * (width + gap) + width + 0.09
        ax.add_patch(FancyBboxPatch((x0, -0.16), x1 - x0, height + 0.32,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.2, edgecolor="0.35", facecolor="none",
                                    linestyle="--"))
        ax.text((x0 + x1) / 2, height + 0.24, loop_label, ha="center", va="bottom",
                fontsize=9, color="0.25")

    ax.set_xlim(-0.25, n * (width + gap) + 0.1)
    ax.set_ylim(-0.4, height + 0.62)
    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")


def pipeline_diagrams() -> plt.Figure:
    """Block diagrams of the classical baseline and of our unrolled ADMM-Net."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))

    _draw_flow(
        axes[0],
        ["fully sampled\nslice $x$", "FFT\n$\\mathcal{F}$", "variable-density\nmask $M$",
         "zero-filled\nIFFT", "multi-level wavelet\nsoft-threshold", "TV\nproximal step",
         "data consistency\n(POCS)", "reconstruction\n$\\hat{x}$"],
        "Baseline: classical compressed sensing (wavelet $\\ell_1$ + TV, POCS) -- no learned parameters",
        loop_label="$\\times$ num_iters", loop_span=(4, 6),
    )

    _draw_flow(
        axes[1],
        ["fully sampled\nslice $x$", "FFT\n$\\mathcal{F}$", "variable-density\nmask $M$",
         "zero-filled\nIFFT", "analysis conv\n(C-update)",
         "learnable\nsoft-threshold\n(Z-update)", "dual update\n(M-update)",
         "synthesis conv", "data consistency\nlearnable $\\rho$\n(X-update)",
         "reconstruction\n$\\hat{x}$"],
        "Our model: unrolled ADMM-Net -- the same prior + data-consistency structure, learned from data",
        loop_label="$\\times$ num_stages (unrolled, independent weights per stage)",
        loop_span=(4, 8),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Exploratory data analysis (requires the dataset)
# ---------------------------------------------------------------------------


def eda_panel(cfg: Dict, num_examples: int = 4) -> plt.Figure:
    """Example slices, the per-split age distribution, and the undersampling artifact."""
    from .dataset import build_datasets, build_splits

    splits, meta, data_cfg = build_splits(cfg)
    datasets = build_datasets(cfg)
    testset = datasets["test"]

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(3, max(num_examples, 4), hspace=0.42, wspace=0.25)

    # Row 1: example central slices (magnitude), the model's actual input domain.
    for i in range(num_examples):
        ax = fig.add_subplot(grid[0, i])
        img = _magnitude(testset[i].unsqueeze(0))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"test slice #{i}", fontsize=9)
        ax.axis("off")
    fig.text(0.02, 0.90, "Central axial slices (magnitude), 128x128, normalized by the "
                         "volume's maximum magnitude", fontsize=10)

    # Row 2 left: age distribution per split (the custom age-stratified split).
    ax = fig.add_subplot(grid[1, :2])
    by_path = meta.set_index("path")
    import pandas as pd

    for name in ("train", "val", "test"):
        ages = pd.to_numeric(by_path.loc[splits[name], data_cfg.age_col],
                             errors="coerce").dropna()
        ax.hist(ages, bins=30, density=True, histtype="step", linewidth=1.6, label=name)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("density")
    ax.set_title("Age distribution is matched across the age-stratified splits", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)

    # Row 2 right: intensity histogram of one slice (image contrast / background share).
    ax = fig.add_subplot(grid[1, 2:])
    mags = np.concatenate([_magnitude(testset[i].unsqueeze(0)).ravel()
                           for i in range(min(20, len(testset)))])
    ax.hist(mags, bins=100, color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("magnitude (normalized)")
    ax.set_ylabel("pixel count (log)")
    ax.set_title("Intensity distribution over 20 slices: a large low-signal background "
                 "and a soft-tissue bulk", fontsize=10)
    ax.grid(alpha=0.3)

    # Row 3: fully sampled vs undersampled k-space and the resulting artifact.
    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), **params)
    label = testset[0].unsqueeze(0)
    full_k = fft2c(chan_to_complex(label))
    under_k = mask * full_k
    zero_filled = torch.cat([ifft2c(under_k).real, ifft2c(under_k).imag], dim=1)

    def _logk(k):
        return to_display(np.log1p(np.abs(np.fft.fftshift(k[0, 0].numpy()))))

    panels = [
        (_logk(full_k), "fully sampled k-space (log)"),
        (_logk(under_k), f"{int(round(float(cfg['mask']['sampling_ratio']) * 100))}% "
                         f"undersampled k-space (log)"),
        (_magnitude(label), "fully sampled image"),
        (_magnitude(zero_filled), "zero-filled reconstruction: aliasing + blurring"),
    ]
    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(grid[2, i])
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Dataset and forward model: what the network sees", fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# 4. Reconstruction across the unrolled stages (requires a checkpoint)
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_stage_reconstruction(checkpoint_path: str, sample_index: int = 0,
                             device: Optional[str] = None) -> plt.Figure:
    """Show the estimate after each unrolled ADMM stage, with its PSNR.

    Makes the model-based structure visible: every stage alternates a learned prior with a
    data-consistency projection, so the aliasing is removed progressively rather than in
    one opaque forward pass.
    """
    from . import model as _model  # noqa: F401  (registers the ADMM-Net variants)
    from .dataset import build_datasets
    from .engine import build_model, undersample
    from .metrics import psnr_channel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if not hasattr(model, "stages"):
        raise ValueError("per_stage_reconstruction expects an unrolled model with .stages")

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    label = build_datasets(cfg)["test"][sample_index].unsqueeze(0).to(device)
    y = undersample(label, mask)

    x = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1)
    z = torch.zeros(1, model.channels, size, size, device=device, dtype=x.dtype)
    m = torch.zeros_like(z)

    def _psnr(est):
        est = est.clamp(-1, 1)
        return float((psnr_channel(est[:, 0], label[:, 0])
                      + psnr_channel(est[:, 1], label[:, 1])).item() / 2)

    stages = [("zero-filled input", _magnitude(x), _psnr(x))]
    for i, stage in enumerate(model.stages, 1):
        x, z, m = stage(x, z, m, y, mask)
        stages.append((f"after stage {i}", _magnitude(x), _psnr(x)))
    stages.append(("ground truth", _magnitude(label), float("nan")))

    cols = len(stages)
    fig, axes = plt.subplots(2, cols, figsize=(2.1 * cols, 5.0),
                             gridspec_kw={"height_ratios": [3, 2]})
    truth = stages[-1][1]
    for col, (title, img, psnr) in enumerate(stages):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=truth.max())
        axes[0, col].set_title(title + ("" if np.isnan(psnr) else f"\n{psnr:.1f} dB"),
                               fontsize=8.5)
        axes[0, col].axis("off")
        axes[1, col].imshow(np.abs(img - truth), cmap="inferno", vmin=0,
                            vmax=0.35 * truth.max())
        axes[1, col].set_title("|error|", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle("Reconstruction across the unrolled ADMM stages "
                 "(top: magnitude, bottom: absolute error)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Results figures derived from the logged CSVs (no dataset required)
# ---------------------------------------------------------------------------


def results_figures(cfg: Dict, baseline: Optional[str] = None,
                    model: str = "admmnet_softthresh") -> Dict[str, plt.Figure]:
    """The required metric plots, built from ``runs.csv`` / ``samples.csv`` alone.

    Returns a ``{filename_stem: figure}`` mapping. These need neither the dataset nor a
    checkpoint, so they can be regenerated anywhere the logs are available. ``baseline``
    defaults to the tuned classical baseline, falling back to the naive one when the tuned
    sweep has not been run yet.
    """
    from . import analysis

    results_root = cfg["paths"]["results_root"]
    df = analysis.load_results(results_root)
    samples = analysis.load_samples(results_root)

    logged = set(df["method"].unique())
    if baseline is None:
        for candidate in ("classical_cs_tv", "classical_cs"):
            if candidate in logged:
                baseline = candidate
                break
        else:
            baseline = "classical_cs_tv"
        if baseline != "classical_cs_tv":
            print(f"note: no 'classical_cs_tv' runs logged yet, plotting '{baseline}'. "
                  "Run configs/experiments/comparison_baseline_tv.yaml for the tuned "
                  "baseline.")
    methods = (baseline, model)

    figures: Dict[str, plt.Figure] = {}
    for base in ("psnr", "ssim"):
        figures[f"{base}_vs_ratio"] = analysis.plot_metric_vs_ratio(
            samples, base=base, methods=methods)
        figures[f"scatter_{base}"] = analysis.scatter_baseline_vs_model(
            samples, base=base, baseline=baseline, model=model)

    figures["depth_vs_psnr"] = analysis.plot_depth_vs_metric(df, model_name=model)
    figures["loss_ablation"] = analysis.plot_categorical_ablation(
        df, experiment="loss_ablation", label_cols=("loss",))
    figures["structure_ablation"] = analysis.plot_categorical_ablation(
        df, experiment="structure_ablation", label_cols=("method", "share_weights"))

    if "unet" in set(df["method"].unique()):
        figures["model_vs_unet"] = analysis.plot_model_vs_unet(df, model_name=model)
    if os.path.exists(os.path.join(results_root, "crossmask.csv")):
        figures["crossmask"] = analysis.plot_crossmask(analysis.load_crossmask(results_root))
    return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_FIGURES = ("masks", "pipelines", "results", "eda", "per_stage")


def _find_admmnet_checkpoint(results_root: str) -> Optional[str]:
    """Newest ADMM-Net ``best.pth`` under the results root, if any."""
    candidates = []
    for entry in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, entry, "best.pth")
        if os.path.exists(path):
            candidates.append(path)
    for path in reversed(candidates):
        try:
            cfg = load_checkpoint(path, map_location="cpu")["config"]
        except Exception:
            continue
        if str(cfg.get("model", {}).get("name", "")).startswith("admmnet"):
            return path
    return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the report's method/data figures.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                   help="Dotted config overrides.")
    p.add_argument("--which", nargs="*", default=list(ALL_FIGURES), choices=ALL_FIGURES,
                   help="Which figures to produce (default: all).")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint for --which per_stage (default: auto-discover).")
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    out_dir = _figure_dir(cfg)
    written: List[str] = []

    def _save(fig: plt.Figure, filename: str) -> None:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}")

    if "masks" in args.which:
        _save(mask_overview(cfg), "masks_overview.png")

    if "pipelines" in args.which:
        _save(pipeline_diagrams(), "pipelines.png")

    if "results" in args.which:
        try:
            for stem, fig in results_figures(cfg).items():
                _save(fig, f"{stem}.png")
        except FileNotFoundError as exc:
            print(f"skipping 'results': {exc}")

    if "eda" in args.which:
        try:
            _save(eda_panel(cfg), "eda_panel.png")
        except Exception as exc:  # the dataset is not reachable from every machine
            print(f"skipping 'eda': {type(exc).__name__}: {exc}")

    if "per_stage" in args.which:
        ckpt = args.checkpoint or _find_admmnet_checkpoint(cfg["paths"]["results_root"])
        if ckpt is None:
            print("skipping 'per_stage': no ADMM-Net checkpoint found under "
                  f"{cfg['paths']['results_root']}")
        else:
            try:
                _save(per_stage_reconstruction(ckpt), "per_stage_reconstruction.png")
            except Exception as exc:
                print(f"skipping 'per_stage': {type(exc).__name__}: {exc}")

    return written


if __name__ == "__main__":
    main()
