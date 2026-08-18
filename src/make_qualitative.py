# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Render the four required qualitative example categories (report section 5.6.2).

The course brief asks for four worked examples -- input / baseline / our model / ground
truth -- covering the cases where (a) both methods do well, (b) both do badly, (c) the
baseline wins, and (d) our model wins.

``src.analysis.qualitative_examples`` already builds this panel, but it reaches the test
slices through ``src.dataset.build_datasets``, which cross-references the metadata CSVs
and therefore needs the raw dataset directory. This entry point reads the **pre-extracted
slice cache** instead (``cache/<fingerprint>/test.npz``, byte-identical to what training
consumed), so the figure can be regenerated on any machine that has the cache and a
checkpoint but not the multi-gigabyte volume directory.

It also adds two things the report needs and the generic helper does not provide:

* an error-map row under every reconstruction, so the *spatial* structure of the residual
  (coherent phase-encode aliasing vs. incoherent high-frequency loss) is visible;
* a k-space column, so each example can be tied back to which phase-encode lines the
  mask actually acquired.

Usage::

    python -m src.make_qualitative --checkpoint results/comparison_d347ec7a7f/best.pth
    python -m src.make_qualitative --ratios 0.2 0.3 0.5      # one panel per ratio
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import model as _model  # noqa: F401  (registers the ADMM-Net variants)
from .baselines import classical_cs as _cs  # noqa: F401  (registers the CS baselines)
from .config import load_config
from .display import to_display
from .masks import build_mask
from .metrics import psnr_channel, ssim_channel
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, fft2c, ifft2c, load_checkpoint

CATEGORY_ORDER = ("both good", "both poor", "closest the baseline comes", "our model wins")


# ---------------------------------------------------------------------------
# Cached test slices
# ---------------------------------------------------------------------------


def load_cached_test_split(cache_root: str = "cache") -> Tuple[torch.Tensor, List[str]]:
    """Load the cached ``test`` slices as a ``(N, 2, H, W)`` tensor plus subject names.

    Picks the newest cache fingerprint when several are present.
    """
    candidates = sorted(glob.glob(os.path.join(cache_root, "*", "test.npz")),
                        key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"no cached test split under {cache_root}/*/test.npz; run "
            "`python -m src.build_cache` on a machine that has the dataset")
    with np.load(candidates[-1], allow_pickle=True) as npz:
        data = npz["data"]
        subjects = [str(s) for s in npz["subjects"].tolist()]
    return torch.from_numpy(np.ascontiguousarray(data)).float(), subjects


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _magnitude(x: torch.Tensor) -> np.ndarray:
    """Magnitude image in viewing orientation, from a 2-channel tensor ``(1, 2, H, W)``."""
    mag = torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).detach().cpu().numpy()
    return to_display(mag)


@torch.no_grad()
def _per_slice_scores(recon: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
    """PSNR/SSIM on the real and imaginary channels for a single slice."""
    return {
        "psnr_real": float(psnr_channel(recon[:, 0], label[:, 0]).item()),
        "psnr_imag": float(psnr_channel(recon[:, 1], label[:, 1]).item()),
        "ssim_real": float(ssim_channel(recon[:, 0], label[:, 0]).item()),
        "ssim_imag": float(ssim_channel(recon[:, 1], label[:, 1]).item()),
    }


@torch.no_grad()
def evaluate_all(checkpoint_path: str, baseline: str = "classical_cs_tv",
                 cache_root: str = "cache", device: str = "cpu"):
    """Run our model and the baseline over every cached test slice.

    Returns ``(records, context)`` where ``records`` is one dict per slice and ``context``
    carries the tensors/objects needed to re-render a chosen slice.
    """
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    base_model = MODEL_REGISTRY.build(baseline).to(device).eval()

    size = cfg["data"]["image_size"]
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    mask_params.setdefault("seed", cfg["train"]["seed"])
    mask, centered = build_mask(cfg["mask"]["name"], (size, size), device, **mask_params)

    slices, subjects = load_cached_test_split(cache_root)

    records = []
    for i in range(slices.shape[0]):
        label = slices[i : i + 1].to(device)
        y = mask * fft2c(chan_to_complex(label))
        zf = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1)
        rec_m = model(y, mask).clamp(-1, 1)
        rec_b = base_model(y, mask).clamp(-1, 1)
        sm = _per_slice_scores(rec_m, label)
        sb = _per_slice_scores(rec_b, label)
        sz = _per_slice_scores(zf, label)
        records.append({
            "index": i,
            "subject": subjects[i] if i < len(subjects) else "",
            "zf_psnr": sz["psnr_real"], "zf_ssim": sz["ssim_real"],
            "model_psnr": sm["psnr_real"], "model_ssim": sm["ssim_real"],
            "base_psnr": sb["psnr_real"], "base_ssim": sb["ssim_real"],
            "model_psnr_imag": sm["psnr_imag"], "base_psnr_imag": sb["psnr_imag"],
        })
    context = {"cfg": cfg, "model": model, "base_model": base_model, "mask": mask,
               "centered": centered, "slices": slices, "device": device}
    return records, context


def pick_categories(records: List[Dict]) -> Dict[str, Dict]:
    """Choose one representative slice for each of the four required categories.

    Selection uses PSNR on the **real** channel, which is the anatomically meaningful one
    for this dataset (the ground-truth imaginary channel is identically zero -- see the
    report's discussion of Hermitian symmetry).
    """
    by_model = sorted(records, key=lambda r: r["model_psnr"])
    by_sum = sorted(records, key=lambda r: r["model_psnr"] + r["base_psnr"])
    by_gap = sorted(records, key=lambda r: r["model_psnr"] - r["base_psnr"])
    return {
        "both good": by_model[-1],
        "both poor": by_sum[0],
        # The brief asks for a slice where the baseline beats our model. There is none, so
        # we show the slice where our margin is *smallest* and say so explicitly.
        "closest the baseline comes": by_gap[0],
        "our model wins": by_gap[-1],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


@torch.no_grad()
def qualitative_panel(records: List[Dict], context: Dict, baseline_label: str) -> plt.Figure:
    """4 (category) x 6 (view) panel: k-space, input, baseline, model, truth, error maps."""
    picks = pick_categories(records)
    model, base_model = context["model"], context["base_model"]
    mask, slices, device = context["mask"], context["slices"], context["device"]
    ratio = float(context["cfg"]["mask"]["sampling_ratio"])

    n_rows, n_cols = len(CATEGORY_ORDER), 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.95 * n_rows))

    col_titles = [
        "acquired k-space\n(log magnitude)",
        "zero-filled input\n(what the model sees)",
        f"baseline\n{baseline_label}",
        "our model\n(unrolled ADMM-Net)",
        "ground truth\n(fully sampled)",
        "|error|\nbaseline",
        "|error|\nours",
    ]

    for row, category in enumerate(CATEGORY_ORDER):
        rec = picks[category]
        idx = rec["index"]
        label = slices[idx : idx + 1].to(device)
        y = mask * fft2c(chan_to_complex(label))
        zf = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1).clamp(-1, 1)
        rec_b = base_model(y, mask).clamp(-1, 1)
        rec_m = model(y, mask).clamp(-1, 1)

        truth = _magnitude(label)
        vmax = float(truth.max()) or 1.0

        # Column 0: the acquired k-space, so the reader can see which lines were kept.
        kshow = np.log1p(np.abs(np.fft.fftshift(y[0, 0].detach().cpu().numpy())))
        axes[row, 0].imshow(to_display(kshow), cmap="gray", aspect="equal")
        axes[row, 0].set_xlabel("$k_y$ (phase encode)", fontsize=8)
        axes[row, 0].set_ylabel("$k_x$ (readout)", fontsize=8)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        # Columns 1-4: magnitude images on one shared grey scale.
        for col, img in enumerate([_magnitude(zf), _magnitude(rec_b),
                                   _magnitude(rec_m), truth], start=1):
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=vmax)
            axes[row, col].axis("off")

        # Columns 5-6: the two error maps, on one shared scale so they are comparable.
        for col, recon in ((5, rec_b), (6, rec_m)):
            axes[row, col].imshow(np.abs(_magnitude(recon) - truth), cmap="inferno",
                                  vmin=0, vmax=0.35 * vmax)
            axes[row, col].axis("off")

        # Per-panel captions: the scores that define the category.
        captions = [
            "",
            f"PSNR {rec['zf_psnr']:.1f} dB",
            f"PSNR {rec['base_psnr']:.1f} dB\nSSIM {rec['base_ssim']:.3f}",
            f"PSNR {rec['model_psnr']:.1f} dB\nSSIM {rec['model_ssim']:.3f}",
            "reference",
            "", "",
        ]
        for col, caption in enumerate(captions):
            prefix = col_titles[col] + "\n" if row == 0 else ""
            axes[row, col].set_title(prefix + caption, fontsize=8.5)

        # Row label on the far left.
        axes[row, 0].text(-0.34, 0.5, f"{category}\n(slice #{idx})",
                          transform=axes[row, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=10, fontweight="bold")

    margin = picks["closest the baseline comes"]
    fig.suptitle(
        f"Qualitative reconstructions at {int(round(ratio * 100))}% of phase-encode lines "
        "(magnitude images; PSNR/SSIM on the real channel).\n"
        "No test slice exists on which the baseline beats our model; row 3 is the slice "
        f"where our margin is smallest (+{margin['model_psnr'] - margin['base_psnr']:.1f} dB).",
        fontsize=12, y=0.997)
    fig.tight_layout(rect=(0.018, 0, 1, 0.975))
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_checkpoint(results_root: str, ratio: Optional[float], seed: int = 0) -> Optional[str]:
    """Newest ``comparison`` ADMM-Net checkpoint at a given sampling ratio and seed."""
    for path in sorted(glob.glob(os.path.join(results_root, "*", "manifest.json"))):
        cfg = json.load(open(path))["config"]
        ckpt = os.path.join(os.path.dirname(path), "best.pth")
        if not os.path.exists(ckpt):
            continue
        if not str(cfg.get("model", {}).get("name", "")).startswith("admmnet"):
            continue
        if cfg.get("name") != "comparison":
            continue
        if seed is not None and int(cfg["train"]["seed"]) != int(seed):
            continue
        if ratio is not None and abs(float(cfg["mask"]["sampling_ratio"]) - ratio) > 1e-9:
            continue
        return ckpt
    return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint; otherwise discovered from --ratios/--seed.")
    p.add_argument("--ratios", nargs="*", type=float, default=[0.3],
                   help="Sampling ratios to render one panel each for.")
    p.add_argument("--seed", type=int, default=0, help="Training seed of the checkpoint.")
    p.add_argument("--baseline", default="classical_cs_tv")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--device", default="cpu")
    p.add_argument("--also", nargs="*", default=["per_stage", "eda"],
                   choices=["per_stage", "eda"],
                   help="Extra cache-based panels to render alongside the examples.")
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    out_dir = os.path.join(cfg["paths"]["results_root"], "figures")
    os.makedirs(out_dir, exist_ok=True)

    jobs = ([(None, args.checkpoint)] if args.checkpoint
            else [(r, _find_checkpoint(cfg["paths"]["results_root"], r, args.seed))
                  for r in args.ratios])

    written: List[str] = []
    for ratio, ckpt in jobs:
        if ckpt is None:
            print(f"skipping ratio {ratio}: no comparison ADMM-Net checkpoint found")
            continue
        records, context = evaluate_all(ckpt, args.baseline, args.cache_root, args.device)
        actual = float(context["cfg"]["mask"]["sampling_ratio"])
        fig = qualitative_panel(records, context, args.baseline)
        stem = ("qualitative_examples" if len(jobs) == 1
                else f"qualitative_examples_r{int(round(actual * 100))}")
        path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}  (from {ckpt})")

        picks = pick_categories(records)
        for category in CATEGORY_ORDER:
            r = picks[category]
            print(f"   {category:15s} slice #{r['index']:3d}  "
                  f"baseline {r['base_psnr']:6.2f} dB / {r['base_ssim']:.3f}   "
                  f"ours {r['model_psnr']:6.2f} dB / {r['model_ssim']:.3f}")

    if "per_stage" in args.also:
        ckpt = _find_checkpoint(cfg["paths"]["results_root"], 0.3, args.seed)
        if ckpt:
            fig = per_stage_panel(ckpt, cache_root=args.cache_root, device=args.device)
            path = os.path.join(out_dir, "per_stage_reconstruction.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")

    if "eda" in args.also:
        fig = eda_panel(args.cache_root, cfg["paths"]["results_root"])
        path = os.path.join(out_dir, "eda_panel.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}")
    return written




# ---------------------------------------------------------------------------
# The reconstruction as it forms across the unrolled stages
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_stage_panel(checkpoint_path: str, sample_index: int = 3,
                    cache_root: str = "cache", device: str = "cpu") -> plt.Figure:
    """The estimate after each unrolled ADMM stage, with its PSNR and error map.

    ``src/figures.py`` has an equivalent, but it reaches the slice through the dataset
    loader and so needs the raw volumes. This one reads the cache, which keeps the whole
    report reproducible from a laptop.
    """
    from .metrics import psnr_channel

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    if not hasattr(model, "stages"):
        raise ValueError("per_stage_panel expects an unrolled model with .stages")

    size = cfg["data"]["image_size"]
    params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **params)

    slices, _ = load_cached_test_split(cache_root)
    label = slices[sample_index : sample_index + 1].to(device)
    y = mask * fft2c(chan_to_complex(label))

    x = torch.cat([ifft2c(y).real, ifft2c(y).imag], dim=1)
    z = torch.zeros(1, model.channels, size, size, device=device, dtype=x.dtype)
    m = torch.zeros_like(z)

    def _psnr(estimate: torch.Tensor) -> float:
        e = estimate.clamp(-1, 1)
        return float(psnr_channel(e[:, 0], label[:, 0]).item())

    panels = [("zero-filled input", _magnitude(x), _psnr(x))]
    corrections: List[float] = []          # max |correction| each stage actually applied
    for i, stage in enumerate(model.stages, 1):
        # Recompute the stage in the open so the correction itself can be measured. This
        # is the same arithmetic as CustomADMMStage.forward, in the same order.
        c = stage.analysis(x)
        z_new = stage.nonlinearity(c + m)
        m_new = m + c - z_new
        correction = stage.synthesis(z_new - m_new)
        x = stage.data_consistency(x + correction, y, mask)
        z, m = z_new, m_new
        corrections.append(float(correction.abs().max().item()))
        panels.append((f"after stage {i}", _magnitude(x), _psnr(x)))
    panels.append(("ground truth", _magnitude(label), float("nan")))

    truth = panels[-1][1]
    vmax = float(truth.max()) or 1.0
    cols = len(panels)
    fig = plt.figure(figsize=(1.95 * cols, 7.4))
    grid = fig.add_gridspec(3, cols, height_ratios=[2.7, 1.9, 2.0], hspace=0.30,
                            wspace=0.06)

    dead = [i + 1 for i, value in enumerate(corrections) if value == 0.0]

    for col, (title, image, psnr) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title + ("" if np.isnan(psnr) else f"\n{psnr:.1f} dB"), fontsize=8.5,
                     color=("tab:red" if (col in dead) else "black"))
        ax.axis("off")
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(np.abs(image - truth), cmap="inferno", vmin=0, vmax=0.35 * vmax)
        ax.set_title("|error|" if col < cols - 1 else "", fontsize=8)
        ax.axis("off")

    # Bottom row: the two curves that explain the flat stretch above.
    ax = fig.add_subplot(grid[2, :])
    stages = np.arange(1, len(corrections) + 1)
    psnrs = [p for _, _, p in panels[1:-1]]
    ax.plot(np.arange(0, len(psnrs) + 1),
            [panels[0][2]] + psnrs, marker="o", color="tab:blue", linewidth=1.8,
            label="PSNR after the stage (left axis)")
    ax.set_xlabel("ADMM stage (0 = the zero-filled input the network receives)")
    ax.set_ylabel("PSNR, real channel (dB)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(np.arange(0, len(psnrs) + 1))
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    floor = 1e-4
    twin.bar(stages, [max(v, floor) for v in corrections], width=0.45, alpha=0.45,
             color=["tab:red" if v == 0.0 else "tab:green" for v in corrections],
             edgecolor="black", linewidth=0.5,
             label="max |correction| the stage applied (right axis)")
    twin.set_yscale("log")
    twin.set_ylim(floor, 2.0)
    twin.set_ylabel("max |correction| (log)")
    for stage_no, value in zip(stages, corrections):
        if value == 0.0:
            twin.text(stage_no, floor * 1.4, "0", ha="center", fontsize=7.5,
                      color="tab:red", fontweight="bold")

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    note = (f"stages {', '.join(map(str, dead))} apply a correction of exactly zero and are "
            "identity maps on this slice" if dead
            else "every stage applies a non-zero correction on this slice")
    ax.set_title(note, fontsize=9.5)

    fig.suptitle(f"Reconstruction across the {len(model.stages)} unrolled ADMM stages at "
                 f"{int(round(float(cfg['mask']['sampling_ratio']) * 100))}% sampling. "
                 "Red titles mark stages whose correction is exactly zero.", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


@torch.no_grad()
def eda_panel(cache_root: str = "cache", results_root: str = "results",
              ratio: float = 0.3, seed: int = 0) -> plt.Figure:
    """Example slices, the intensity distribution, and the forward model made visible."""
    from .display import load_split_members, to_display

    slices, subjects = load_cached_test_split(cache_root)
    real = slices[:, 0].numpy()

    fig = plt.figure(figsize=(15.5, 8.6))
    grid = fig.add_gridspec(3, 4, hspace=0.36, wspace=0.2)

    # Row 1: four example slices, which is what the methods actually receive.
    for i, index in enumerate((3, 14, 151, 273)):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(to_display(real[index]), cmap="gray")
        ax.set_title(f"test slice #{index}", fontsize=9)
        ax.axis("off")
    fig.text(0.012, 0.945, "Central axial slices (128×128), each normalized by its own "
                           "volume's maximum magnitude", fontsize=9.5)

    # Row 2 left: subject counts per split; right: the intensity distribution.
    ax = fig.add_subplot(grid[1, :2])
    members = load_split_members(results_root)
    names = ["train", "val", "test"]
    counts = [len(members.get(n, [])) for n in names]
    bars = ax.bar(names, counts, color=["tab:blue", "tab:orange", "tab:green"],
                  edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 60, f"{count}",
                ha="center", fontsize=9)
    ax.set_ylabel("subjects (= slices)")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_title("Age-stratified 80 / 10 / 10 split at the subject level", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(grid[1, 2:])
    pooled = real[:40].ravel()
    ax.hist(pooled, bins=140, color="0.45")
    ax.set_yscale("log")
    ax.set_xlabel("normalized intensity")
    ax.set_ylabel("voxel count (log)")
    background = float((pooled < 0.02).mean())
    ax.set_title(f"Intensity distribution over 40 slices: {background:.0%} of voxels are "
                 "background,\nthe rest a broad soft-tissue bulk", fontsize=10)
    ax.grid(alpha=0.3)

    # Row 3: the forward model, from full k-space to the aliased input.
    label = slices[3:4]
    mask, _ = build_mask("gaussian1d", (128, 128), None, sampling_ratio=ratio,
                         std_scale=0.25, center_lines=0, seed=seed)
    full_k = fft2c(chan_to_complex(label))
    under_k = mask * full_k
    zero_filled = torch.cat([ifft2c(under_k).real, ifft2c(under_k).imag], dim=1)

    def _logk(k):
        return to_display(np.log1p(np.abs(np.fft.fftshift(k[0, 0].numpy()))))

    panels = [
        (_logk(full_k), "fully sampled k-space (log)"),
        (_logk(under_k), f"{int(round(ratio * 100))}% undersampled k-space (log)"),
        (_magnitude(label), "fully sampled image"),
        (_magnitude(zero_filled), "zero-filled: aliasing + blurring"),
    ]
    for i, (image, title) in enumerate(panels):
        ax = fig.add_subplot(grid[2, i])
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Dataset and forward model: what the two methods are given", fontsize=13)
    return fig


if __name__ == "__main__":
    main()
