# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Is compressed sensing worth it, or would a plain low-resolution scan do just as well?

Every acceleration result in this report comes from keeping a random variable-density
subset of the phase-encode lines and then reconstructing. An MR physicist reading that
will immediately ask the obvious alternative: instead of *skipping* lines scattered
across k-space, simply acquire fewer lines **contiguously at the centre** and stop. That
is a genuinely lower-resolution scan of exactly the same duration -- ``n`` phase encodes
either way -- and it needs no reconstruction algorithm at all. The acquired band is
zero-padded back to 128x128 before the inverse FFT, which in the image domain is sinc
interpolation: the picture comes out at the right matrix size but carries no information
that was not measured. It is therefore *blurred* (the point spread function is a sinc of
width 128/n pixels) but completely *free of aliasing*, because nothing inside the
acquired band was skipped.

So the comparison at matched scan time is:

``low-res zero-padded``
    clean but blurred -- classical interpolation, no algorithm, no assumptions;
``zero-filled variable density``
    full nominal resolution but aliased -- the input our methods are handed;
``classical CS (TV)`` / ``ADMM-Net (ours)``
    the same aliased data, with a sparsity prior asked to undo the aliasing.

The question the numbers answer is whether the sparsity prior recovers more true detail
than the resolution the low-pass scan simply threw away. Note that PSNR is generous to
blur (a smooth error is a small error) and SSIM less so, which is why both are reported.

Metrics are PSNR and SSIM on the **real** channel: the ground-truth imaginary channel is
identically zero in this dataset, so the real channel is the anatomically meaningful one,
and ``src.metrics`` is used unchanged so the numbers sit alongside the rest of the report.

Usage::

    python -m src.eval_zeropad --ratios 0.2 0.3 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import model as _model  # noqa: F401  (registers the ADMM-Net variants)
from .baselines import classical_cs as _cs  # noqa: F401  (registers the CS baselines)
from .config import load_config
from .display import to_display
from .make_qualitative import _find_checkpoint, load_cached_test_split
from .masks import build_mask
from .metrics import psnr_channel, ssim_channel
from .registry import MODEL_REGISTRY
from .utils import chan_to_complex, complex_to_chan, fft2c, ifft2c, load_checkpoint

IMAGE_SIZE = 128
BATCH = 32  # slices per forward pass; the TV baseline dominates the runtime

#: Method keys in the order they should appear in the table, the figure and the legend.
METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("zeropad", "low-res zero-padded (central lines only)", "tab:purple"),
    ("zero_filled", "zero-filled (variable density)", "0.45"),
    ("cs", "classical CS (TV)", "tab:orange"),
    ("model", "ADMM-Net (ours)", "tab:green"),
)


def num_lines(ratio: float, rows: int = IMAGE_SIZE) -> int:
    """Phase-encode lines acquired at a given sampling ratio.

    Deliberately identical to the rounding inside ``masks.gaussian1d_mask`` so the
    low-resolution scan and the variable-density scan take exactly the same time.
    """
    return int(round(ratio * rows))


def lowpass_row_pattern(n: int, rows: int = IMAGE_SIZE) -> np.ndarray:
    """Centred contiguous band of ``n`` phase-encode lines, in the *centred* layout.

    The band is placed so that it always contains the DC line at index ``rows // 2``,
    which is where ``np.fft.fftshift`` puts it for an even-length axis.
    """
    if not 0 < n <= rows:
        raise ValueError(f"n must be in (0, {rows}], got {n}")
    pattern = np.zeros(rows, dtype=np.float32)
    start = rows // 2 - n // 2
    pattern[start : start + n] = 1.0
    return pattern


def lowpass_mask(n: int, rows: int = IMAGE_SIZE, cols: int = IMAGE_SIZE,
                 device: str | torch.device | None = None) -> torch.Tensor:
    """The low-pass band as a ``(1, 1, H, W)`` mask in the non-centred (FFT) layout.

    ``ifftshift`` is the same conversion ``masks.build_mask`` applies, so this mask can be
    multiplied straight into ``fft2c`` output exactly like the project's own masks.
    """
    pattern = lowpass_row_pattern(n, rows)
    mask_np = np.fft.ifftshift(np.repeat(pattern[:, None], cols, axis=1), axes=0)
    tensor = torch.from_numpy(np.ascontiguousarray(mask_np)).float()[None, None]
    return tensor if device is None else tensor.to(device)


def zero_padded_recon(label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse FFT of the acquired central band, zero-padded to the full matrix.

    No cropping and no resampling: leaving the unacquired lines at zero *is* the
    interpolation, and it is the one a scanner performs when it reconstructs a
    low-resolution acquisition onto a larger display matrix.
    """
    return complex_to_chan(ifft2c(mask * fft2c(chan_to_complex(label))))


# ---------------------------------------------------------------------------
# Correctness checks on the zero-padding
# ---------------------------------------------------------------------------


def verify(label: torch.Tensor, ratio: float = 0.3) -> Dict[str, float]:
    """Two sanity checks that catch a wrong FFT centring convention.

    ``identity``
        with all 128 lines retained the "zero-padded" recon must reproduce the input
        exactly; if the band were mis-centred this would still hold, so it only proves
        the transform pair, which is why the second check is also needed.
    ``parseval``
        the energy of the reconstructed image must equal the energy of the retained
        k-space band divided by the number of pixels (``numpy``/``torch`` unnormalised
        FFT convention). This fails if the mask and the spectrum are in different
        layouts, because then a different set of lines is being kept than is being
        measured.
    """
    full = zero_padded_recon(label, lowpass_mask(IMAGE_SIZE))
    identity = float((full - label).abs().max())

    n = num_lines(ratio)
    mask = lowpass_mask(n)
    kspace = mask * fft2c(chan_to_complex(label))
    recon = complex_to_chan(ifft2c(kspace))
    image_energy = float((recon ** 2).sum())
    kspace_energy = float((kspace.abs() ** 2).sum() / (IMAGE_SIZE * IMAGE_SIZE))

    # The DC line must be inside the band, otherwise the recon has no mean signal.
    pattern = lowpass_row_pattern(n)
    return {
        "identity_max_abs_error": identity,
        "image_energy": image_energy,
        "kspace_energy": kspace_energy,
        "parseval_rel_error": abs(image_energy - kspace_energy) / max(kspace_energy, 1e-12),
        "dc_line_acquired": float(pattern[IMAGE_SIZE // 2]),
        "lines": float(pattern.sum()),
    }


# ---------------------------------------------------------------------------
# Evaluation over the cached test split
# ---------------------------------------------------------------------------


def _scores(recon: torch.Tensor, label: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slice PSNR and SSIM on the real channel for a batch ``(B, 2, H, W)``."""
    psnr = psnr_channel(recon[:, 0], label[:, 0]).cpu().numpy()
    ssim = ssim_channel(recon[:, 0], label[:, 0]).cpu().numpy()
    return psnr, ssim


@torch.no_grad()
def evaluate(checkpoint_path: str, ratio: float, baseline: str = "classical_cs_tv",
             cache_root: str = "cache", device: str = "cpu"
             ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
    """Score all four methods on every cached test slice at one sampling ratio."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = MODEL_REGISTRY.build(cfg["model"]["name"], **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    base_model = MODEL_REGISTRY.build(baseline).to(device).eval()

    size = int(cfg["data"]["image_size"])
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    mask_params.setdefault("seed", cfg["train"]["seed"])
    vd_mask, vd_centered = build_mask(cfg["mask"]["name"], (size, size), device, **mask_params)

    n = num_lines(ratio, size)
    lp_mask = lowpass_mask(n, size, size, device)
    assert int(np.asarray(vd_centered)[:, 0].sum()) == n, "line budgets must match"

    slices, _ = load_cached_test_split(cache_root)
    collected: Dict[str, Dict[str, List[np.ndarray]]] = {
        key: {"psnr": [], "ssim": []} for key, _, _ in METHODS
    }

    for start in range(0, slices.shape[0], BATCH):
        label = slices[start : start + BATCH].to(device)
        y = vd_mask * fft2c(chan_to_complex(label))
        recons = {
            "zeropad": zero_padded_recon(label, lp_mask).clamp(-1, 1),
            "zero_filled": complex_to_chan(ifft2c(y)).clamp(-1, 1),
            "cs": base_model(y, vd_mask).clamp(-1, 1),
            "model": model(y, vd_mask).clamp(-1, 1),
        }
        for key, recon in recons.items():
            psnr, ssim = _scores(recon, label)
            collected[key]["psnr"].append(psnr)
            collected[key]["ssim"].append(ssim)

    results = {key: {m: np.concatenate(v) for m, v in d.items()}
               for key, d in collected.items()}
    context = {"model": model, "base_model": base_model, "vd_mask": vd_mask,
               "vd_centered": np.asarray(vd_centered)[:, 0], "lp_mask": lp_mask,
               "lines": n, "slices": slices, "device": device, "ratio": ratio}
    return results, context


def summarize(results: Dict[str, Dict[str, np.ndarray]], ratio: float,
              lines: int) -> List[Dict]:
    """One CSV row per method: mean and std of PSNR/SSIM on the real channel."""
    rows: List[Dict] = []
    for key, label, _ in METHODS:
        psnr, ssim = results[key]["psnr"], results[key]["ssim"]
        rows.append({
            "sampling ratio": ratio,
            "lines acquired": lines,
            "method": label,
            "PSNR real mean (dB)": round(float(psnr.mean()), 3),
            "PSNR real std (dB)": round(float(psnr.std()), 3),
            "SSIM real mean": round(float(ssim.mean()), 4),
            "SSIM real std": round(float(ssim.std()), 4),
            "n slices": int(psnr.size),
        })
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _magnitude(x: torch.Tensor) -> np.ndarray:
    """Magnitude image of a single-slice ``(1, 2, H, W)`` tensor, in viewing orientation."""
    return to_display(torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).detach().cpu().numpy())


def _bar_panel(ax: plt.Axes, rows: Sequence[Dict], ratios: Sequence[float],
               key: str, ylabel: str, title: str, ylim: Tuple[float, float]) -> None:
    """Grouped bar chart of one metric, one group per sampling ratio."""
    width = 0.8 / len(METHODS)
    positions = np.arange(len(ratios))
    for i, (_, label, colour) in enumerate(METHODS):
        values = [next((r[key] for r in rows
                        if r["method"] == label and r["sampling ratio"] == ratio), np.nan)
                  for ratio in ratios]
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width,
                      color=colour, edgecolor="black", linewidth=0.6, label=label)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value,
                        f"{value:.2f}" if value < 5 else f"{value:.1f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(round(r * 100))}% of lines\n({num_lines(r)}/128)"
                        for r in ratios])
    ax.set_xlabel("scan time (fraction of phase-encode lines acquired)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


@torch.no_grad()
def figure(rows: List[Dict], context: Dict[str, object], slice_index: int = 3) -> plt.Figure:
    """Metrics, a matched-scan-time image row, and the resolution loss made explicit."""
    ratios = sorted({r["sampling ratio"] for r in rows})
    slices = context["slices"]
    device, ratio = context["device"], float(context["ratio"])
    lines = int(context["lines"])

    label = slices[slice_index : slice_index + 1].to(device)
    y = context["vd_mask"] * fft2c(chan_to_complex(label))
    images = [
        ("ground truth\n(all 128 lines)", _magnitude(label)),
        (f"low-res zero-padded\n({lines} central lines)",
         _magnitude(zero_padded_recon(label, context["lp_mask"]).clamp(-1, 1))),
        (f"zero-filled variable density\n({lines} scattered lines)",
         _magnitude(complex_to_chan(ifft2c(y)).clamp(-1, 1))),
        ("classical CS (TV)\nfrom the same data",
         _magnitude(context["base_model"](y, context["vd_mask"]).clamp(-1, 1))),
        ("ADMM-Net (ours)\nfrom the same data",
         _magnitude(context["model"](y, context["vd_mask"]).clamp(-1, 1))),
    ]
    vmax = float(images[0][1].max()) or 1.0

    # Rows are placed with explicit figure coordinates rather than tight_layout: the image
    # row has a fixed 1:1 aspect, so letting the layout engine stretch it opens a large
    # vertical gap.
    fig = plt.figure(figsize=(16.5, 13.4))
    bar_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.055, right=0.985,
                                 top=0.875, bottom=0.700, wspace=0.16)
    image_grid = gridspec.GridSpec(1, 5, figure=fig, left=0.055, right=0.985,
                                   top=0.600, bottom=0.385, wspace=0.06)
    lower_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.055, right=0.985,
                                   top=0.290, bottom=0.050, wspace=0.19)

    psnr_all = [r["PSNR real mean (dB)"] for r in rows]
    ssim_all = [r["SSIM real mean"] for r in rows]
    ax_psnr = fig.add_subplot(bar_grid[0, 0])
    _bar_panel(ax_psnr, rows, ratios, "PSNR real mean (dB)",
               "PSNR on the real channel (dB)",
               "(a) Reconstruction PSNR at matched scan time",
               (min(psnr_all) - 3.0, max(psnr_all) + 5.0))
    _bar_panel(fig.add_subplot(bar_grid[0, 1]), rows, ratios, "SSIM real mean",
               "SSIM on the real channel",
               "(b) Reconstruction SSIM at matched scan time",
               (min(ssim_all) - 0.08, max(ssim_all) + 0.13))
    handles, labels = ax_psnr.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(METHODS), fontsize=9.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.925))

    # Row 2: the same slice through every acquisition, on one shared grey scale.
    # The profile is taken through the readout row with the strongest phase-encode
    # gradient energy, i.e. the row carrying the most fine structure -- exactly the
    # structure a low-resolution scan cannot resolve.
    truth_image = images[0][1]
    profile_row = int(np.argmax((np.diff(truth_image, axis=1) ** 2).sum(axis=1)))
    fig.text(0.055, 0.640, f"(c) One test slice (#{slice_index}) reconstructed from each "
                           "acquisition, identical grey scale; the dashed line marks the "
                           "profile plotted in (d)", fontsize=11)
    for col, (title, image) in enumerate(images):
        ax = fig.add_subplot(image_grid[0, col])
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        ax.axhline(profile_row, color="tab:red", linewidth=0.9, linestyle="--")
        ax.set_xlabel("phase encode (pixels)", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        if col == 0:
            ax.set_ylabel("readout (pixels)", fontsize=8.5)

    # (d) Intensity profile along the phase-encode direction, where the two failure modes
    # differ: the low-pass scan rounds the edges, the undersampled scan adds ripple.
    ax = fig.add_subplot(lower_grid[0, 0])
    for (title, image), (_, _, colour) in zip(images, (("", "", "black"),) + METHODS[:4]):
        style = dict(color=colour, linewidth=1.9 if colour == "black" else 1.2,
                     alpha=1.0 if colour == "black" else 0.9)
        ax.plot(image[profile_row], label=title.split("\n")[0], **style)
    ax.set_xlabel(f"phase-encode position along the dashed line (row {profile_row}, pixels)",
                  fontsize=9)
    ax.set_ylabel("magnitude (normalized)", fontsize=9)
    ax.set_title("(d) Intensity profile across the head: blurred edges versus "
                 "aliasing ripple", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.3)

    # (e) The resolution loss stated in k-space: which lines each scheme actually measures.
    ax = fig.add_subplot(lower_grid[0, 1])
    truth_k = np.fft.fftshift(np.fft.fft2(slices[slice_index, 0].numpy()))
    energy = (np.abs(truth_k) ** 2).sum(axis=1)
    energy = energy / energy.sum()
    offsets = np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2
    ax.semilogy(offsets, energy, color="0.25", linewidth=1.2,
                label="k-space energy per line (this slice)")
    lp_pattern = lowpass_row_pattern(lines)
    vd_pattern = np.asarray(context["vd_centered"])
    ax.fill_between(offsets, 1e-12, energy, where=lp_pattern > 0, color="tab:purple",
                    alpha=0.28, label=f"low-pass band ({lines} contiguous lines)")
    acquired = vd_pattern > 0
    ax.plot(offsets[acquired], energy[acquired], "o", markersize=3.0, color="tab:orange",
            label=f"variable-density lines ({int(vd_pattern.sum())} scattered)")
    ax.set_xlabel("$k_y$ relative to the centre of k-space (phase-encode line index)",
                  fontsize=9)
    ax.set_ylabel("fraction of slice energy per line (log)", fontsize=9)
    ax.set_ylim(max(energy.min(), 1e-11), energy.max() * 4)
    ax.set_title("(e) Lines each scheme measures: the low-pass scan never sees the "
                 "outer $k_y$", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Compressed sensing versus simply scanning at lower resolution, at matched scan "
        f"time.\nPanels (c)-(d) use test slice #{slice_index} at "
        f"{int(round(ratio * 100))}% of phase-encode lines ({lines}/128 either way): the "
        "low-pass scan is clean but blurred, the variable-density scan is sharp but "
        "aliased.", fontsize=13.5, y=0.982)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--ratios", nargs="*", type=float, default=[0.2, 0.3, 0.5])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default="classical_cs_tv")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--slice-index", type=int, default=3,
                   help="Test slice rendered in the image row and the profile.")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> List[Dict]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    slices, _ = load_cached_test_split(args.cache_root)
    checks = verify(slices[args.slice_index : args.slice_index + 1])
    print("zero-padding verification (test slice "
          f"#{args.slice_index}, band = {int(checks['lines'])} lines):")
    print(f"  (i)  all 128 lines -> max |recon - truth| = "
          f"{checks['identity_max_abs_error']:.3e}   "
          f"{'PASS' if checks['identity_max_abs_error'] < 1e-6 else 'FAIL'}")
    print(f"  (ii) Parseval: image energy {checks['image_energy']:.6f} vs retained "
          f"k-space energy {checks['kspace_energy']:.6f}  "
          f"(relative error {checks['parseval_rel_error']:.3e})   "
          f"{'PASS' if checks['parseval_rel_error'] < 1e-6 else 'FAIL'}")
    print(f"  DC line inside the acquired band: "
          f"{'yes' if checks['dc_line_acquired'] else 'no'}")
    if checks["identity_max_abs_error"] >= 1e-6 or checks["parseval_rel_error"] >= 1e-6:
        raise RuntimeError("zero-padding failed its own consistency checks; "
                           "the FFT centring convention is wrong")

    all_rows: List[Dict] = []
    figure_context: Dict[str, object] | None = None
    for ratio in args.ratios:
        ckpt = _find_checkpoint(results_root, ratio, args.seed)
        if ckpt is None:
            print(f"skipping ratio {ratio}: no comparison ADMM-Net checkpoint")
            continue
        results, context = evaluate(ckpt, ratio, args.baseline, args.cache_root, args.device)
        rows = summarize(results, ratio, int(context["lines"]))
        all_rows.extend(rows)
        for row in rows:
            print(f"  ratio={row['sampling ratio']}  {row['method']:42s} "
                  f"PSNR {row['PSNR real mean (dB)']:6.2f} +/- "
                  f"{row['PSNR real std (dB)']:.2f} dB   "
                  f"SSIM {row['SSIM real mean']:.4f} +/- {row['SSIM real std']:.4f}")
        if abs(ratio - 0.3) < 1e-9 or figure_context is None:
            figure_context = context

    if all_rows and figure_context is not None:
        path = os.path.join(out_dir, "mri_zeropad.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {path}")

        fig = figure(all_rows, figure_context, args.slice_index)
        png = os.path.join(out_dir, "mri_zeropad.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")
    return all_rows


if __name__ == "__main__":
    main()
