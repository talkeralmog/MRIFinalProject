# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""MRI-principle figures: the physics behind our undersampling and metric choices.

``src/figures.py`` documents the *pipeline* (masks, block diagrams, EDA) and
``src/analysis.py`` plots the *results*. This module adds the figures that tie the
project back to the acquisition physics taught in the course, each measured on our own
data rather than quoted from a textbook:

``energy``
    Where the signal energy actually lives along the phase-encode axis of k-space, for
    the 478 test slices. Quantifies the lecture's "central k-space lines have large
    signal amplitude; outer lines have smaller amplitude but carry fine spatial detail"
    and so justifies the variable-density mask.
``psf``
    The point spread function of three undersampling schemes at the *same* sampling
    ratio -- low-pass truncation, regular (equispaced) skipping, and our random
    variable-density draw -- and the artifact each produces. Because there is no
    anti-aliasing filter in the phase-encode direction, skipping k_y lines wraps the
    anatomy around the FOV; whether that wrap is coherent (ghosts) or incoherent
    (noise-like) is what decides whether a sparsity prior can undo it.
``hermitian``
    Why a dataset of real-valued (magnitude) images still produces a non-zero
    "imaginary channel" after undersampling: a real image has conjugate-symmetric
    k-space, and a random mask keeps only about half of the conjugate pairs, so the
    zero-filled reconstruction picks up a spurious imaginary part. This is the same
    symmetry that partial-Fourier acquisition exploits.
``tradeoff``
    The acquisition trade-off made quantitative for our experiment: nominal scan-time
    reduction and the sqrt(N) SNR penalty of acquiring fewer lines, against the measured
    reconstruction quality.
``contrast``
    The contrast weighting of the dataset, read off the images and the intensity
    histogram (bright white matter, mid-grey cortex, dark CSF => T1-weighted), with the
    course's relaxation-time table as the reference.

None of these need the raw dataset -- they read the pre-extracted slice cache -- so they
can be regenerated anywhere.

Usage::

    python -m src.figures_mri --config configs/default.yaml
    python -m src.figures_mri --which energy psf
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

from .config import load_config
from .display import to_display
from .make_qualitative import load_cached_test_split
from .masks import build_mask

RATIOS = (0.2, 0.3, 0.5)
N_ROWS = 128

# Proton density and relaxation times of brain tissue at 1.5 T, as given in the course
# (class 3, "Contrast Mechanism"; source: Medical Imaging Signals & Systems).
TISSUE_TABLE = [
    ("White matter", 0.61, 67, 510),
    ("Grey matter", 0.69, 77, 760),
    ("CSF", 1.00, 280, 2650),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _test_slices(cache_root: str) -> np.ndarray:
    """Real channel of the cached test slices, as ``(N, H, W)``."""
    data, _ = load_cached_test_split(cache_root)
    return data[:, 0].numpy()


def _ky_energy_profile(images: np.ndarray) -> np.ndarray:
    """Total k-space energy per phase-encode row (centred), summed over the test set."""
    spectrum = np.fft.fftshift(np.fft.fft2(images, axes=(1, 2)), axes=(1, 2))
    return (np.abs(spectrum) ** 2).sum(axis=(0, 2))


def _row_mask(kind: str, ratio: float, rows: int = N_ROWS, seed: int = 0) -> np.ndarray:
    """A 1D phase-encode sampling pattern of length ``rows`` keeping ``ratio`` of lines.

    ``variable_density`` is the project's actual mask; the other two are references used
    only to make the artifact comparison, and all three keep the same number of lines.
    """
    keep = int(round(ratio * rows))
    centre = rows // 2
    if kind == "lowpass":
        m = np.zeros(rows)
        m[centre - keep // 2 : centre - keep // 2 + keep] = 1.0
        return m
    if kind == "regular":
        m = np.zeros(rows)
        step = max(int(round(rows / keep)), 1)
        m[centre % step :: step] = 1.0
        return m
    if kind == "uniform":
        # Uniform random: every line equally likely, ignoring where the energy is.
        rng = np.random.default_rng(seed)
        m = np.zeros(rows)
        m[rng.choice(rows, size=keep, replace=False)] = 1.0
        return m
    if kind == "variable_density":
        _, centered = build_mask("gaussian1d", (rows, rows), sampling_ratio=ratio,
                                 std_scale=0.25, center_lines=0, seed=seed)
        return np.asarray(centered)[:, 0]
    raise ValueError(f"unknown mask kind: {kind}")


def _forced_centre_mask(ratio: float, center_lines: int, seed: int,
                        rows: int = N_ROWS) -> np.ndarray:
    """Our variable-density pattern with a small central band force-sampled."""
    _, centered = build_mask("gaussian1d", (rows, rows), sampling_ratio=ratio,
                             std_scale=0.25, center_lines=center_lines, seed=seed)
    return np.asarray(centered)[:, 0]


def _zero_fill(image: np.ndarray, row_pattern: np.ndarray) -> np.ndarray:
    """Zero-filled reconstruction of one slice under a centred row-sampling pattern."""
    k = np.fft.fftshift(np.fft.fft2(image))
    k = k * row_pattern[:, None]
    return np.fft.ifft2(np.fft.ifftshift(k))


def _figure_dir(cfg: Dict) -> str:
    path = os.path.join(cfg["paths"]["results_root"], "figures")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Where the energy lives along k_y
# ---------------------------------------------------------------------------


def energy_figure(cache_root: str = "cache") -> Tuple[plt.Figure, List[Dict]]:
    """Measured k-space energy along the phase-encode axis, vs the mask's sampling rate."""
    images = _test_slices(cache_root)
    profile = _ky_energy_profile(images)
    centre = N_ROWS // 2
    total = profile.sum()
    offsets = np.arange(N_ROWS) - centre

    # Cumulative energy inside a central band of k_y, as a function of the band width.
    widths = np.arange(1, centre + 1)
    cumulative = np.array([profile[centre - w : centre + w + 1].sum() / total for w in widths])
    band_fraction = (2 * widths + 1) / N_ROWS

    rows: List[Dict] = []
    for ratio in RATIOS:
        w = int(round(ratio * N_ROWS / 2))
        rows.append({
            "central band (fraction of k_y rows)": round(float((2 * w + 1) / N_ROWS), 3),
            "rows": int(2 * w + 1),
            "energy fraction": round(float(profile[centre - w : centre + w + 1].sum() / total), 5),
        })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) The energy profile itself, on a log scale: five orders of magnitude of decay.
    axes[0].semilogy(offsets, profile / total, color="0.25", linewidth=1.2)
    axes[0].set_xlabel("$k_y$ relative to the centre of k-space (phase-encode line index)")
    axes[0].set_ylabel("fraction of total energy per line (log)")
    axes[0].set_title("(a) Signal energy per phase-encode line\n"
                      "(measured over all 478 test slices)", fontsize=10)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].axvline(0, color="crimson", linestyle=":", linewidth=1)
    axes[0].annotate(f"DC line alone:\n{profile[centre] / total * 100:.0f}% of all energy",
                     xy=(0, profile[centre] / total), xytext=(14, 0.06),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
                     fontsize=8.5, color="crimson")

    # (b) Cumulative energy vs band width, with our three sampling ratios marked.
    axes[1].plot(band_fraction, cumulative * 100, color="0.2", linewidth=2)
    offsets_xy = [(0.05, -9.0), (0.05, -5.0), (0.05, -1.0)]
    for (ratio, colour, (dx, dy)) in zip(RATIOS, ("tab:blue", "tab:orange", "tab:green"),
                                         offsets_xy):
        w = int(round(ratio * N_ROWS / 2))
        value = profile[centre - w : centre + w + 1].sum() / total * 100
        axes[1].plot([ratio], [value], "o", color=colour, markersize=8)
        axes[1].annotate(f"{int(ratio * 100)}% of rows $\\to$ {value:.1f}% of energy",
                         xy=(ratio, value), xytext=(ratio + dx, value + dy),
                         arrowprops=dict(arrowstyle="-", color=colour, lw=0.8),
                         fontsize=8.5, color=colour)
    axes[1].set_xlabel("width of the central $k_y$ band (fraction of all rows)")
    axes[1].set_ylabel("cumulative % of total k-space energy")
    axes[1].set_ylim(60, 103)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_title("(b) Most of the energy is concentrated in a\n"
                      "narrow band of central lines", fontsize=10)
    axes[1].grid(alpha=0.3)

    # (c) Energy profile against the probability our mask acquires each line.
    ax = axes[2]
    ax.semilogy(offsets, profile / total, color="0.25", linewidth=1.2,
                label="energy per line (measured)")
    ax.set_xlabel("$k_y$ relative to the centre of k-space")
    ax.set_ylabel("fraction of total energy per line (log)")
    twin = ax.twinx()
    for ratio, colour in zip(RATIOS, ("tab:blue", "tab:orange", "tab:green")):
        probability = np.mean([_row_mask("variable_density", ratio, seed=s)
                               for s in range(200)], axis=0)
        twin.plot(offsets, probability, color=colour, linewidth=1.6,
                  label=f"P(acquired), {int(ratio * 100)}%")
    twin.set_ylabel("P(line acquired) over 200 mask draws")
    twin.set_ylim(0, 1.32)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_title("(c) The variable-density mask spends its\n"
                 "line budget where the energy is", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Why a variable-density phase-encode mask: the energy distribution of "
                 "our own data", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, rows


# ---------------------------------------------------------------------------
# 2. Point spread function and the shape of the artifact
# ---------------------------------------------------------------------------


def psf_figure(cache_root: str = "cache", ratio: float = 0.3,
               slice_index: int = 3) -> plt.Figure:
    """Three ways to keep the same number of lines, and the artifact each one causes."""
    images = _test_slices(cache_root)
    truth = images[slice_index]

    schemes = [
        ("lowpass", "low-pass truncation\n(central lines only)"),
        ("regular", "regular skipping\n(every $R$-th line)"),
        ("uniform", "uniform random\n(every line equally likely)"),
        ("variable_density", "random variable-density\n(ours)"),
    ]

    fig, axes = plt.subplots(4, len(schemes), figsize=(3.9 * len(schemes), 14.4),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 2.4, 2.4]})

    for col, (kind, label) in enumerate(schemes):
        pattern = _row_mask(kind, ratio)
        recon = _zero_fill(truth, pattern)
        magnitude = np.abs(recon)
        error = np.abs(magnitude - truth)

        # Row 0: the sampling pattern along k_y.
        axes[0, col].stem(np.arange(N_ROWS) - N_ROWS // 2, pattern,
                          markerfmt=" ", basefmt=" ", linefmt="C0-")
        axes[0, col].set_ylim(0, 1.2)
        axes[0, col].set_yticks([])
        axes[0, col].set_xlabel("$k_y$", fontsize=9)
        axes[0, col].set_title(f"{label}\nkeeps {int(pattern.sum())}/{N_ROWS} lines "
                               f"({pattern.mean() * 100:.0f}%)", fontsize=9.5)

        # Row 1: the resulting point spread function along the phase-encode axis.
        psf = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pattern))))
        psf = psf / psf.max()
        axes[1, col].plot(np.arange(N_ROWS) - N_ROWS // 2, psf, color="0.2", linewidth=1.2)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(1e-4, 1.5)
        axes[1, col].set_xlabel("phase-encode direction (pixels)", fontsize=9)
        axes[1, col].set_ylabel("|PSF| (log)", fontsize=9)
        side = psf.copy()
        side[N_ROWS // 2] = 0
        axes[1, col].set_title(f"point spread function\npeak sidelobe = "
                               f"{side.max() * 100:.1f}% of the main lobe", fontsize=9)
        axes[1, col].grid(alpha=0.3, which="both")

        # Row 2: the zero-filled magnitude image.
        axes[2, col].imshow(to_display(magnitude), cmap="gray", vmin=0, vmax=truth.max())
        axes[2, col].axis("off")
        axes[2, col].set_title("zero-filled reconstruction", fontsize=9.5)

        # Row 3: the error map, where the artifact's structure is legible.
        axes[3, col].imshow(to_display(error), cmap="inferno", vmin=0, vmax=0.35 * truth.max())
        axes[3, col].axis("off")
        rmse = float(np.sqrt((error ** 2).mean()))
        axes[3, col].set_title(f"|error| (RMSE {rmse:.4f})", fontsize=9.5)

    fig.suptitle(
        f"Same line budget ({int(ratio * 100)}% of phase-encode lines), four different "
        "artifacts.\nThere is no anti-aliasing filter in the phase-encode direction, so "
        "skipping $k_y$ lines wraps signal around the FOV; only an incoherent\n(noise-like) "
        "wrap can be removed by a sparsity prior -- but the wrap must also leave the "
        "low-frequency energy intact.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# ---------------------------------------------------------------------------
# 3. Hermitian symmetry and the spurious imaginary channel
# ---------------------------------------------------------------------------


def hermitian_figure(cache_root: str = "cache", slice_index: int = 3
                     ) -> Tuple[plt.Figure, List[Dict]]:
    """Where the non-zero "imaginary channel" comes from for a real-valued dataset."""
    images = _test_slices(cache_root)
    truth = images[slice_index]

    rows: List[Dict] = []
    for ratio in RATIOS:
        pattern = _row_mask("variable_density", ratio)
        mirror = np.roll(pattern[::-1], 1)  # the conjugate partner of each row
        paired = float((pattern * mirror).sum() / pattern.sum())

        recon = _zero_fill(truth, pattern)
        e_real = float((recon.real ** 2).mean())
        e_imag = float((recon.imag ** 2).mean())

        symmetric = np.maximum(pattern, mirror)
        recon_sym = _zero_fill(truth, symmetric)
        e_imag_sym = float((recon_sym.imag ** 2).mean())

        rows.append({
            "sampling ratio": ratio,
            "conjugate pairs kept": round(paired, 3),
            "imag energy share": round(e_imag / (e_real + e_imag), 4),
            "PSNR_imag of zero-filling (dB)": round(10 * np.log10(4.0 / e_imag), 2),
            "symmetrized mask rate": round(float(symmetric.mean()), 3),
            "PSNR_imag, symmetrized (dB)": round(10 * np.log10(4.0 / max(e_imag_sym, 1e-30)), 1),
        })

    ratio = 0.3
    pattern = _row_mask("variable_density", ratio)
    mirror = np.roll(pattern[::-1], 1)
    recon = _zero_fill(truth, pattern)
    recon_sym = _zero_fill(truth, np.maximum(pattern, mirror))

    fig = plt.figure(figsize=(14.0, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.34, wspace=0.24)

    # (a) which acquired lines have their conjugate partner acquired too
    ax = fig.add_subplot(grid[0, 0])
    offsets = np.arange(N_ROWS) - N_ROWS // 2
    ax.fill_between(offsets, pattern, step="mid", color="0.8", label="acquired")
    ax.fill_between(offsets, pattern * mirror, step="mid", color="tab:green",
                    label="conjugate pair also acquired")
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("$k_y$ relative to the centre of k-space")
    kept = float((pattern * mirror).sum() / pattern.sum())
    ax.set_title(f"(a) A real image needs $F(-k) = F^*(k)$, but at {int(ratio * 100)}% "
                 f"sampling\nonly {kept * 100:.0f}% of the acquired lines have their "
                 "partner", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # (b) the imaginary part the undersampling creates -- the only image worth showing,
    #     because its structure is the evidence that the mask caused it.
    ax = fig.add_subplot(grid[0, 1])
    handle = ax.imshow(to_display(recon.imag), cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    ax.axis("off")
    ax.set_title("(b) So the zero-filled reconstruction is no longer real:\n"
                 "an imaginary part appears, structured along $k_y$", fontsize=10)
    bar = fig.colorbar(handle, ax=ax, fraction=0.046)
    bar.set_label("imaginary part", fontsize=8)

    # (c) how big that imaginary part is, in the three cases, on a log axis
    ax = fig.add_subplot(grid[1, 0])
    cases = [
        ("fully sampled reference\n(real by construction)",
         float(np.sqrt((np.zeros_like(truth) ** 2).mean())), "0.55"),
        (f"zero-filled, our mask\n({kept * 100:.0f}% of pairs kept)",
         float(np.sqrt((recon.imag ** 2).mean())), "tab:red"),
        ("zero-filled, mask made\nconjugate-symmetric",
         float(np.sqrt((recon_sym.imag ** 2).mean())), "tab:green"),
    ]
    floor = 1e-10
    values = [max(v, floor) for _, v, _ in cases]
    bars = ax.barh([c[0] for c in cases], values, color=[c[2] for c in cases],
                   edgecolor="black", height=0.55)
    for rect, (_, raw, _) in zip(bars, cases):
        label = "0, exactly" if raw == 0 else f"{raw:.1e}"
        ax.text(max(raw, floor) * 1.6, rect.get_y() + rect.get_height() / 2, label,
                va="center", fontsize=9, family="monospace")
    ax.set_xscale("log")
    ax.set_xlim(floor, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("RMS of the imaginary part (log scale)")
    ax.set_title("(c) Restoring the symmetry removes it: nine orders of\n"
                 "magnitude between a symmetric mask and ours", fontsize=10)
    ax.grid(axis="x", alpha=0.3, which="both")

    # (d) the energy share of the spurious channel, per sampling ratio
    ax = fig.add_subplot(grid[1, 1])
    shares = [r["imag energy share"] * 100 for r in rows]
    pairs = [r["conjugate pairs kept"] * 100 for r in rows]
    bars = ax.bar([f"{int(r * 100)}%" for r in RATIOS], shares,
                  color=["tab:blue", "tab:orange", "tab:green"], edgecolor="black",
                  width=0.55)
    for rect, value, pair in zip(bars, shares, pairs):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.4,
                f"{value:.0f}%\n({pair:.0f}% of\npairs kept)", ha="center", fontsize=8.5)
    ax.set_ylabel("share of zero-filled image energy\nheld in the imaginary part (%)",
                  fontsize=9)
    ax.set_xlabel("fraction of phase-encode lines acquired")
    ax.set_ylim(0, max(shares) * 1.5)
    ax.set_title("(d) The more nearly symmetric the mask happens to be,\n"
                 "the less spurious phase it injects", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("A real-valued image has conjugate-symmetric k-space; a random mask "
                 "breaks that symmetry\n(the same symmetry partial-Fourier acquisition "
                 "exploits)", fontsize=12.5)
    return fig, rows


# ---------------------------------------------------------------------------
# 4. The acquisition trade-off, made quantitative
# ---------------------------------------------------------------------------


def tradeoff_figure(results_root: str = "results") -> Tuple[plt.Figure, List[Dict]]:
    """Scan time and the sqrt(N) SNR penalty of undersampling, against measured quality."""
    measured = _measured_psnr(results_root)

    rows: List[Dict] = []
    for ratio in RATIOS:
        lines = int(round(ratio * N_ROWS))
        rows.append({
            "sampling ratio": ratio,
            "phase-encode lines": lines,
            "acceleration R": round(N_ROWS / lines, 2),
            "scan time (fraction of full)": round(lines / N_ROWS, 3),
            "predicted SNR loss (dB)": round(10 * np.log10(ratio), 2),
            "zero-filled PSNR (dB)": measured.get(("zero_filled", ratio), float("nan")),
            "baseline PSNR (dB)": measured.get(("classical_cs_tv", ratio), float("nan")),
            "our model PSNR (dB)": measured.get(("admmnet_softthresh", ratio), float("nan")),
        })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))

    accelerations = [N_ROWS / int(round(r * N_ROWS)) for r in RATIOS]
    labels = [f"{int(r * 100)}%" for r in RATIOS]

    # (a) what undersampling buys: scan time
    axes[0].bar(labels, [1.0] * len(RATIOS), color="0.88", edgecolor="black",
                label="fully sampled")
    axes[0].bar(labels, list(RATIOS), color="tab:blue", edgecolor="black",
                label="accelerated")
    for x, (ratio, acc) in enumerate(zip(RATIOS, accelerations)):
        axes[0].text(x, ratio + 0.03, f"R = {acc:.1f}$\\times$\nfaster",
                     ha="center", fontsize=9)
    axes[0].set_ylabel("acquisition time (fraction of a full scan)")
    axes[0].set_xlabel("fraction of phase-encode lines acquired")
    axes[0].set_ylim(0, 1.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("(a) What acceleration buys\n"
                      "scan time $\\propto N_{PE}\\cdot TR\\cdot NSA$", fontsize=10)

    # (b) what it costs: fewer samples averaged into each voxel
    axes[1].bar(labels, [10 * np.log10(r) for r in RATIOS], color="tab:red",
                edgecolor="black")
    for x, ratio in enumerate(RATIOS):
        value = 10 * np.log10(ratio)
        axes[1].text(x, value - 0.45, f"{value:.1f} dB", ha="center", fontsize=9)
    axes[1].set_ylabel("intrinsic SNR change (dB)")
    axes[1].set_xlabel("fraction of phase-encode lines acquired")
    axes[1].set_title("(b) What it costs\n"
                      "SNR $\\propto\\sqrt{N_{samples}}$", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    # (c) what reconstruction recovers
    for method, label, colour in (
            ("zero_filled", "zero-filled (no reconstruction)", "0.45"),
            ("classical_cs_tv", "baseline: CS (wavelet + TV)", "tab:orange"),
            ("admmnet_softthresh", "our model: ADMM-Net", "tab:green")):
        values = [measured.get((method, r), np.nan) for r in RATIOS]
        if all(np.isnan(v) for v in values):
            continue
        axes[2].plot(RATIOS, values, marker="o", linewidth=2, label=label, color=colour)
    axes[2].set_xlabel("fraction of phase-encode lines acquired")
    axes[2].set_ylabel("PSNR, real channel (dB)")
    axes[2].set_title("(c) What reconstruction recovers\n"
                      "(test set, pooled over three mask realizations)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Resolution, SNR and scan time are not independent: the trade-off our "
                 "experiment sits inside", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, rows


def _measured_psnr(results_root: str) -> Dict[Tuple[str, float], float]:
    """Mean PSNR on the real channel per (method, ratio), from ``samples.csv``."""
    path = os.path.join(results_root, "samples.csv")
    if not os.path.exists(path):
        return {}
    buckets: Dict[Tuple[str, float], List[float]] = {}
    seen = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["name"] != "comparison" or row["split"] != "test":
                continue
            key = (row["method"], float(row["sampling_ratio"]))
            unique = key + (row["seed"], row["sample_index"])
            if unique in seen:      # the zero-filled reference is logged by every run
                continue
            seen.add(unique)
            buckets.setdefault(key, []).append(float(row["psnr_real"]))
    return {k: round(float(np.mean(v)), 2) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# 5. Contrast weighting of the dataset
# ---------------------------------------------------------------------------


def contrast_figure(cache_root: str = "cache", slice_index: int = 3) -> plt.Figure:
    """Identify the contrast weighting of the dataset from the images themselves."""
    images = _test_slices(cache_root)
    truth = images[slice_index]

    fig = plt.figure(figsize=(16.0, 5.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.15], wspace=0.28)

    # (a) an annotated slice
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(to_display(truth), cmap="gray")
    ax.axis("off")
    ax.set_title("(a) Central axial slice\n$128\\times128$, normalized to max magnitude 1",
                 fontsize=10)

    brain = truth[truth > 0.08]
    lo, mid, hi = np.percentile(brain, [5, 50, 97])
    legend = "\n".join([
        f"white matter (bright)  $\\approx$ {hi:.2f}",
        f"grey matter (mid)      $\\approx$ {mid:.2f}",
        f"CSF / ventricles (dark) $\\approx$ {lo:.2f}",
    ])
    ax.text(0.02, -0.04, legend, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9,
                      edgecolor="0.7"))

    # (b) the intensity histogram, where the tissue classes separate
    ax = fig.add_subplot(grid[0, 1])
    pooled = images[:40]
    inside = pooled[pooled > 0.02]
    ax.hist(inside, bins=160, color="0.55", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(lo, color="tab:cyan", linestyle="--", linewidth=1.4, label="CSF level")
    ax.axvline(mid, color="tab:orange", linestyle="--", linewidth=1.4, label="grey matter")
    ax.axvline(hi, color="tab:red", linestyle="--", linewidth=1.4, label="white matter")
    ax.set_xlabel("normalized intensity")
    ax.set_ylabel("voxel count (log)")
    ax.set_title("(b) Intensity histogram over 40 slices:\n"
                 "CSF darkest, white matter brightest", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) the course's relaxation table and the resulting weighting
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    lines = ["Brain tissue at 1.5 T (course, class 3)", "",
             f"{'tissue':<14}{'PD':>6}{'T2 [ms]':>10}{'T1 [ms]':>10}"]
    lines += [f"{name:<14}{pd:>6.2f}{t2:>10.0f}{t1:>10.0f}"
              for name, pd, t2, t1 in TISSUE_TABLE]
    lines += ["", "CSF has the highest proton density, yet it is",
              "the darkest tissue in these images. Only a",
              "short-TR / short-TE (T1-weighted) acquisition",
              "produces that ordering: CSF's very long T1",
              "(2650 ms) leaves it far from full recovery, so",
              "it contributes almost no signal, while white",
              "matter's short T1 (510 ms) recovers most fully.",
              "",
              "=> the dataset is T1-weighted structural MRI,",
              "   which is what fixes the sparsity structure",
              "   both reconstruction methods rely on."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.0,
            family="monospace")
    ax.set_title("(c) The relaxation times behind that ordering", fontsize=10)

    fig.suptitle("Contrast weighting of the reconstruction dataset, read off the "
                 "images themselves", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# 6. Which lines the mask happens to draw
# ---------------------------------------------------------------------------


def dc_line_figure(cache_root: str = "cache", results_root: str = "results"
                   ) -> Tuple[plt.Figure, List[Dict]]:
    """Does a mask realization acquire the centre of k-space, and does it matter?

    The three seeds in the headline comparison differ only in the mask realization. Our
    mask draws lines from a normal distribution with ``center_lines = 0``, i.e. nothing
    forces the DC line to be acquired -- and the DC line alone carries about half the
    energy of these slices. This figure tests whether "which lines were drawn" explains
    the large seed-to-seed spread we measured.
    """
    images = _test_slices(cache_root)
    profile = _ky_energy_profile(images)
    profile = profile / profile.sum()
    centre = N_ROWS // 2
    measured = _measured_psnr_by_seed(results_root)

    rows: List[Dict] = []
    for ratio in RATIOS:
        for seed in (0, 1, 2):
            pattern = _row_mask("variable_density", ratio, seed=seed)
            rows.append({
                "sampling ratio": ratio,
                "seed": seed,
                "DC line acquired": bool(pattern[centre]),
                "central 5 lines acquired": int(pattern[centre - 2 : centre + 3].sum()),
                "captured energy": round(float((pattern * profile).sum()), 4),
                "zero-filled PSNR (dB)": measured.get(("zero_filled", ratio, seed), np.nan),
                "baseline PSNR (dB)": measured.get(("classical_cs_tv", ratio, seed), np.nan),
                "our model PSNR (dB)": measured.get(("admmnet_softthresh", ratio, seed), np.nan),
            })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) the three mask realizations at 30%, zoomed on the centre of k-space
    ratio = 0.3
    for seed, colour in zip((0, 1, 2), ("tab:blue", "tab:orange", "tab:green")):
        pattern = _row_mask("variable_density", ratio, seed=seed)
        acquired = np.where(pattern > 0)[0] - centre
        axes[0].scatter(acquired, np.full_like(acquired, seed, dtype=float),
                        s=18, color=colour, marker="|", linewidths=2)
        got_dc = pattern[centre] > 0
        axes[0].text(-30, seed + 0.18,
                     f"seed {seed}: DC line {'ACQUIRED' if got_dc else 'MISSED'}"
                     f"  ({(pattern * profile).sum() * 100:.0f}% of energy captured)",
                     fontsize=8.5, color=colour, fontweight="bold")
    axes[0].axvline(0, color="crimson", linestyle="--", linewidth=1.4)
    axes[0].text(1, -0.55, "DC ($k_y = 0$): 52% of all energy",
                 color="crimson", fontsize=8.5, rotation=90, va="bottom")
    axes[0].set_xlim(-32, 32)
    axes[0].set_ylim(-0.7, 2.6)
    axes[0].set_yticks([0, 1, 2], ["seed 0", "seed 1", "seed 2"])
    axes[0].set_xlabel("$k_y$ relative to the centre of k-space")
    axes[0].set_title(f"(a) The three mask realizations at {int(ratio * 100)}% sampling\n"
                      "(central region only)", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    # (b) captured energy vs measured quality, all nine (ratio, seed) runs
    captured = np.array([r["captured energy"] for r in rows]) * 100
    markers = {0.2: "o", 0.3: "s", 0.5: "^"}
    for method, label, colour in (
            ("zero-filled PSNR (dB)", "zero-filled", "0.45"),
            ("baseline PSNR (dB)", "baseline: CS (wavelet + TV)", "tab:orange"),
            ("our model PSNR (dB)", "our model: ADMM-Net", "tab:green")):
        values = np.array([r[method] for r in rows], dtype=float)
        good = ~np.isnan(values)
        if not good.any():
            continue
        for ratio in RATIOS:
            sel = good & np.array([r["sampling ratio"] == ratio for r in rows])
            axes[1].scatter(captured[sel], values[sel], color=colour,
                            marker=markers[ratio], s=55, edgecolor="black", linewidth=0.5)
        r_value = float(np.corrcoef(captured[good], values[good])[0, 1])
        axes[1].plot([], [], "o", color=colour, label=f"{label}  (r = {r_value:+.2f})")
    for ratio in RATIOS:
        axes[1].scatter([], [], color="0.3", marker=markers[ratio],
                        label=f"{int(ratio * 100)}% sampling")
    axes[1].set_xlabel("% of the test set's k-space energy the mask actually acquired")
    axes[1].set_ylabel("mean PSNR, real channel (dB)")
    axes[1].set_title("(b) Mask realization quality predicts reconstruction quality\n"
                      "(9 runs = 3 ratios $\\times$ 3 seeds)", fontsize=10)
    axes[1].legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    axes[1].grid(alpha=0.3)

    # (c) the same question asked properly: average over 200 realizations, and compare
    #     against uniform random sampling and against forcing a central band.
    ax = axes[2]
    truth = images[3]
    schemes = [
        ("uniform random", lambda seed: _row_mask("uniform", ratio, seed=seed), "0.6"),
        ("variable density\n(ours)",
         lambda seed: _row_mask("variable_density", ratio, seed=seed), "tab:orange"),
        ("variable density\n+ 3 central lines",
         lambda seed: _forced_centre_mask(ratio, 3, seed), "tab:blue"),
        ("variable density\n+ 7 central lines",
         lambda seed: _forced_centre_mask(ratio, 7, seed), "tab:green"),
    ]
    draws = 200
    labels, rmses, energies = [], [], []
    for label, make, colour in schemes:
        errors, captured = [], []
        for seed in range(draws):
            pattern = make(seed)
            recon = np.abs(_zero_fill(truth, pattern))
            errors.append(float(np.sqrt(((recon - truth) ** 2).mean())))
            captured.append(float((pattern * profile).sum()))
        labels.append(label)
        rmses.append(float(np.mean(errors)))
        energies.append(100 * float(np.mean(captured)))

    positions = np.arange(len(labels))
    bars = ax.bar(positions, rmses, color=[c for _, _, c in schemes], edgecolor="black")
    for bar, rmse, energy in zip(bars, rmses, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, rmse + 0.004,
                f"{rmse:.3f}\n({energy:.0f}% of\nenergy)", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("zero-filled RMSE, mean over 200 mask draws")
    ax.set_ylim(0, max(rmses) * 1.42)
    ax.set_title("(c) Averaged over 200 realizations, not 3:\n"
                 "variable density beats uniform, and forcing a\n"
                 "central band beats both", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("The hidden variable behind our seed-to-seed spread: whether the mask "
                 "drew the centre of k-space", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, rows


def _measured_psnr_by_seed(results_root: str) -> Dict[Tuple[str, float, int], float]:
    """Mean PSNR on the real channel per (method, ratio, seed), from ``samples.csv``."""
    path = os.path.join(results_root, "samples.csv")
    if not os.path.exists(path):
        return {}
    buckets: Dict[Tuple[str, float, int], List[float]] = {}
    seen = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["name"] != "comparison" or row["split"] != "test":
                continue
            key = (row["method"], float(row["sampling_ratio"]), int(row["seed"]))
            unique = key + (row["sample_index"],)
            if unique in seen:
                continue
            seen.add(unique)
            buckets.setdefault(key, []).append(float(row["psnr_real"]))
    return {k: round(float(np.mean(v)), 2) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_FIGURES = ("sequence", "energy", "psf", "hermitian", "tradeoff", "contrast",
               "dcline")


def _write_table(path: str, rows: Sequence[Dict]) -> None:
    """Persist a figure's underlying numbers next to it, as CSV."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--which", nargs="*", default=list(ALL_FIGURES), choices=ALL_FIGURES)
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--slice-index", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    out_dir = _figure_dir(cfg)
    written: List[str] = []

    def _save(fig: plt.Figure, stem: str) -> None:
        path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}")

    if "sequence" in args.which:
        _save(sequence_figure(), "mri_pulse_sequence")

    if "energy" in args.which:
        fig, rows = energy_figure(args.cache_root)
        _save(fig, "mri_kspace_energy")
        _write_table(os.path.join(out_dir, "mri_kspace_energy.csv"), rows)

    if "psf" in args.which:
        _save(psf_figure(args.cache_root, slice_index=args.slice_index), "mri_psf_aliasing")

    if "hermitian" in args.which:
        fig, rows = hermitian_figure(args.cache_root, slice_index=args.slice_index)
        _save(fig, "mri_hermitian")
        _write_table(os.path.join(out_dir, "mri_hermitian.csv"), rows)

    if "tradeoff" in args.which:
        fig, rows = tradeoff_figure(cfg["paths"]["results_root"])
        _save(fig, "mri_tradeoff")
        _write_table(os.path.join(out_dir, "mri_tradeoff.csv"), rows)

    if "contrast" in args.which:
        _save(contrast_figure(args.cache_root, slice_index=args.slice_index),
              "mri_contrast")

    if "dcline" in args.which:
        fig, rows = dc_line_figure(args.cache_root, cfg["paths"]["results_root"])
        _save(fig, "mri_dc_line")
        _write_table(os.path.join(out_dir, "mri_dc_line.csv"), rows)

    return written




# ---------------------------------------------------------------------------
# 7. Where the undersampling actually happens in the pulse sequence
# ---------------------------------------------------------------------------


def sequence_figure(ratio: float = 0.3) -> plt.Figure:
    """A 2D gradient-echo timing diagram showing that one TR buys one k-space line.

    Every other figure in this report starts from k-space. This one starts one step
    earlier, at the pulse sequence, because that is where the thing we are saving --
    time -- is actually spent: the phase-encode loop runs once per line, so a line we do
    not acquire is a repetition we do not play out.
    """
    keep = int(round(ratio * N_ROWS))
    fig = plt.figure(figsize=(15.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.2)

    # ---- left: the sequence timing diagram --------------------------------
    ax = fig.add_subplot(grid[0, 0])
    channels = ["RF", "$G_z$\n(slice select)", "$G_y$\n(phase encode)",
                "$G_x$\n(readout)", "ADC"]
    baselines = {name: 4 - i for i, name in enumerate(channels)}
    for name, y0 in baselines.items():
        ax.axhline(y0, color="0.85", linewidth=0.8, zorder=0)
        ax.text(-0.6, y0, name, ha="right", va="center", fontsize=9)

    def blip(x0, width, height, y0, colour, alpha=1.0, hatch=None):
        ax.add_patch(plt.Rectangle((x0, y0), width, height, facecolor=colour,
                                   edgecolor="black", linewidth=0.8, alpha=alpha,
                                   hatch=hatch, zorder=3))

    for rep, x_offset in enumerate((0.0, 6.2)):
        y = baselines["RF"]
        t = np.linspace(0, 1.4, 200)
        envelope = np.sinc(3 * (t - 0.7))
        ax.plot(x_offset + 0.3 + t, y + 0.34 * envelope, color="tab:red", linewidth=1.3,
                zorder=3)
        ax.text(x_offset + 1.0, y + 0.45, r"$\alpha^\circ$", fontsize=9, color="tab:red",
                ha="center")

        blip(x_offset + 0.3, 1.4, 0.34, baselines["$G_z$\n(slice select)"], "tab:blue")
        blip(x_offset + 1.7, 0.7, -0.22, baselines["$G_z$\n(slice select)"],
             "tab:blue", alpha=0.55)

        # The phase-encode blip: several amplitudes drawn on top of each other.
        y = baselines["$G_y$\n(phase encode)"]
        for k, amplitude in enumerate(np.linspace(-0.42, 0.42, 7)):
            ax.add_patch(plt.Rectangle((x_offset + 1.8, y), 0.6, amplitude,
                                       facecolor="tab:green", edgecolor="black",
                                       linewidth=0.6, alpha=0.35, zorder=3))
        ax.annotate("", xy=(x_offset + 2.1, y + 0.5), xytext=(x_offset + 2.1, y - 0.5),
                    arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.3))

        blip(x_offset + 2.6, 0.8, -0.28, baselines["$G_x$\n(readout)"], "tab:purple",
             alpha=0.55)
        blip(x_offset + 3.4, 1.9, 0.34, baselines["$G_x$\n(readout)"], "tab:purple")
        blip(x_offset + 3.5, 1.7, 0.3, baselines["ADC"], "0.55")
        ax.text(x_offset + 4.35, baselines["ADC"] + 0.15, "one $k_x$ readout",
                fontsize=7.5, va="center")

        ax.annotate("", xy=(x_offset + 0.3, 4.85), xytext=(x_offset + 6.5, 4.85),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1))
        ax.text(x_offset + 3.4, 4.95, "TR", ha="center", fontsize=9.5, color="0.25")
        ax.text(x_offset + 3.4, 5.35,
                f"repetition {rep + 1}  $\\rightarrow$  $k_y$ line {rep + 1}",
                ha="center", fontsize=9)

    ax.text(12.9, 3.2, "$\\cdots$ repeat once per\nphase-encode line $\\cdots$",
            fontsize=10, ha="center", va="center", color="0.3")

    ax.set_xlim(-2.6, 15.2)
    ax.set_ylim(-0.6, 5.9)
    ax.axis("off")
    ax.set_title("(a) A 2D Cartesian gradient-echo sequence: one repetition, one line of "
                 "k-space\nThe phase-encode blip (green) is the only thing that changes "
                 "between repetitions.", fontsize=10.5, loc="left")

    # ---- right: what skipping repetitions buys and costs -------------------
    ax = fig.add_subplot(grid[0, 1])
    pattern = _row_mask("variable_density", ratio)
    acquired = np.where(pattern > 0)[0]
    skipped = np.where(pattern == 0)[0]

    ax.scatter(np.zeros_like(skipped), skipped - N_ROWS // 2, marker="_", s=90,
               color="0.82", linewidths=1.4, label=f"skipped ({len(skipped)} TRs saved)")
    ax.scatter(np.zeros_like(acquired), acquired - N_ROWS // 2, marker="_", s=90,
               color="tab:green", linewidths=1.6,
               label=f"acquired ({len(acquired)} TRs played out)")
    ax.set_xlim(-0.6, 1.9)
    ax.set_xticks([])
    ax.set_ylabel("$k_y$ relative to the centre of k-space")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"(b) Keeping {int(ratio * 100)}% of the lines\n"
                 f"$T_{{scan}}$ falls to {len(acquired)}/{N_ROWS} = "
                 f"{100 * len(acquired) / N_ROWS:.0f}% "
                 f"($R = {N_ROWS / len(acquired):.1f}\\times$)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Where the saving comes from: undersampling k-space means playing out "
                 "fewer repetitions of the sequence", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig

if __name__ == "__main__":
    main()
