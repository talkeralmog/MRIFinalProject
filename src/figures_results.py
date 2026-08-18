# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""The brief's required results plots, drawn per channel.

``src/analysis.py`` plots PSNR/SSIM averaged over the real and imaginary channels. That
average turned out to be misleading on this dataset: the ground-truth imaginary channel
is identically zero (the course volumes are magnitude images), so the imaginary-channel
metric measures suppression of the spurious phase that the non-Hermitian mask injects,
not phase fidelity -- and its absolute dB is inflated because the MSE is taken against
zero. See ``src/figures_mri.hermitian_figure``.

This module therefore redraws the two required line plots and the two required scatter
plots with the **real and imaginary channels side by side**, which is what the brief asks
for ("compute PSNR and SSIM on the real and imaginary components separately"), and labels
the methods in words rather than by registry key.

Everything is read from ``results/samples.csv``, so no dataset, checkpoint or GPU is
needed.

Usage::

    python -m src.figures_results --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config

RATIOS = (0.2, 0.3, 0.5)
SEEDS = (0, 1, 2)

METHODS = [
    ("zero_filled", "zero-filled input (no reconstruction)", "0.45"),
    ("classical_cs", "naive CS (single-level wavelet)", "tab:purple"),
    ("classical_cs_tv", "baseline: classical CS (wavelet $\\ell_1$ + TV)", "tab:orange"),
    ("admmnet_softthresh", "our model: unrolled ADMM-Net", "tab:green"),
]
RATIO_COLOURS = {0.2: "tab:blue", 0.3: "tab:orange", 0.5: "tab:green"}
CHANNELS = (("real", "real channel"), ("imag", "imaginary channel"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_paired_samples(results_root: str, experiment: str = "comparison",
                        split: str = "test") -> Dict[Tuple[str, float, int, str], float]:
    """``(method, ratio, seed, sample_index) -> {metric: value}`` from ``samples.csv``.

    De-duplicates the zero-filled reference, which every run logs a copy of.
    """
    path = os.path.join(results_root, "samples.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no samples.csv in {results_root}")
    out: Dict[Tuple[str, float, int, str], Dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["name"] != experiment or row["split"] != split:
                continue
            key = (row["method"], float(row["sampling_ratio"]), int(row["seed"]),
                   row["sample_index"])
            if key in out:            # duplicate zero-filled row
                continue
            out[key] = {m: float(row[m]) for m in
                        ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag")}
    return out


def _values(samples, method: str, ratio: float, metric: str,
            seeds: Sequence[int] = SEEDS) -> np.ndarray:
    """All per-slice values of one metric for a (method, ratio), pooling the seeds."""
    return np.array([v[metric] for (m, r, s, _), v in samples.items()
                     if m == method and r == ratio and s in seeds])


# ---------------------------------------------------------------------------
# 1. Required line plots: metric vs sampling ratio
# ---------------------------------------------------------------------------


def metric_vs_ratio(samples, base: str = "psnr") -> plt.Figure:
    """One panel per channel: metric vs sampling ratio, band = 1 std across the test set."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharex=True)

    for ax, (channel, channel_label) in zip(axes, CHANNELS):
        metric = f"{base}_{channel}"
        for method, label, colour in METHODS:
            means, stds, xs = [], [], []
            for ratio in RATIOS:
                v = _values(samples, method, ratio, metric)
                if v.size == 0:
                    continue
                xs.append(ratio)
                means.append(v.mean())
                stds.append(v.std(ddof=1))
            if not xs:
                continue
            means, stds = np.array(means), np.array(stds)
            ax.plot(xs, means, marker="o", linewidth=2, color=colour, label=label)
            ax.fill_between(xs, means - stds, means + stds, color=colour, alpha=0.15)

        ax.set_xlabel("sampling ratio (fraction of phase-encode lines acquired)")
        ax.set_ylabel(f"{base.upper()}" + (" (dB)" if base == "psnr" else ""))
        ax.set_xticks(list(RATIOS))
        ax.set_title(channel_label, fontsize=11)
        ax.grid(alpha=0.3)

    axes[1].text(0.98, 0.03,
                 "reference $\\equiv 0$ here:\nthis panel measures suppression of\n"
                 "the spurious phase, not phase fidelity",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8.5,
                 style="italic",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                           edgecolor="0.7"))

    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle(f"{base.upper()} vs sampling ratio, real and imaginary channels "
                 "separately\n(test set, n = 3 seeds $\\times$ 478 slices per point; "
                 "shaded band = $\\pm$1 std across the test set)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# ---------------------------------------------------------------------------
# 2. Required scatter plots: sample-wise baseline vs our model
# ---------------------------------------------------------------------------


def scatter_baseline_vs_model(samples, base: str = "psnr",
                              baseline: str = "classical_cs_tv",
                              model: str = "admmnet_softthresh",
                              seed: int = 0) -> plt.Figure:
    """One panel per channel: per-slice baseline vs our model, coloured by sampling ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))

    for ax, (channel, channel_label) in zip(axes, CHANNELS):
        metric = f"{base}_{channel}"
        all_x, all_y = [], []
        for ratio in RATIOS:
            b = {k[3]: v[metric] for k, v in samples.items()
                 if k[0] == baseline and k[1] == ratio and k[2] == seed}
            m = {k[3]: v[metric] for k, v in samples.items()
                 if k[0] == model and k[1] == ratio and k[2] == seed}
            keys = sorted(set(b) & set(m))
            if not keys:
                continue
            x = np.array([b[k] for k in keys])
            y = np.array([m[k] for k in keys])
            all_x.append(x); all_y.append(y)
            wins = int((y > x).sum())
            r = float(np.corrcoef(x, y)[0, 1])
            ax.scatter(x, y, s=13, alpha=0.55, color=RATIO_COLOURS[ratio],
                       label=f"{int(ratio * 100)}% lines:  r = {r:+.3f},  "
                             f"we win {wins}/{len(keys)}")

        if not all_x:
            continue
        x_all = np.concatenate(all_x); y_all = np.concatenate(all_y)
        lo = min(x_all.min(), y_all.min()); hi = max(x_all.max(), y_all.max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.7,
                linewidth=1.2, label="$y = x$ (tie)")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")

        r_all = float(np.corrcoef(x_all, y_all)[0, 1])
        wins_all = int((y_all > x_all).sum())
        unit = " (dB)" if base == "psnr" else ""
        ax.set_xlabel(f"baseline: classical CS {base.upper()}{unit}")
        ax.set_ylabel(f"our model: ADMM-Net {base.upper()}{unit}")
        ax.set_title(f"{channel_label}\npooled r = {r_all:+.3f}, "
                     f"we win {wins_all}/{len(x_all)} slice-ratio pairs", fontsize=10.5)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Sample-wise {base.upper()}: every point is one test slice "
                 f"(seed {seed}); points above the dashed line are ones we win",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# 3. Headline table, as CSV, so the report and the figures cannot disagree
# ---------------------------------------------------------------------------


def headline_rows(samples) -> List[Dict]:
    """``mean ± std across the test set`` per (ratio, method, metric), as flat rows."""
    rows: List[Dict] = []
    for ratio in RATIOS:
        for method, label, _ in METHODS:
            row: Dict[str, object] = {"sampling ratio": ratio, "method": label}
            n = 0
            for metric in ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag"):
                v = _values(samples, method, ratio, metric)
                if v.size == 0:
                    continue
                n = v.size
                digits = 4 if metric.startswith("ssim") else 2
                row[metric] = f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"
            if n:
                row["n"] = n
                rows.append(row)
    return rows


def win_rate_rows(samples, baseline: str = "classical_cs_tv",
                  model: str = "admmnet_softthresh") -> List[Dict]:
    """Paired win counts of our model over the baseline, per ratio and metric."""
    rows: List[Dict] = []
    for ratio in RATIOS:
        for metric in ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag"):
            b = {(k[2], k[3]): v[metric] for k, v in samples.items()
                 if k[0] == baseline and k[1] == ratio}
            m = {(k[2], k[3]): v[metric] for k, v in samples.items()
                 if k[0] == model and k[1] == ratio}
            keys = sorted(set(b) & set(m))
            if not keys:
                continue
            wins = sum(1 for k in keys if m[k] > b[k])
            rows.append({"sampling ratio": ratio, "metric": metric,
                         "our model wins": wins, "pairs": len(keys),
                         "win rate": round(wins / len(keys), 4)})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--seed", type=int, default=0, help="Seed shown in the scatter plots.")
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    samples = load_paired_samples(results_root)
    written: List[str] = []

    def _save(fig: plt.Figure, stem: str) -> None:
        path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"wrote {path}")

    for base in ("psnr", "ssim"):
        _save(metric_vs_ratio(samples, base), f"{base}_vs_ratio_per_channel")
        _save(scatter_baseline_vs_model(samples, base, seed=args.seed),
              f"scatter_{base}_per_channel")

    _write_csv(os.path.join(out_dir, "headline_by_channel.csv"), headline_rows(samples))
    _write_csv(os.path.join(out_dir, "win_rates.csv"), win_rate_rows(samples))
    return written


if __name__ == "__main__":
    main()
