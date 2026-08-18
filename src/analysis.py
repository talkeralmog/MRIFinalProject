# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Analysis helpers: turn the logged results into the report's MRI figures and tables.

These functions keep ``analysis.ipynb`` thin. They read two logs written by training:

* ``results/runs.csv``    -- dataset-wide mean/std per metric per run (long format).
* ``results/samples.csv`` -- per-sample per-channel metrics (for scatter / Pearson r
  and for selecting qualitative examples).

Metrics are per channel (``psnr_real``/``psnr_imag``/``ssim_real``/``ssim_imag``); the
line plots and scatter plots use the average of the real and imaginary components.
Plotting functions return a Matplotlib figure; table functions return a DataFrame.
All functions degrade gracefully (warn + empty figure/table) when a log is missing.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import build_datasets
from .display import to_display
from .engine import build_model, undersample
from .masks import build_mask
from .registry import MODEL_REGISTRY
from .utils import load_checkpoint, zero_filled_recon

# Importing these registers every model/baseline so checkpoints can be rebuilt.
from . import model as _model  # noqa: F401
from .baselines import zero_filled as _zf, classical_cs as _cs, unet as _unet, ista_net as _ista  # noqa: F401,E501

_HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "nmse": False}


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------


def load_results(results_dir: str) -> pd.DataFrame:
    """Load ``runs.csv`` from a results directory."""
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no runs.csv in {results_dir}; run some experiments first")
    return pd.read_csv(path)


def load_samples(results_dir: str) -> pd.DataFrame:
    """Load per-sample ``samples.csv`` from a results directory."""
    path = os.path.join(results_dir, "samples.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no samples.csv in {results_dir}; run some experiments first")
    return pd.read_csv(path)


def load_crossmask(results_dir: str) -> pd.DataFrame:
    """Load ``crossmask.csv`` written by ``python -m src.eval_crossmask``."""
    path = os.path.join(results_dir, "crossmask.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no crossmask.csv in {results_dir}; run `python -m src.eval_crossmask` first")
    return pd.read_csv(path)


def _select(df: pd.DataFrame, split: str, metric: str, **filters) -> pd.DataFrame:
    """Filter rows by split, metric, and arbitrary equality filters."""
    mask = (df["split"] == split) & (df["metric"] == metric)
    for key, value in filters.items():
        if value is not None and key in df.columns:
            mask &= df[key] == value
    return df[mask]


def comparable_runs(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Rows of one experiment, restricted to a single training budget.

    An ablation is only interpretable if every variant was trained the same way. Editing a
    sweep's ``train.epochs`` changes the run-id hash, so a re-run leaves the old rows behind
    and the same variant can end up logged twice under two different budgets. This keeps the
    budget that most runs of the experiment used and warns about anything it drops.
    """
    sub = df[df["name"] == experiment]
    if sub.empty or "epochs" not in sub.columns:
        return sub
    budgets = sub.drop_duplicates("run_id")["epochs"].value_counts()
    if len(budgets) <= 1:
        return sub
    keep = budgets.idxmax()
    dropped = sorted(set(budgets.index) - {keep})
    warnings.warn(f"experiment '{experiment}' contains runs trained for {dropped} epochs "
                  f"as well as {keep}; keeping only the {keep}-epoch runs so the variants "
                  "are comparable")
    return sub[sub["epochs"] == keep]


def aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate the per-run ``value`` column across seeds into mean and std."""
    grouped = df.groupby(list(group_cols))["value"]
    out = grouped.agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def combined_metric(df: pd.DataFrame, base: str, split: str = "test") -> pd.DataFrame:
    """Average the real & imaginary components of a metric into one value per run.

    Returns a tidy frame with columns ``[method, sampling_ratio, seed, run_id, value]``
    where ``value`` is the mean of ``<base>_real`` and ``<base>_imag`` for that run.
    """
    keep = [f"{base}_real", f"{base}_imag"]
    d = df[(df["split"] == split) & (df["metric"].isin(keep))]
    if d.empty:
        return d
    id_cols = [c for c in ("run_id", "method", "sampling_ratio", "seed") if c in d.columns]
    return (d.groupby(id_cols)["value"].mean().reset_index())


def select_samples(samples: pd.DataFrame, split: str = "test",
                   experiment: Optional[str] = "comparison") -> pd.DataFrame:
    """Per-slice rows for one experiment, with duplicate references removed.

    Two filters matter here and are easy to forget:

    * ``experiment`` -- every appendix ablation also runs at the 0.3 sampling ratio and
      also logs per-sample rows, so aggregating without this filter silently mixes the
      ablations into the headline numbers.
    * de-duplication -- the zero-filled reference is logged once per run, so it appears
      several times for the same ``(sampling_ratio, seed, sample_index)``. Those copies
      are identical and would inflate the sample count.
    """
    d = samples[samples["split"] == split]
    if experiment is not None and "name" in d.columns:
        d = d[d["name"] == experiment]
    key = [c for c in ("method", "sampling_ratio", "seed", "sample_index") if c in d.columns]
    return d.drop_duplicates(subset=key).copy()


def combined_samples(samples: pd.DataFrame, base: str, split: str = "test",
                     experiment: Optional[str] = "comparison") -> pd.DataFrame:
    """Per-slice metric averaged over the real and imaginary channels.

    The course brief asks for the mean and standard deviation **across the test set**, so
    every aggregate the report quotes is computed from these per-slice values rather than
    from the across-seed spread of run averages.
    """
    d = select_samples(samples, split, experiment)
    if d.empty:
        return d
    d["value"] = d[[f"{base}_real", f"{base}_imag"]].mean(axis=1)
    keep = [c for c in ("method", "sampling_ratio", "seed", "sample_index", "value")
            if c in d.columns]
    return d[keep]


# ---------------------------------------------------------------------------
# 1. Required results table: one row per sampling ratio x method
# ---------------------------------------------------------------------------


def results_table_by_ratio(
    samples: pd.DataFrame,
    methods: Sequence[str] = ("classical_cs_tv", "admmnet_softthresh"),
    split: str = "test",
    metrics: Sequence[str] = ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag"),
    experiment: Optional[str] = "comparison",
) -> pd.DataFrame:
    """Table of ``mean +/- std`` **across the test set** per (sampling_ratio, method).

    Built from the per-slice metrics, because the brief asks for the spread over the test
    set rather than the spread over seeds. Seeds are pooled, so each cell aggregates
    ``n_seeds x n_test_slices`` measurements.
    """
    d = select_samples(samples, split, experiment)
    d = d[d["method"].isin(methods)]
    if d.empty:
        warnings.warn("no per-sample rows for the results-by-ratio table")
        return pd.DataFrame()

    records: List[Dict] = []
    for metric in metrics:
        if metric not in d.columns:
            continue
        digits = 4 if metric.startswith("ssim") else 2
        agg = d.groupby(["sampling_ratio", "method"])[metric].agg(["mean", "std"])
        for (ratio, method), r in agg.iterrows():
            records.append({
                "sampling_ratio": ratio,
                "method": method,
                "metric": metric,
                "cell": f"{r['mean']:.{digits}f} +/- {r['std']:.{digits}f}",
            })
    if not records:
        warnings.warn("no rows for the results-by-ratio table")
        return pd.DataFrame()
    tidy = pd.DataFrame(records)
    table = tidy.pivot_table(index=["sampling_ratio", "method"], columns="metric",
                             values="cell", aggfunc="first")
    return table.sort_index()


# ---------------------------------------------------------------------------
# 2. Required line plots: metric vs sampling ratio, baseline vs model
# ---------------------------------------------------------------------------


def plot_metric_vs_ratio(
    samples: pd.DataFrame,
    base: str = "psnr",
    methods: Sequence[str] = ("classical_cs_tv", "admmnet_softthresh"),
    split: str = "test",
    experiment: Optional[str] = "comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Line plot of a metric (avg of real & imag) vs sampling ratio, one line per method.

    The shaded band is +/- one standard deviation **across the test set**, as the brief
    requires, computed from the per-slice metrics in ``samples.csv`` and pooled over the
    seeds (so it reflects both slice-to-slice difficulty and mask-realization variability).
    ``base`` is ``'psnr'`` or ``'ssim'``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    d = combined_samples(samples, base, split, experiment)
    if d.empty:
        warnings.warn(f"no per-sample rows for metric '{base}'")
        return ax.figure

    for method in methods:
        dm = d[d["method"] == method]
        if dm.empty:
            warnings.warn(f"no per-sample rows for '{method}'")
            continue
        agg = dm.groupby("sampling_ratio")["value"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        agg = agg.sort_values("sampling_ratio")
        ax.plot(agg["sampling_ratio"], agg["mean"], marker="o", linewidth=2, label=method)
        ax.fill_between(agg["sampling_ratio"], agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"], alpha=0.2)

    unit = " (dB)" if base == "psnr" else ""
    ax.set_xlabel("Sampling ratio (fraction of k-space lines kept)")
    ax.set_ylabel(f"{base.upper()}{unit}")
    ax.set_title(f"{base.upper()} vs sampling ratio ({split} set, band = 1 std across the set)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax.figure


# ---------------------------------------------------------------------------
# 3. Required scatter plots: sample-wise baseline vs model, with Pearson r
# ---------------------------------------------------------------------------


def scatter_baseline_vs_model(
    samples: pd.DataFrame,
    base: str = "psnr",
    baseline: str = "classical_cs_tv",
    model: str = "admmnet_softthresh",
    split: str = "test",
    seed: Optional[int] = 0,
    experiment: Optional[str] = "comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Sample-wise baseline-vs-model scatter (avg real/imag), coloured by ratio.

    Pearson r (over all points) is printed on the plot. Points are paired by
    (sampling_ratio, seed, sample_index) within a single experiment, so each test slice
    contributes exactly one point per sampling ratio.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    d = combined_samples(samples, base, split, experiment)
    if seed is not None and "seed" in d.columns:
        d = d[d["seed"] == seed]

    key = ["sampling_ratio", "seed", "sample_index"]
    b = d[d["method"] == baseline][key + ["value"]].rename(columns={"value": "baseline"})
    m = d[d["method"] == model][key + ["value"]].rename(columns={"value": "model"})
    merged = b.merge(m, on=key, how="inner")
    if merged.empty:
        warnings.warn("no paired baseline/model samples for the scatter plot")
        return ax.figure

    for ratio, g in merged.groupby("sampling_ratio"):
        ax.scatter(g["baseline"], g["model"], s=14, alpha=0.6, label=f"ratio {ratio}")

    r = float(np.corrcoef(merged["baseline"], merged["model"])[0, 1])
    lo = float(min(merged["baseline"].min(), merged["model"].min()))
    hi = float(max(merged["baseline"].max(), merged["model"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="y = x")

    unit = " (dB)" if base == "psnr" else ""
    ax.set_xlabel(f"Baseline {base.upper()}{unit} ({baseline})")
    ax.set_ylabel(f"Model {base.upper()}{unit} ({model})")
    ax.set_title(f"Sample-wise {base.upper()} (Pearson r = {r:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax.figure


# ---------------------------------------------------------------------------
# 4. Appendix ablations (depth / loss / structure), adapted to the new metrics
# ---------------------------------------------------------------------------


def plot_depth_vs_metric(df: pd.DataFrame, model_name: str = "admmnet_softthresh",
                         split: str = "test", base: str = "psnr",
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Metric (avg real/imag) vs unrolling depth for the depth_sweep experiment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    d = combined_metric(comparable_runs(df, "depth_sweep"), base, split)
    if d.empty:
        warnings.warn("no depth_sweep rows")
        return ax.figure
    # num_stages lives in runs.csv; join it back via run_id
    stages = df[["run_id", "num_stages"]].drop_duplicates()
    d = d.merge(stages, on="run_id", how="left")
    d = d[d["method"] == model_name]
    agg = d.groupby("num_stages")["value"].agg(["mean", "std"]).reset_index().sort_values("num_stages")
    agg["std"] = agg["std"].fillna(0.0)
    ax.errorbar(agg["num_stages"], agg["mean"], yerr=agg["std"], marker="o",
                capsize=4, linewidth=2)
    ax.set_xlabel("Number of ADMM stages (unrolling depth)")
    ax.set_ylabel(f"{base.upper()}{' (dB)' if base == 'psnr' else ''}")
    ax.set_title(f"{base.upper()} vs unrolling depth ({split} set)")
    ax.grid(True, alpha=0.3)
    return ax.figure


def plot_categorical_ablation(df: pd.DataFrame, experiment: str, label_cols: Sequence[str],
                              base: str = "psnr", split: str = "test",
                              ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Generic bar chart of a metric over a categorical ablation (loss / structure)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    sub = comparable_runs(df, experiment)
    d = combined_metric(sub, base, split)
    if d.empty:
        warnings.warn(f"no rows for experiment '{experiment}'")
        return ax.figure
    d = d[d["method"] != "zero_filled"]
    # Pull in only the label columns combined_metric did not already return, otherwise the
    # merge renames the shared ones with _x / _y suffixes and the labels break.
    missing = [c for c in label_cols if c in df.columns and c not in d.columns]
    if missing:
        d = d.merge(df[["run_id"] + missing].drop_duplicates(), on="run_id", how="left")
    available = [c for c in label_cols if c in d.columns]
    d["label"] = d[available].astype(str).agg("\n".join, axis=1)
    agg = d.groupby("label")["value"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg = agg.sort_values("mean", ascending=not _HIGHER_IS_BETTER[base])
    ax.bar(agg["label"], agg["mean"], yerr=agg["std"], capsize=4, color="0.6", edgecolor="black")
    ax.set_ylabel(f"{base.upper()}{' (dB)' if base == 'psnr' else ''}")
    ax.set_title(f"{experiment} ({split} set)")
    ax.grid(axis="y", alpha=0.3)
    return ax.figure


def plot_model_vs_unet(df: pd.DataFrame, model_name: str = "admmnet_softthresh",
                       reference: str = "unet", base: str = "psnr",
                       split: str = "test", ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Our unrolled model against a plain U-Net at each sampling ratio.

    The U-Net has no data-consistency step, so this isolates the contribution of the
    model-based structure rather than of network capacity (the U-Net is the larger model).
    Reads the ``unet_reference`` experiment and the headline ``comparison`` runs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    d = combined_metric(df, base, split)
    if d.empty:
        warnings.warn("no rows for the model-vs-U-Net comparison")
        return ax.figure

    for method, label in ((model_name, "our model (unrolled ADMM-Net)"),
                          (reference, "U-Net (no data consistency)")):
        sub = d[d["method"] == method]
        if sub.empty:
            warnings.warn(f"no runs found for '{method}'")
            continue
        agg = sub.groupby("sampling_ratio")["value"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        ax.errorbar(agg["sampling_ratio"], agg["mean"], yerr=agg["std"], marker="o",
                    capsize=4, linewidth=2, label=label)

    ax.set_xlabel("k-space sampling ratio (fraction of phase-encode lines kept)")
    ax.set_ylabel(f"{base.upper()}{' (dB)' if base == 'psnr' else ''}")
    ax.set_title(f"Model-based unrolling vs a generic CNN ({split} set)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax.figure


def plot_crossmask(cross: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Performance on the mask seen in training vs unseen mask realizations.

    Quantifies how much of the learned reconstruction is tied to one sampling operator.
    The classical baseline is included as a control: having no learned parameters, its
    variation across masks reflects only how hard each realization is.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if cross.empty:
        warnings.warn("empty crossmask frame")
        return ax.figure

    methods = list(cross["method"].unique())
    width = 0.8 / max(len(methods), 1)
    positions = np.arange(2)
    for i, method in enumerate(methods):
        sub = cross[cross["method"] == method]
        means, errs = [], []
        for seen in (True, False):
            values = sub[sub["mask_seen_in_training"] == seen]["psnr_real"]
            means.append(values.mean() if len(values) else np.nan)
            errs.append(values.std() if len(values) > 1 else 0.0)
        ax.bar(positions + i * width - 0.4 + width / 2, means, width, yerr=errs,
               capsize=4, label=method, edgecolor="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(["mask seen in training", "unseen mask realizations"])
    ax.set_ylabel("PSNR, real channel (dB)")
    ax.set_title("Specificity of the reconstruction to one undersampling pattern")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax.figure


# ---------------------------------------------------------------------------
# 5. Qualitative examples (Input / Baseline / Model / Ground truth)
# ---------------------------------------------------------------------------


def _magnitude(chan_img: torch.Tensor) -> np.ndarray:
    """Magnitude image in viewing orientation, from a 2-channel tensor ``(1, 2, H, W)``."""
    x = chan_img.detach().cpu()
    return to_display(torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2).numpy())


def _build_eval_context(checkpoint_path: str, baseline: str, device: Optional[str] = None):
    """Load the trained model + a (parameter-free) baseline + the test dataset + mask."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    base_model = MODEL_REGISTRY.build(baseline).to(device).eval()

    datasets = build_datasets(cfg)
    size = cfg["data"]["image_size"]
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    mask_params.setdefault("seed", cfg["train"]["seed"])
    mask, _ = build_mask(cfg["mask"]["name"], (size, size), device, **mask_params)
    return cfg, model, base_model, datasets["test"], mask, device


@torch.no_grad()
def qualitative_examples(checkpoint_path: str, baseline: str = "classical_cs_tv",
                         base: str = "psnr", device: Optional[str] = None) -> plt.Figure:
    """Render the four required example categories in a 4x4 (row x column) panel.

    Rows: both-good, both-poor, baseline-wins, model-wins. Columns: zero-filled input,
    baseline, model, ground truth (all shown as magnitude images). Titles carry the
    per-sample metric so the categories are self-evident.
    """
    from .metrics import psnr_channel, ssim_channel

    cfg, model, base_model, testset, mask, device = _build_eval_context(
        checkpoint_path, baseline, device)

    def _metric(recon, label):
        if base == "psnr":
            return float((psnr_channel(recon[:, 0], label[:, 0]) +
                          psnr_channel(recon[:, 1], label[:, 1])).item() / 2)
        return float((ssim_channel(recon[:, 0], label[:, 0]) +
                      ssim_channel(recon[:, 1], label[:, 1])).item() / 2)

    records = []
    for i in range(len(testset)):
        label = testset[i].unsqueeze(0).to(device)
        y = undersample(label, mask)
        rec_m = model(y, mask).clamp(-1, 1)
        rec_b = base_model(y, mask).clamp(-1, 1)
        records.append({"i": i, "model": _metric(rec_m, label), "baseline": _metric(rec_b, label)})
    stats = pd.DataFrame(records)
    if stats.empty:
        raise RuntimeError("empty test set; cannot build qualitative examples")

    stats["gap"] = stats["model"] - stats["baseline"]

    picks = {
        "both good": stats.sort_values(["model", "baseline"], ascending=False).iloc[0]["i"],
        "both poor": stats.assign(s=stats["model"] + stats["baseline"]).sort_values("s").iloc[0]["i"],
        "baseline wins": stats.sort_values("gap").iloc[0]["i"],
        "model wins": stats.sort_values("gap", ascending=False).iloc[0]["i"],
    }

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    col_titles = ["Zero-filled input", f"Baseline ({baseline})", "Our model (ADMM-Net)", "Ground truth"]
    for row, (cat, idx) in enumerate(picks.items()):
        idx = int(idx)
        label = testset[idx].unsqueeze(0).to(device)
        y = undersample(label, mask)
        zf = zero_filled_recon(y).clamp(-1, 1)
        rec_b = base_model(y, mask).clamp(-1, 1)
        rec_m = model(y, mask).clamp(-1, 1)
        imgs = [_magnitude(zf), _magnitude(rec_b), _magnitude(rec_m), _magnitude(label)]
        mb, mm = _metric(rec_b, label), _metric(rec_m, label)
        for col, im in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
        axes[row, 0].set_ylabel(cat, fontsize=12, rotation=90)
        axes[row, 0].axis("on"); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 1].set_title(f"{base.upper()} {mb:.2f}", fontsize=10)
        axes[row, 2].set_title(f"{base.upper()} {mm:.2f}", fontsize=10)
    fig.suptitle("Qualitative reconstructions by category (magnitude images)", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def list_checkpoints(results_dir: str) -> Dict[str, str]:
    """Map run_id -> best.pth path for every run dir that saved a checkpoint."""
    out = {}
    for entry in sorted(os.listdir(results_dir)):
        ckpt = os.path.join(results_dir, entry, "best.pth")
        if os.path.exists(ckpt):
            out[entry] = ckpt
    return out
