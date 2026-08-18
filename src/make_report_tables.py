# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Render the report's tables from the logged results, as Markdown.

Keeping the numbers in the report in sync with ``results/runs.csv`` by hand is how reports
end up quoting stale figures. This script regenerates every table the report needs and
writes them to ``report/tables.md``, ready to paste (or include) in the report:

1. The headline test-set table: PSNR and SSIM on the real and imaginary channels, one row
   per sampling ratio for the baseline and for our model, as mean +/- std across seeds.
2. The same PSNR values restated with ``peak = 1.0`` -- the convention used in the
   literature -- so the absolute numbers can be compared with published work.
3. The per-split age statistics of our custom age-stratified split.
4. The appendix ablations: unrolling depth, training loss, architecture.
5. The baseline calibration sweep (selected on the validation split).
6. The U-Net reference and the unseen-mask generalization check, when those runs exist.

Usage::

    python -m src.make_report_tables --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import pandas as pd

from .analysis import (
    combined_metric,
    comparable_runs,
    load_results,
    load_samples,
    select_samples,
)
from .config import load_config
from .metrics import DATA_RANGE, convert_psnr

BASELINE = "classical_cs_tv"
NAIVE_BASELINE = "classical_cs"
MODEL = "admmnet_softthresh"
ZERO_FILLED = "zero_filled"
METRICS = ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag")

PRETTY = {
    ZERO_FILLED: "Zero-filled (model input)",
    NAIVE_BASELINE: "Naive CS (single-level wavelet)",
    BASELINE: "Baseline: CS (wavelet + TV)",
    MODEL: "Our model: ADMM-Net",
    "admmnet_pwl": "ADMM-Net (piecewise-linear)",
    "unet": "U-Net (no data consistency)",
}


def _pretty(method: str) -> str:
    return PRETTY.get(method, method)


def _cell(value) -> str:
    """Render one cell, keeping floats short and missing values blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` so the project does not
    depend on ``tabulate``, which is not installed on the HPC environment.
    """
    columns = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in record] for record in df.itertuples(index=False)]
    widths = [max(len(columns[i]), *(len(r[i]) for r in rows)) if rows else len(columns[i])
              for i in range(len(columns))]

    def line(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [line(columns), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def headline_table(samples: pd.DataFrame, methods: List[str],
                   split: str = "test") -> pd.DataFrame:
    """PSNR/SSIM per channel, mean +/- std **across the test set**, per ratio and method.

    The brief asks for the spread over the test set, so this aggregates the per-slice
    metrics; the three seeds are pooled, giving n_seeds x n_test_slices measurements
    per cell.
    """
    d = select_samples(samples, split, experiment="comparison")
    d = d[d["method"].isin(methods)]
    if d.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for metric in METRICS:
        if metric not in d.columns:
            continue
        digits = 4 if metric.startswith("ssim") else 2
        agg = d.groupby(["sampling_ratio", "method"])[metric].agg(["mean", "std"])
        for (ratio, method), r in agg.iterrows():
            rows.append({
                "Sampling ratio": f"{int(round(ratio * 100))}%",
                "Method": _pretty(method),
                "metric": metric,
                "cell": f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}",
            })
    if not rows:
        return pd.DataFrame()
    tidy = pd.DataFrame(rows)
    table = tidy.pivot_table(index=["Sampling ratio", "Method"], columns="metric",
                             values="cell", aggfunc="first")
    table = table.reindex(columns=[m for m in METRICS if m in table.columns])
    table.columns = ["PSNR real (dB)", "PSNR imag (dB)", "SSIM real", "SSIM imag"]
    order = {_pretty(m): i for i, m in enumerate(methods)}
    table = table.reset_index()
    table["_order"] = table["Method"].map(order)
    table = table.sort_values(["Sampling ratio", "_order"]).drop(columns="_order")
    return table


def _comparison_runs(df: pd.DataFrame, methods: List[str], split: str) -> pd.DataFrame:
    """One run average per (method, sampling_ratio, seed) in the headline comparison.

    The zero-filled reference is logged by every run, so without de-duplication it would
    appear once per trained model and be counted twice.
    """
    d = combined_metric(df[df["name"] == "comparison"], "psnr", split)
    if d.empty:
        return d
    d = d[d["method"].isin(methods)]
    return d.drop_duplicates(subset=["method", "sampling_ratio", "seed"])


def seed_spread_table(df: pd.DataFrame, methods: List[str],
                      split: str = "test") -> pd.DataFrame:
    """How much the run average moves when only the mask realization changes.

    Reported separately from the headline table so the two sources of variability are not
    confused: the headline std is slice-to-slice, this one is across the three seeds.
    """
    d = _comparison_runs(df, methods, split)
    if d.empty:
        return pd.DataFrame()
    agg = d.groupby(["sampling_ratio", "method"])["value"].agg(["mean", "std", "count"])
    agg = agg.reset_index()
    agg["Sampling ratio"] = (agg["sampling_ratio"] * 100).round().astype(int).astype(str) + "%"
    agg["Method"] = agg["method"].map(_pretty)
    agg["Mean PSNR (dB)"] = agg["mean"].round(2)
    agg["Std across seeds (dB)"] = agg["std"].fillna(0.0).round(2)
    agg["# seeds"] = agg["count"]
    order = {_pretty(m): i for i, m in enumerate(methods)}
    agg["_order"] = agg["Method"].map(order)
    return (agg.sort_values(["Sampling ratio", "_order"])
            [["Sampling ratio", "Method", "Mean PSNR (dB)",
              "Std across seeds (dB)", "# seeds"]])


def psnr_convention_table(df: pd.DataFrame, methods: List[str],
                          split: str = "test") -> pd.DataFrame:
    """PSNR under both peak conventions, averaged over the real and imaginary channels."""
    d = _comparison_runs(df, methods, split)
    if d.empty:
        return pd.DataFrame()
    agg = d.groupby(["sampling_ratio", "method"])["value"].mean().reset_index()
    agg["Sampling ratio"] = (agg["sampling_ratio"] * 100).round().astype(int).astype(str) + "%"
    agg["Method"] = agg["method"].map(_pretty)
    agg[f"PSNR, peak={DATA_RANGE:.1f} (as logged)"] = agg["value"].round(2)
    agg["PSNR, peak=1.0 (literature convention)"] = agg["value"].map(
        lambda v: round(convert_psnr(v, DATA_RANGE, 1.0), 2))
    order = {_pretty(m): i for i, m in enumerate(methods)}
    agg["_order"] = agg["Method"].map(order)
    return (agg.sort_values(["Sampling ratio", "_order"])
            .drop(columns=["sampling_ratio", "method", "value", "_order"]))


def split_table(results_root: str) -> pd.DataFrame:
    """Per-split subject counts and age statistics, read from any run's split.json."""
    files = sorted(glob.glob(os.path.join(results_root, "*", "split.json")))
    if not files:
        return pd.DataFrame()
    with open(files[0]) as f:
        stats = json.load(f)["age_stats"]
    rows = []
    for name in ("train", "val", "test"):
        if name not in stats:
            continue
        s = stats[name]
        rows.append({
            "Split": name,
            "# subjects": s["n_subjects"],
            "Age mean": round(s["age_mean"], 1),
            "Age std": round(s["age_std"], 1),
            "Age range": f"{s['age_min']:.0f}-{s['age_max']:.0f}",
        })
    return pd.DataFrame(rows)


def ablation_table(df: pd.DataFrame, experiment: str, label_cols: List[str],
                   label_name: str, split: str = "test") -> pd.DataFrame:
    """PSNR and SSIM (averaged over real/imag) for one categorical/numeric ablation."""
    sub = comparable_runs(df, experiment)
    if sub.empty:
        return pd.DataFrame()
    out: Optional[pd.DataFrame] = None
    for base, column in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
        d = combined_metric(sub, base, split)
        if d.empty:
            continue
        d = d[d["method"] != ZERO_FILLED]
        # Only pull in label columns that combined_metric did not already provide,
        # otherwise the merge would rename them with _x / _y suffixes.
        missing = [c for c in label_cols if c in sub.columns and c not in d.columns]
        if missing:
            d = d.merge(sub[["run_id"] + missing].drop_duplicates(), on="run_id", how="left")
        available = [c for c in label_cols if c in d.columns]
        d[label_name] = d[available].astype(str).agg(", ".join, axis=1)
        agg = d.groupby(label_name)["value"].mean().round(3).reset_index(name=column)
        out = agg if out is None else out.merge(agg, on=label_name, how="outer")
    return out if out is not None else pd.DataFrame()


def baseline_tuning_table(df: pd.DataFrame) -> pd.DataFrame:
    """The baseline calibration sweep, ranked on the VALIDATION split."""
    sub = df[df["name"] == "baseline_tuning"]
    if sub.empty:
        return pd.DataFrame()
    d = combined_metric(sub, "psnr", "val")
    if d.empty:
        return pd.DataFrame()
    label_cols = [c for c in ("lam", "tv_weight") if c in sub.columns]
    if label_cols:
        extra = sub[["run_id"] + label_cols].drop_duplicates()
        d = d.merge(extra, on="run_id", how="left")
    d = d[d["method"] != ZERO_FILLED]
    d = d.rename(columns={"value": "Validation PSNR (dB)"})
    keep = label_cols + ["Validation PSNR (dB)"]
    return (d[keep].round(3)
            .sort_values("Validation PSNR (dB)", ascending=False)
            .reset_index(drop=True))


def crossmask_table(results_root: str) -> pd.DataFrame:
    """Trained-on mask vs unseen mask realizations, per method."""
    path = os.path.join(results_root, "crossmask.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    cross = pd.read_csv(path)
    rows = []
    for method, group in cross.groupby("method"):
        seen = group[group["mask_seen_in_training"]]["psnr_real"].mean()
        unseen = group[~group["mask_seen_in_training"]]["psnr_real"].mean()
        rows.append({
            "Method": _pretty(method),
            "Trained-on mask (dB)": round(seen, 2),
            "Unseen masks (dB)": round(unseen, 2),
            "Gap (dB)": round(seen - unseen, 2),
        })
    return pd.DataFrame(rows)


def build_document(cfg: Dict) -> str:
    """Assemble every table into one Markdown document."""
    results_root = cfg["paths"]["results_root"]
    df = load_results(results_root)
    samples = load_samples(results_root)
    methods = [ZERO_FILLED, NAIVE_BASELINE, BASELINE, MODEL]
    present = [m for m in methods if m in set(df["method"].unique())]

    parts: List[str] = [
        "# Report tables",
        "",
        "Generated by `python -m src.make_report_tables` from `results/runs.csv`.",
        "Do not edit by hand: re-run the script instead.",
        "",
    ]

    def section(title: str, table: pd.DataFrame, note: str = "") -> None:
        parts.append(f"## {title}\n")
        if note:
            parts.append(note + "\n")
        if table is None or table.empty:
            parts.append("_No runs logged for this table yet._\n")
        else:
            parts.append(_markdown(table) + "\n")

    section("1. Test-set results by sampling ratio", headline_table(samples, present),
            "Mean ± one standard deviation **across the test set**, as the brief requires, "
            "pooling the three seeds. The seeds vary the k-space mask realization and the "
            "network initialization; the data split is identical in every run.")

    section("1b. Variability across mask realizations", seed_spread_table(df, present),
            "The same runs summarized differently: how much the test-set average moves "
            "when only the mask realization (and the network initialization) changes.")

    section("2. The same PSNR under both peak conventions",
            psnr_convention_table(df, present),
            f"Logged with `peak = {DATA_RANGE:.1f}` (the width of the [-1, 1] channel "
            "range). The right-hand column restates it with `peak = 1.0`, the maximum "
            "magnitude of the normalized image, which is the convention used in the "
            "reconstruction literature. The shift is a constant 6.02 dB and affects all "
            "methods equally.")

    section("3. Custom age-stratified split", split_table(results_root),
            "Built by cross-referencing the three provided CSVs against the NumPy volumes "
            "actually present on disk, then splitting within quantile age bins.")

    section("4. Baseline calibration (validation split)", baseline_tuning_table(df),
            "The classical baseline has two free parameters. They are selected here on the "
            "validation split only; the test set is never used for this choice.")

    section("5. Appendix: unrolling depth",
            ablation_table(df, "depth_sweep", ["num_stages"], "ADMM stages"))
    section("6. Appendix: training loss",
            ablation_table(df, "loss_ablation", ["loss"], "Loss"))
    section("7. Appendix: architecture",
            ablation_table(df, "structure_ablation", ["method", "share_weights"],
                          "Nonlinearity, weight sharing"))
    section("8. Appendix: U-Net reference",
            ablation_table(df, "unet_reference", ["sampling_ratio"], "Sampling ratio"),
            "A generic CNN with no data-consistency step (467,554 parameters against "
            "ADMM-Net's 317,320).")
    section("9. Appendix: generalization to unseen undersampling masks",
            crossmask_table(results_root),
            "Each trained network re-evaluated, without retraining, under mask "
            "realizations drawn with different seeds.")

    return "\n".join(parts)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the report's tables as Markdown.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--out", default=os.path.join("report", "tables.md"))
    return p.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    document = build_document(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(document)
    print(document)
    print(f"\nwrote {args.out}")
    return document


if __name__ == "__main__":
    main()
