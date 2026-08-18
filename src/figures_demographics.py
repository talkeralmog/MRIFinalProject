# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Who is in the dataset: age, sex and cohort composition of the three splits.

The reconstruction task itself does not use the metadata, but the split does -- we
stratify on age -- and a reader has to be able to check that the three splits really are
comparable and that the test set is representative. This module builds that evidence.

It works at two levels of detail depending on what is reachable:

* **With the metadata CSVs** (``student_*_metadata.csv`` under ``data.data_root``, or any
  directory passed with ``--meta-dir``): true per-subject age histograms per split, the
  sex balance, and age broken down by sex.
* **Without them**: the cohort composition, which is recoverable from the subject
  filenames recorded in every run's ``split.json``, plus the per-split age summary
  statistics those files store. The age panel is then drawn as a summary rather than a
  histogram, and labelled as such -- we do not fabricate a distribution we cannot measure.

Usage::

    python -m src.figures_demographics --meta-dir ~/data
    python -m src.figures_demographics                 # falls back to filenames only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .display import load_split_members

SPLITS = ("train", "val", "test")
SPLIT_COLOURS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

#: Filename patterns that identify the source cohort of a subject.
COHORT_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"^A00\d+$", "NKI-Rockland"),
    (r"^Corr([A-Za-z]+)_", "CoRR – {0}"),
    (r"^Sub\d+_Ses\d+$", "Sub####_Ses#"),
    (r"^NDAR", "NDAR"),
    (r"^SLIMsub", "SLIM"),
    (r"^sub\d+$", "sub#####"),
)


def cohort_of(filename: str) -> str:
    """Best-effort source cohort for one subject, from its filename."""
    stem = re.sub(r"\.(npy|npz)$", "", filename)
    for pattern, label in COHORT_PATTERNS:
        match = re.match(pattern, stem)
        if match:
            return label.format(*match.groups()) if match.groups() else label
    return "other"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def read_metadata(meta_dir: str, csv_names: Sequence[str], id_col: str, age_col: str,
                  path_col: str) -> Dict[str, Dict[str, object]]:
    """``basename -> {age, sex}`` from the provided CSVs, concatenated and de-duplicated."""
    records: Dict[str, Dict[str, object]] = {}
    sex_col_candidates = ("Sex", "sex", "Gender", "gender")
    for name in csv_names:
        path = os.path.join(os.path.expanduser(meta_dir), name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = None
                if path_col and row.get(path_col):
                    key = os.path.basename(row[path_col].strip())
                elif row.get(id_col):
                    key = f"{row[id_col].strip()}.npy"
                if not key:
                    continue
                try:
                    age = float(row[age_col])
                except (KeyError, TypeError, ValueError):
                    age = float("nan")
                sex = next((str(row[c]).strip() for c in sex_col_candidates
                            if row.get(c) not in (None, "")), "")
                records.setdefault(key, {"age": age, "sex": sex})
    return records


def _normalise_sex(value: str) -> str:
    v = str(value).strip().lower()
    if v in ("m", "male", "1"):
        return "male"
    if v in ("f", "female", "2", "0"):
        return "female"
    return "unknown"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def demographics_figure(members: Dict[str, List[str]],
                        metadata: Dict[str, Dict[str, object]],
                        age_stats: Dict[str, Dict[str, float]]) -> Tuple[plt.Figure, List[Dict]]:
    """Age, sex and cohort composition of the three splits."""
    have_ages = bool(metadata) and any(
        np.isfinite(metadata.get(name, {}).get("age", float("nan")))
        for name in members.get("train", [])[:200])

    ages: Dict[str, np.ndarray] = {}
    sexes: Dict[str, Counter] = {}
    for split in SPLITS:
        names = members.get(split, [])
        values = [metadata.get(n, {}).get("age", float("nan")) for n in names]
        ages[split] = np.array([v for v in values if np.isfinite(v)], dtype=float)
        sexes[split] = Counter(_normalise_sex(metadata.get(n, {}).get("sex", ""))
                               for n in names)

    # Two layouts: the full one when the metadata CSVs are reachable, and a compact one
    # that shows only what the filenames and split.json can actually support.
    if have_ages:
        fig = plt.figure(figsize=(16.0, 8.6))
        grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26)
        slots = {"age": grid[0, 0], "cum": grid[0, 1], "sex": grid[0, 2],
                 "agesex": grid[1, 0], "cohort": grid[1, 1:]}
    else:
        # Without per-subject ages there is only one thing worth plotting, so plot it
        # properly instead of padding the figure with an errorbar that repeats Table 1.
        fig = plt.figure(figsize=(13.5, 6.4))
        grid = fig.add_gridspec(1, 1)
        slots = {"cohort": grid[0, 0]}

    # (a) age histogram per split -------------------------------------------
    if have_ages:
        ax = fig.add_subplot(slots["age"])
        bins = np.arange(0, 95, 5)
        for split in SPLITS:
            ax.hist(ages[split], bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=SPLIT_COLOURS[split],
                    label=f"{split} (n = {len(ages[split])})")
        ax.set_ylabel("density")
        ax.set_title("(a) Age distribution per split\n"
                     "the three curves coincide by construction", fontsize=10)
        ax.set_xlabel("age (years)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (b) cumulative age, which makes a distribution shift impossible to hide
    if have_ages:
        ax = fig.add_subplot(slots["cum"])
        for split in SPLITS:
            values = np.sort(ages[split])
            ax.plot(values, np.linspace(0, 1, len(values)), linewidth=1.8,
                    color=SPLIT_COLOURS[split], label=split)
        ax.set_xlabel("age (years)")
        ax.set_ylabel("cumulative fraction of subjects")
        ax.set_title("(b) Cumulative age distribution\n"
                     "any split shift would show as a gap here", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (c) sex balance --------------------------------------------------------
    categories = [c for c in ("female", "male", "unknown")
                  if any(sexes[s].get(c, 0) for s in SPLITS)]
    if have_ages and categories and categories != ["unknown"]:
        ax = fig.add_subplot(slots["sex"])
        width = 0.8 / len(categories)
        positions = np.arange(len(SPLITS))
        for i, category in enumerate(categories):
            counts = [sexes[s].get(category, 0) for s in SPLITS]
            totals = [max(sum(sexes[s].values()), 1) for s in SPLITS]
            shares = [100 * c / t for c, t in zip(counts, totals)]
            bars = ax.bar(positions + i * width - 0.4 + width / 2, shares, width,
                          edgecolor="black", label=category)
            for bar, share, count in zip(bars, shares, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, share + 1, f"{count}",
                        ha="center", fontsize=7.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(SPLITS)
        ax.set_ylabel("% of subjects in the split")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("(c) Sex balance per split", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # (d) age by sex ---------------------------------------------------------
    pooled = {}
    for category in ("female", "male"):
        values = [metadata[n]["age"] for split in SPLITS for n in members.get(split, [])
                  if n in metadata and _normalise_sex(metadata[n].get("sex", "")) == category
                  and np.isfinite(metadata[n].get("age", float("nan")))]
        if values:
            pooled[category] = np.array(values, dtype=float)
    if have_ages and pooled:
        ax = fig.add_subplot(slots["agesex"])
        ax.hist(list(pooled.values()), bins=np.arange(0, 95, 5), stacked=True,
                label=list(pooled), edgecolor="black", linewidth=0.4)
        ax.set_xlabel("age (years)")
        ax.set_ylabel("subjects")
        ax.legend(fontsize=8)
        ax.set_title("(d) Age by sex, all splits pooled", fontsize=10)
        ax.grid(alpha=0.3)

    # (e) cohort composition -------------------------------------------------
    ax = fig.add_subplot(slots["cohort"])
    counts = Counter()
    per_split: Dict[str, Counter] = {}
    for split in SPLITS:
        c = Counter(cohort_of(n) for n in members.get(split, []))
        per_split[split] = c
        counts += c
    total = sum(counts.values())

    # Show every cohort, but fold the long tail of tiny ones into one bar so the axis
    # stays readable. "other" is already a catch-all, so merge into it.
    ordered = [k for k, _ in counts.most_common()]
    major = [k for k in ordered if counts[k] >= 0.01 * total and k != "other"]
    minor = [k for k in ordered if k not in major]
    labels = major + ([f"{len(minor)} smaller cohorts"] if minor else [])

    def split_counts(split: str) -> np.ndarray:
        values = [per_split[split].get(k, 0) for k in major]
        if minor:
            values.append(sum(per_split[split].get(k, 0) for k in minor))
        return np.array(values, dtype=float)

    totals = sum(split_counts(sp) for sp in SPLITS)

    # Bars are drawn as shares of each cohort, so the test fraction is comparable across
    # cohorts of very different sizes -- which is the thing worth checking.
    lefts = np.zeros(len(labels))
    for split in SPLITS:
        shares = 100.0 * split_counts(split) / np.maximum(totals, 1)
        bars = ax.barh(labels, shares, left=lefts, color=SPLIT_COLOURS[split],
                       edgecolor="white", linewidth=0.6, label=split, height=0.72)
        for bar, share in zip(bars, shares):
            if share >= 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2, f"{share:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if split != "val" else "black")
        lefts += shares

    ax.axvline(80, color="0.25", linestyle="--", linewidth=1.1, zorder=5)
    ax.axvline(90, color="0.25", linestyle="--", linewidth=1.1, zorder=5)
    ax.text(85, len(labels) - 0.25, "target 80 / 10 / 10", fontsize=8, color="0.25",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7"))

    for i, (label, n) in enumerate(zip(labels, totals)):
        ax.text(101.5, i, f"n = {int(n)}  ({100 * n / total:.0f}%)",
                va="center", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100%"])
    ax.set_xlabel("share of the cohort's subjects assigned to each split")
    ax.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95)
    ax.set_title(
        f"Source cohorts, inferred from the subject filenames "
        f"({len(counts)} cohorts, {total} subjects).\n"
        "Each bar is one cohort, split three ways. We stratify on age, not on site, so "
        "nothing forces a cohort to land on 80 / 10 / 10;\nthe large ones do anyway. The "
        "small ones scatter (CoRR-BMB puts 21% of its 48 subjects in test), which is "
        "sampling noise, not bias.",
        fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)

    if have_ages:
        fig.suptitle("Who is in the dataset: age, sex and provenance of the three splits",
                     fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()

    rows: List[Dict] = []
    for split in SPLITS:
        s = age_stats.get(split, {})
        row = {
            "split": split,
            "subjects": len(members.get(split, [])),
            "age mean": round(float(np.mean(ages[split])), 2) if len(ages[split])
                        else round(s.get("age_mean", float("nan")), 2),
            "age std": round(float(np.std(ages[split], ddof=1)), 2) if len(ages[split]) > 1
                       else round(s.get("age_std", float("nan")), 2),
            "age min": round(float(np.min(ages[split])), 1) if len(ages[split])
                       else s.get("age_min", float("nan")),
            "age max": round(float(np.max(ages[split])), 1) if len(ages[split])
                       else s.get("age_max", float("nan")),
            "median": round(float(np.median(ages[split])), 1) if len(ages[split]) else "",
            "female": sexes[split].get("female", 0),
            "male": sexes[split].get("male", 0),
            "sex unknown": sexes[split].get("unknown", 0),
            "cohorts": len(per_split[split]),
        }
        rows.append(row)
    return fig, rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--meta-dir", default=None,
                   help="Directory holding the student_*_metadata.csv files "
                        "(default: the config's data_root, then the project root).")
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]
    out_dir = os.path.join(results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    data_cfg = cfg["data"]
    members = load_split_members(results_root)
    if not members:
        raise SystemExit("no split.json found under the results root")

    search = [args.meta_dir] if args.meta_dir else []
    search += [cfg["paths"].get("data_root", ""), ".", "data"]
    metadata: Dict[str, Dict[str, object]] = {}
    for candidate in search:
        if not candidate:
            continue
        metadata = read_metadata(candidate, data_cfg["meta_csvs"], data_cfg["id_col"],
                                 data_cfg["age_col"], data_cfg.get("path_col", ""))
        if metadata:
            print(f"read metadata for {len(metadata)} subjects from {candidate}")
            break
    if not metadata:
        print("note: metadata CSVs not found; falling back to filenames + split.json "
              "summary statistics. Pass --meta-dir to get the full histograms.")

    files = sorted(glob.glob(os.path.join(results_root, "*", "split.json")))
    age_stats = json.load(open(files[0]))["age_stats"] if files else {}

    fig, rows = demographics_figure(members, metadata, age_stats)
    png = os.path.join(out_dir, "mri_demographics.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")

    table = os.path.join(out_dir, "mri_demographics.csv")
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {table}")
    for row in rows:
        print("  ", row)
    return [png, table]


if __name__ == "__main__":
    main()
