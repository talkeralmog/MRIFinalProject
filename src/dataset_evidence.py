# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Collect the dataset-side evidence the report cannot produce from the slice cache.

Everything else in this project runs from ``cache/<fingerprint>/*.npz``, which holds one
central slice per subject and no metadata. Four things the report wants are therefore only
obtainable on the machine that has the dataset directory:

``demographics``
    The per-subject age and sex distributions of the three splits. Without the metadata
    CSVs the report can only show cohort composition and the summary statistics that
    ``split.json`` happens to record, which is circular because our own splitter wrote them.

``audit``
    Where the subjects go. The report states that 5,242 metadata rows reduce to 4,791
    split members; the two intermediate losses (no matching volume on disk, then no
    parseable age) are inferred from the run logs and from reading the splitter, not
    measured. This measures them.

``acquisition``
    Whether the CSVs carry sequence parameters (TR, TE, field strength, manufacturer,
    site). If they do, the report can *confirm* the T1-weighting it currently infers from
    image contrast, and test whether the low-contrast failure cases cluster by scanner.

``slices``
    The through-plane signal profile of a few volumes, which is the evidence for the
    one-central-slice rule: it shows where the chosen slice sits and how much anatomy the
    choice discards.

Only small derived artefacts are written (CSV, JSON, PNG). No image data is copied, so the
outputs are safe to bring back to a laptop and safe to commit.

Run this on the server, from the project root::

    python -m src.dataset_evidence --data-root ~/MRI_2026_datasets/brain_age
    python -m src.dataset_evidence --which audit demographics
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SPLITS = ("train", "val", "test")
SPLIT_COLOURS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

#: Column names to look for, in order of preference.
AGE_COLUMNS = ("Age", "age", "AGE", "age_years")
SEX_COLUMNS = ("Sex", "sex", "Gender", "gender", "SEX")
ID_COLUMNS = ("Subject", "subject", "subject_id", "SubjectID", "ID", "id")
PATH_COLUMNS = ("filePath", "filepath", "path", "file", "Path")
#: Acquisition parameters worth reporting if the CSVs happen to carry them.
ACQUISITION_HINTS = ("tr", "te", "ti", "flip", "field", "tesla", "manufacturer",
                     "scanner", "site", "model", "sequence", "voxel", "thickness",
                     "resolution", "matrix", "study", "cohort", "dataset")


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def find_data_root(explicit: Optional[str]) -> str:
    """Locate the directory holding the metadata CSVs and the NumPy volumes."""
    candidates = [explicit] if explicit else []
    candidates += ["~/data", "~/MRI_2026_datasets/brain_age",
                   "../MRI_2026_datasets/brain_age", "./data"]
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(candidate)
        if glob.glob(os.path.join(path, "*metadata*.csv")):
            return path
    tried = ", ".join(str(c) for c in candidates if c)
    raise SystemExit(
        f"could not find the metadata CSVs. Tried: {tried}\n"
        "Pass the directory explicitly, e.g.\n"
        "  python -m src.dataset_evidence --data-root ~/MRI_2026_datasets/brain_age")


def _pick(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if row.get(name) not in (None, ""):
            return name
    return None


def read_metadata(data_root: str) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
    """Every metadata row, the CSV column names, and the resolved key columns."""
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    for path in sorted(glob.glob(os.path.join(data_root, "*metadata*.csv"))):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or list(reader.fieldnames or [])
            for row in reader:
                row["__source_csv__"] = os.path.basename(path)
                rows.append(row)
    if not rows:
        raise SystemExit(f"no *metadata*.csv rows found under {data_root}")

    sample = rows[0]
    columns = {
        "id": _pick(sample, ID_COLUMNS) or (fieldnames[0] if fieldnames else "Subject"),
        "age": _pick(sample, AGE_COLUMNS),
        "sex": _pick(sample, SEX_COLUMNS),
        "path": _pick(sample, PATH_COLUMNS),
    }
    return rows, fieldnames, columns


def parse_age(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def normalise_sex(value: object) -> str:
    v = str(value).strip().lower()
    if v in ("m", "male", "1"):
        return "male"
    if v in ("f", "female", "2", "0"):
        return "female"
    return "unknown"


def discover_volumes(data_root: str, numpy_dir: str = "selected_npy") -> Dict[str, int]:
    """Basename -> size in bytes for every volume on disk (0-byte files included)."""
    root = os.path.join(data_root, numpy_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"NumPy directory not found: {root}")
    out: Dict[str, int] = {}
    for name in sorted(os.listdir(root)):
        if name.endswith((".npy", ".npz")):
            out[name] = os.path.getsize(os.path.join(root, name))
    return out


def load_split_members(results_root: str = "results") -> Dict[str, List[str]]:
    """The per-split subject filename lists any run recorded in ``split.json``."""
    files = sorted(glob.glob(os.path.join(results_root, "*", "split.json")))
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f).get("subjects", {})


# ---------------------------------------------------------------------------
# 1. The subject accounting
# ---------------------------------------------------------------------------


def audit(data_root: str, out_dir: str, results_root: str) -> Dict[str, object]:
    """Measure every step from metadata rows to split members."""
    rows, fieldnames, columns = read_metadata(data_root)
    volumes = discover_volumes(data_root)
    members = load_split_members(results_root)
    in_splits = {name for names in members.values() for name in names}

    unique_ids = {row.get(columns["id"], "") for row in rows}
    deduped: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    for row in rows:
        deduped.setdefault(row.get(columns["id"], ""), row)

    def resolve(row: Dict[str, str]) -> Optional[str]:
        """Which volume on disk this row refers to, matching the project's own rule."""
        if columns["path"] and row.get(columns["path"]):
            base = os.path.basename(str(row[columns["path"]]).strip())
            if base in volumes:
                return base
            stem = re.sub(r"\.(npy|npz)$", "", base)
            for name in volumes:
                if stem and stem in name:
                    return name
        sid = str(row.get(columns["id"], "")).strip()
        if f"{sid}.npy" in volumes:
            return f"{sid}.npy"
        for name in volumes:
            if sid and sid in name:
                return name
        return None

    matched, unmatched, empty_file = [], [], []
    for row in deduped.values():
        name = resolve(row)
        if name is None:
            unmatched.append(row)
        elif volumes[name] == 0:
            empty_file.append(name)
        else:
            matched.append((name, row))

    ages = {name: parse_age(row.get(columns["age"])) for name, row in matched}
    no_age = [name for name, value in ages.items() if not np.isfinite(value)]
    usable = [name for name in ages if np.isfinite(ages[name])]

    report = {
        "data_root": data_root,
        "csv_columns": fieldnames,
        "resolved_columns": columns,
        "metadata_rows_total": len(rows),
        "unique_subject_ids": len(unique_ids),
        "rows_after_dedup": len(deduped),
        "volumes_on_disk": len(volumes),
        "volumes_zero_byte": sum(1 for size in volumes.values() if size == 0),
        "matched_to_a_volume": len(matched),
        "no_matching_volume": len(unmatched),
        "matched_but_zero_byte": len(empty_file),
        "matched_without_parseable_age": len(no_age),
        "matched_with_usable_age": len(usable),
        "subjects_recorded_in_splits": len(in_splits),
        "split_sizes": {k: len(v) for k, v in members.items()},
    }
    if in_splits:
        report["in_splits_but_no_age"] = len(in_splits & set(no_age))
        report["usable_but_not_in_any_split"] = len(set(usable) - in_splits)

    path = os.path.join(out_dir, "dataset_audit.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {path}")

    print("\n--- subject accounting ---")
    for key in ("metadata_rows_total", "rows_after_dedup", "no_matching_volume",
                "matched_to_a_volume", "matched_without_parseable_age",
                "matched_with_usable_age", "subjects_recorded_in_splits"):
        print(f"  {key:34s} {report[key]}")
    if "in_splits_but_no_age" in report:
        print(f"  {'in_splits_but_no_age':34s} {report['in_splits_but_no_age']}")
    print("\nThis is the chain the report's section 1.5 describes. If "
          "`matched_with_usable_age` equals `subjects_recorded_in_splits`, the missing-age "
          "explanation is confirmed; if not, something else drops subjects too.")

    if no_age:
        path = os.path.join(out_dir, "subjects_without_age.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["volume", "in_a_split"])
            for name in sorted(no_age):
                writer.writerow([name, name in in_splits])
        print(f"wrote {path}  ({len(no_age)} subjects)")
    return report


# ---------------------------------------------------------------------------
# 2. Acquisition parameters, if the CSVs carry any
# ---------------------------------------------------------------------------


def acquisition(data_root: str, out_dir: str) -> List[Dict[str, object]]:
    """Report any sequence/scanner columns the metadata happens to include."""
    rows, fieldnames, _ = read_metadata(data_root)
    interesting = [c for c in fieldnames
                   if any(hint in c.lower() for hint in ACQUISITION_HINTS)]

    print("\n--- CSV columns ---")
    print("  all:", ", ".join(fieldnames) or "(none)")
    print("  acquisition-related:", ", ".join(interesting) or "(none found)")
    if not interesting:
        print("\nNo sequence parameters in the metadata, so the report's T1-weighting "
              "inference stays an inference from image contrast. That is fine, but say so.")
        return []

    summary: List[Dict[str, object]] = []
    for column in interesting:
        values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
        numeric = [parse_age(v) for v in values]
        numeric = [v for v in numeric if np.isfinite(v)]
        entry: Dict[str, object] = {"column": column, "non_empty": len(values),
                                    "distinct": len(set(values))}
        if numeric and len(numeric) > 0.5 * len(values):
            entry.update({"kind": "numeric", "min": round(min(numeric), 3),
                          "median": round(float(np.median(numeric)), 3),
                          "max": round(max(numeric), 3)})
        else:
            entry.update({"kind": "categorical",
                          "top": "; ".join(f"{k}={v}" for k, v in
                                           Counter(values).most_common(6))})
        summary.append(entry)
        print(f"  {column}: {entry}")

    path = os.path.join(out_dir, "acquisition_columns.csv")
    fields = sorted({k for e in summary for k in e})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {path}")
    return summary


# ---------------------------------------------------------------------------
# 3. Repeat sessions, which would give a real noise estimate
# ---------------------------------------------------------------------------


def repeats(data_root: str, out_dir: str) -> Dict[str, List[str]]:
    """Find subjects with more than one session on disk (test-retest pairs)."""
    volumes = discover_volumes(data_root)
    groups: Dict[str, List[str]] = {}
    for name in volumes:
        stem = re.sub(r"\.(npy|npz)$", "", name)
        m = re.match(r"^(.*?)[_-]?(?:Ses|ses|session|visit|run)[_-]?(\d+)$", stem)
        if m:
            groups.setdefault(m.group(1), []).append(name)
    multi = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    print(f"\n--- repeat sessions ---")
    print(f"  subjects with a session suffix: {len(groups)}")
    print(f"  of those, with more than one session on disk: {len(multi)}")
    if multi:
        print("  A repeat pair of the same subject gives a direct estimate of scanner")
        print("  noise, which would let the forward model carry real noise instead of")
        print("  being exactly consistent. That is the biggest simulation limitation the")
        print("  report admits to. Examples:")
        for k, v in list(multi.items())[:5]:
            print(f"    {k}: {v}")
        path = os.path.join(out_dir, "repeat_sessions.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "sessions"])
            for k, v in sorted(multi.items()):
                writer.writerow([k, " ".join(v)])
        print(f"  wrote {path}")
    else:
        print("  None, so the noise-estimation experiment is not available.")
    return multi


# ---------------------------------------------------------------------------
# 4. Through-plane profile: the evidence for the one-central-slice rule
# ---------------------------------------------------------------------------


def slice_profiles(data_root: str, out_dir: str, results_root: str, n_subjects: int = 6,
                   slice_axis: int = 2, image_size: int = 128,
                   min_signal: float = 0.01) -> plt.Figure:
    """Signal along the slice axis for a few test volumes, with the chosen slice marked."""
    members = load_split_members(results_root)
    names = members.get("test", [])[:n_subjects] or None
    root = os.path.join(data_root, "selected_npy")
    if names is None:
        names = [n for n in sorted(os.listdir(root))
                 if n.endswith(".npy")][:n_subjects]

    fig, axes = plt.subplots(2, n_subjects, figsize=(2.6 * n_subjects, 6.0),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    rows: List[Dict[str, object]] = []

    for col, name in enumerate(names):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        volume = np.squeeze(np.load(path, allow_pickle=True))
        volume = np.moveaxis(volume, slice_axis, -1)
        magnitude = np.abs(volume)
        profile = magnitude.mean(axis=(0, 1))
        n = profile.size
        middle = n // 2

        # The project's rule: the middle slice, walking outward to the first slice whose
        # mean magnitude clears min_signal.
        chosen = middle
        for offset in range(n):
            for idx in ({middle + offset, middle - offset} if offset else {middle}):
                if 0 <= idx < n and profile[idx] > min_signal:
                    chosen = idx
                    break
            else:
                continue
            break

        usable = int((profile > min_signal).sum())
        rows.append({"volume": name, "slices": n, "chosen_slice": chosen,
                     "usable_slices": usable,
                     "fraction_used": round(1.0 / max(usable, 1), 5),
                     "shape": "x".join(map(str, volume.shape))})

        ax = axes[0, col]
        ax.imshow(np.abs(volume[..., chosen]).T, cmap="gray")
        ax.set_title(f"{name[:16]}\nslice {chosen} of {n}", fontsize=8)
        ax.axis("off")

        ax = axes[1, col]
        ax.plot(np.arange(n), profile, color="0.3", linewidth=1.2)
        ax.axhline(min_signal, color="tab:orange", linestyle=":", linewidth=1,
                   label="min_signal")
        ax.axvline(chosen, color="tab:green", linewidth=1.4, label="slice we keep")
        ax.fill_between(np.arange(n), 0, profile, where=profile > min_signal,
                        color="tab:blue", alpha=0.15)
        ax.set_xlabel("slice index", fontsize=8)
        if col == 0:
            ax.set_ylabel("mean |signal| per slice", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("The one-central-slice rule: what it selects and what it discards.\n"
                 "Each volume contributes exactly one slice to the experiment; the shaded "
                 "region is the anatomy that clears the signal threshold and is not used.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    png = os.path.join(out_dir, "dataset_slice_profiles.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {png}")

    if rows:
        path = os.path.join(out_dir, "dataset_slice_profiles.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
        used = [r["usable_slices"] for r in rows]
        print(f"  volumes carry {min(used)}-{max(used)} slices above the signal threshold, "
              f"of which we use 1. That is the through-plane context the report says a 3D "
              f"method would exploit.")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL = ("audit", "acquisition", "repeats", "slices")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None,
                   help="Directory holding the metadata CSVs and selected_npy/. "
                        "Auto-detected from ~/data and ~/MRI_2026_datasets/brain_age.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default=os.path.join("results", "dataset_evidence"))
    p.add_argument("--which", nargs="*", default=list(ALL), choices=ALL)
    p.add_argument("--n-subjects", type=int, default=6,
                   help="How many volumes to profile for the slice figure.")
    p.add_argument("--slice-axis", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    data_root = find_data_root(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data root : {data_root}")
    print(f"output dir: {args.out_dir}")

    if "audit" in args.which:
        audit(data_root, args.out_dir, args.results_root)
    if "acquisition" in args.which:
        acquisition(data_root, args.out_dir)
    if "repeats" in args.which:
        repeats(data_root, args.out_dir)
    if "slices" in args.which:
        slice_profiles(data_root, args.out_dir, args.results_root, args.n_subjects,
                       args.slice_axis)

    print(f"\nAll outputs are small derived files; no image data was copied.")
    return args.out_dir


if __name__ == "__main__":
    main()
