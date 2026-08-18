# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Every number the final report quotes, loaded once from the logged results.

The report is generated, not typed, so that it cannot drift out of sync with the
experiments. This module is the single place that reads the logs
(``results/samples.csv``, ``results/runs.csv``, ``results/crossmask.csv``, the per-figure
CSVs under ``results/figures/`` and each run's ``manifest.json`` / ``split.json``) and
turns them into the aggregates the text and tables need.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

RATIOS = (0.2, 0.3, 0.5)
METRICS = ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag")


class Results:
    """Everything the report quotes, loaded once from the logs."""

    def __init__(self, results_root: str = "results", cache_root: str = "cache"):
        self.results_root = results_root
        self.cache_root = cache_root
        self.samples = self._load_samples()
        self.runs = list(csv.DictReader(open(os.path.join(results_root, "runs.csv"))))
        self.figures = os.path.join(results_root, "figures")

    # -- loaders -----------------------------------------------------------
    def _load_samples(self) -> Dict[Tuple[str, float, int, str], Dict[str, float]]:
        out: Dict[Tuple[str, float, int, str], Dict[str, float]] = {}
        path = os.path.join(self.results_root, "samples.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["name"] != "comparison" or row["split"] != "test":
                    continue
                key = (row["method"], float(row["sampling_ratio"]),
                       int(row["seed"]), row["sample_index"])
                if key in out:
                    continue
                out[key] = {m: float(row[m]) for m in METRICS}
        return out

    def csv_rows(self, name: str) -> List[Dict[str, str]]:
        path = os.path.join(self.figures, name)
        if not os.path.exists(path):
            path = os.path.join(self.results_root, name)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    # -- aggregates --------------------------------------------------------
    def values(self, method: str, ratio: float, metric: str,
               seed: Optional[int] = None) -> List[float]:
        return [v[metric] for (m, r, s, _), v in self.samples.items()
                if m == method and r == ratio and (seed is None or s == seed)]

    def stat(self, method: str, ratio: float, metric: str,
             seed: Optional[int] = None) -> Tuple[float, float, int]:
        import statistics
        v = self.values(method, ratio, metric, seed)
        if not v:
            return float("nan"), float("nan"), 0
        return (statistics.fmean(v),
                statistics.stdev(v) if len(v) > 1 else 0.0,
                len(v))

    def cell(self, method: str, ratio: float, metric: str) -> str:
        mean, std, n = self.stat(method, ratio, metric)
        if not n:
            return "—"
        digits = 4 if metric.startswith("ssim") else 2
        return f"{mean:.{digits}f} &plusmn; {std:.{digits}f}"

    def mean(self, method: str, ratio: float, metric: str,
             seed: Optional[int] = None) -> float:
        return self.stat(method, ratio, metric, seed)[0]

    def wins(self, ratio: float, metric: str, baseline: str = "classical_cs_tv",
             model: str = "admmnet_softthresh") -> Tuple[int, int]:
        b = {(s, i): v[metric] for (m, r, s, i), v in self.samples.items()
             if m == baseline and r == ratio}
        g = {(s, i): v[metric] for (m, r, s, i), v in self.samples.items()
             if m == model and r == ratio}
        keys = set(b) & set(g)
        return sum(1 for k in keys if g[k] > b[k]), len(keys)

    def pearson(self, ratio: float, metric: str, seed: int = 0,
                baseline: str = "classical_cs_tv",
                model: str = "admmnet_softthresh") -> float:
        import statistics
        b = {i: v[metric] for (m, r, s, i), v in self.samples.items()
             if m == baseline and r == ratio and s == seed}
        g = {i: v[metric] for (m, r, s, i), v in self.samples.items()
             if m == model and r == ratio and s == seed}
        keys = sorted(set(b) & set(g))
        if len(keys) < 3:
            return float("nan")
        x = [b[k] for k in keys]
        y = [g[k] for k in keys]
        mx, my = statistics.fmean(x), statistics.fmean(y)
        sxy = sum((a - mx) * (c - my) for a, c in zip(x, y))
        sxx = sum((a - mx) ** 2 for a in x)
        syy = sum((c - my) ** 2 for c in y)
        return sxy / ((sxx * syy) ** 0.5) if sxx and syy else float("nan")

    # -- run-level tables --------------------------------------------------
    def ablation(self, experiment: str, label_cols: Sequence[str],
                 epochs: Optional[int] = None) -> List[Tuple[str, float, float]]:
        """(label, mean PSNR, mean SSIM) averaged over real+imag, per ablation cell."""
        manifests = {}
        for f in glob.glob(os.path.join(self.results_root, "*", "manifest.json")):
            run = os.path.basename(os.path.dirname(f))
            manifests[run] = json.load(open(f))["config"]

        buckets: Dict[str, Dict[str, List[float]]] = {}
        for row in self.runs:
            if row["name"] != experiment or row["split"] != "test":
                continue
            if row["method"] == "zero_filled":
                continue
            cfg = manifests.get(row["run_id"])
            if cfg is None:
                continue
            if epochs is not None and int(cfg["train"]["epochs"]) != epochs:
                continue
            label = ", ".join(str(_dig(cfg, c)) for c in label_cols)
            base = "psnr" if row["metric"].startswith("psnr") else "ssim"
            if row["metric"] not in METRICS:
                continue
            buckets.setdefault(label, {}).setdefault(base, []).append(float(row["value"]))

        import statistics
        out = []
        for label, d in buckets.items():
            psnr = statistics.fmean(d.get("psnr", [float("nan")]))
            ssim = statistics.fmean(d.get("ssim", [float("nan")]))
            out.append((label, psnr, ssim))
        return sorted(out, key=lambda r: _sortkey(r[0]))

    def unet_vs_model(self, seed: int = 0) -> List[Dict[str, float]]:
        """Per-ratio U-Net vs ADMM-Net on both channels, at a matched seed."""
        path = os.path.join(self.results_root, "samples.csv")
        buckets: Dict[Tuple[str, float, str], List[float]] = {}
        seen = set()
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["split"] != "test" or int(row["seed"]) != seed:
                    continue
                if row["name"] not in ("unet_reference", "comparison"):
                    continue
                if row["method"] not in ("unet", "admmnet_softthresh"):
                    continue
                uniq = (row["method"], row["sampling_ratio"], row["sample_index"])
                if uniq in seen:
                    continue
                seen.add(uniq)
                for m in METRICS:
                    buckets.setdefault(
                        (row["method"], float(row["sampling_ratio"]), m), []
                    ).append(float(row[m]))
        import statistics
        out = []
        for ratio in RATIOS:
            row = {"ratio": ratio}
            for method in ("unet", "admmnet_softthresh"):
                for m in METRICS:
                    v = buckets.get((method, ratio, m), [])
                    row[f"{method}_{m}"] = statistics.fmean(v) if v else float("nan")
            out.append(row)
        return out

    def crossmask(self) -> List[Dict[str, float]]:
        rows = self.csv_rows("crossmask.csv")
        import statistics
        out = []
        for method in ("admmnet_softthresh", "classical_cs_tv"):
            sub = [r for r in rows if r["method"] == method]
            if not sub:
                continue
            seen = [float(r["psnr_real"]) for r in sub
                    if r["mask_seen_in_training"] == "True"]
            unseen = [float(r["psnr_real"]) for r in sub
                      if r["mask_seen_in_training"] == "False"]
            out.append({"method": method,
                        "seen": statistics.fmean(seen) if seen else float("nan"),
                        "unseen": statistics.fmean(unseen) if unseen else float("nan"),
                        "n_seen": len(seen), "n_unseen": len(unseen)})
        return out

    def split_stats(self) -> List[Dict]:
        files = sorted(glob.glob(os.path.join(self.results_root, "*", "split.json")))
        if not files:
            return []
        stats = json.load(open(files[0]))["age_stats"]
        return [dict(name=n, **stats[n]) for n in ("train", "val", "test") if n in stats]

    def baseline_grid(self) -> List[Tuple[float, float, float, float]]:
        """(lam, tv_weight, val PSNR real, val PSNR imag) for the calibration sweep."""
        manifests = {}
        for f in glob.glob(os.path.join(self.results_root, "baseline_tuning_*",
                                        "manifest.json")):
            run = os.path.basename(os.path.dirname(f))
            manifests[run] = json.load(open(f))["config"]
        vals: Dict[str, Dict[str, float]] = {}
        for row in self.runs:
            if (row["name"] == "baseline_tuning" and row["split"] == "val"
                    and row["method"] != "zero_filled"
                    and row["metric"] in ("psnr_real", "psnr_imag")):
                vals.setdefault(row["run_id"], {})[row["metric"]] = float(row["value"])
        out = []
        for run, v in vals.items():
            cfg = manifests.get(run)
            if not cfg:
                continue
            out.append((float(cfg["model"]["lam"]), float(cfg["model"]["tv_weight"]),
                        v.get("psnr_real", float("nan")),
                        v.get("psnr_imag", float("nan"))))
        return sorted(out, key=lambda r: -(r[2] + r[3]) / 2)


def _dig(cfg: Dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _sortkey(label: str):
    try:
        return (0, float(label.split(",")[0]))
    except ValueError:
        return (1, label)
