# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Sweep orchestration: run many configurations from a single sweep YAML.

A sweep file specifies a base config, a set of fixed overrides applied to every run,
and a grid of axes whose Cartesian product defines the individual runs::

    base: configs/default.yaml
    fixed:
      train.epochs: 100
    grid:
      model.name: [zero_filled, classical_cs, unet, ista_net, admmnet_softthresh, admmnet_pwl]
      mask.acceleration: [4, 8]
      train.seed: [0, 1, 2]

Each grid point is translated into ``key=value`` overrides and passed to
``train.main``, which appends its metrics to the shared ``results/runs.csv``.

Resumability operates at two levels:

* Sweep level (here): before launching a run, its ``run_id`` (the config hash,
  identical to the one ``train.main`` computes) is checked against the run ids already
  present in ``runs.csv``; completed runs are skipped. A sweep can therefore be
  interrupted and re-launched, and it will only run what is missing. Failures in one
  run are reported but do not abort the rest of the sweep.
* Within-run (``train.main``): a run that crashed before finishing wrote no ``runs.csv``
  row, so it is *not* skipped -- it is re-launched, and ``train.main`` automatically
  resumes it from its ``last.pth`` checkpoint instead of restarting from epoch 1.

Usage::

    python -m src.run_experiments --sweep configs/experiments/comparison.yaml
    python -m src.run_experiments --sweep <file> --dry-run
    python -m src.run_experiments --sweep <file> --set paths.local.data_root=/path/to/data
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import traceback
from typing import Dict, List

import yaml

from .config import load_config
from .train import main as train_main
from .utils import run_id


def load_sweep(path: str) -> Dict:
    with open(path, "r") as f:
        sweep = yaml.safe_load(f) or {}
    sweep.setdefault("base", "configs/default.yaml")
    sweep.setdefault("fixed", {})
    sweep.setdefault("grid", {})
    return sweep


def _fmt(value) -> str:
    """Render an override value as a YAML/JSON-parseable string."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _as_overrides(mapping: Dict) -> List[str]:
    return [f"{k}={_fmt(v)}" for k, v in mapping.items()]


def expand_grid(grid: Dict[str, List]) -> List[Dict]:
    """Cartesian product of grid axes into a list of override mappings."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def completed_run_ids(csv_path: str) -> set:
    """Run ids that finished and logged everything the analysis needs.

    A run counts as complete only when it appears in ``runs.csv`` *and* in
    ``samples.csv``. Requiring both catches runs that were interrupted between the two
    writes: they would otherwise be skipped forever, leaving the per-sample scatter plots
    silently short of a seed.
    """
    if not os.path.exists(csv_path):
        return set()
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns:
        return set()
    done = set(df["run_id"].unique())

    samples_path = os.path.join(os.path.dirname(csv_path), "samples.csv")
    if os.path.exists(samples_path):
        samples = pd.read_csv(samples_path)
        if "run_id" in samples.columns:
            done &= set(samples["run_id"].unique())
    return done


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a configuration sweep.")
    p.add_argument("--sweep", required=True, help="Path to a sweep YAML.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                   help="Extra overrides applied to every run (highest priority).")
    p.add_argument("--dry-run", action="store_true",
                   help="List the runs that would execute without training.")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, int]:
    args = parse_args(argv)
    sweep = load_sweep(args.sweep)
    base = sweep["base"]
    fixed = _as_overrides(sweep["fixed"])
    grid_points = expand_grid(sweep["grid"])

    # Resolve the results CSV path once from the base config.
    base_cfg = load_config(base, cli_overrides=fixed + list(args.set))
    csv_path = os.path.join(base_cfg["paths"]["results_root"], "runs.csv")
    done = completed_run_ids(csv_path)

    print(f"sweep: {args.sweep}")
    print(f"base: {base} | runs: {len(grid_points)} | already complete: {len(done)}")
    print(f"results csv: {csv_path}\n")

    counts = {"ran": 0, "skipped": 0, "failed": 0}
    for i, point in enumerate(grid_points, 1):
        overrides = fixed + _as_overrides(point) + list(args.set)
        cfg = load_config(base, cli_overrides=overrides)
        rid = run_id(cfg)
        label = ", ".join(_as_overrides(point)) or "(base)"

        if rid in done:
            print(f"[{i}/{len(grid_points)}] SKIP  {rid}  {label}")
            counts["skipped"] += 1
            continue

        if args.dry_run:
            print(f"[{i}/{len(grid_points)}] PLAN  {rid}  {label}")
            counts["ran"] += 1
            continue

        print(f"[{i}/{len(grid_points)}] RUN   {rid}  {label}")
        try:
            train_main(["--config", base, "--set", *overrides])
            done.add(rid)
            counts["ran"] += 1
        except Exception:  # keep the sweep going if a single run fails
            counts["failed"] += 1
            print(f"[{i}/{len(grid_points)}] FAIL  {rid}\n{traceback.format_exc()}")

    print(f"\nsweep done | ran={counts['ran']} skipped={counts['skipped']} "
          f"failed={counts['failed']}")
    return counts


if __name__ == "__main__":
    result = main()
    raise SystemExit(1 if result["failed"] else 0)
