# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""One-time slice-cache builder.

Reads every subject volume from the (slow, networked) dataset directory exactly once,
extracts the configured slices (central slice by default), and writes a compact per-split
cache under ``paths.cache_root``. Every subsequent training/evaluation run then loads its
slices instantly from this cache instead of re-reading the network filesystem, which is
the single biggest runtime win for the full sweep.

Because the train/val/test split is fixed (``data.split_seed``, independent of
``train.seed``), one cache serves all runs and all seeds.

Usage::

    python -m src.build_cache --config configs/default.yaml
    python -m src.build_cache --config configs/default.yaml --rebuild \
        --set paths.local.data_root=../MRI_2026_datasets/brain_age
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from .config import load_config
from .dataset import (
    build_splits,
    describe_split,
    _cache_dir,
    split_fingerprint,
    build_datasets,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the one-time MRI slice cache.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                   help="Dotted config overrides (e.g. paths.local.data_root=...).")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild the cache even if it already exists.")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, int]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)

    splits, meta, data_cfg = build_splits(cfg)
    fingerprint = split_fingerprint(splits, data_cfg)
    cache_dir = _cache_dir(cfg, fingerprint)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"data_root : {cfg['paths']['data_root']}")
    print(f"cache dir : {cache_dir}")
    print(f"central_slice_only={data_cfg.central_slice_only} | "
          f"split={data_cfg.split} | split_seed={data_cfg.split_seed} | "
          f"max_train_subjects={data_cfg.max_train_subjects}")
    print("subjects  : " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    datasets = build_datasets(cfg, use_cache=True, rebuild_cache=args.rebuild)
    counts = {name: len(ds) for name, ds in datasets.items()}
    print("slices    : " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Record the age-stratified split so the report can show the distributions match.
    split_info = {
        "fingerprint": fingerprint,
        "age_stats": describe_split(meta, splits, data_cfg),
        "n_slices": counts,
        "subjects": {k: [os.path.basename(p) for p in v] for k, v in splits.items()},
    }
    with open(os.path.join(cache_dir, "split.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"wrote     : {os.path.join(cache_dir, 'split.json')}")
    print("cache ready. subsequent runs will load slices from this cache.")
    return counts


if __name__ == "__main__":
    main()
