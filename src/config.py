# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Configuration loading: YAML + deep-merge overrides + runtime path resolution.

A config is a plain nested dict. Experiment configs inherit from a base config via
an optional top-level ``inherit:`` key, and any field can be overridden either by an
experiment file or by ``key.subkey=value`` strings on the command line. Data and
result paths are resolved differently on Google Colab vs a local machine so the same
config runs in both environments.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(value: str) -> Any:
    """Parse a CLI string into an int/float/bool/None/str using YAML rules."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_dotted_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> None:
    """Apply ``a.b.c=value`` style overrides in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must be of the form key.subkey=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce(raw)


def load_config(
    path: str | os.PathLike,
    cli_overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load a YAML config, resolving inheritance, CLI overrides, and paths."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("inherit", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)

    if cli_overrides:
        _apply_dotted_overrides(cfg, list(cli_overrides))

    cfg = resolve_paths(cfg)
    return cfg


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``paths.data_root`` / ``paths.results_root`` for Colab vs local."""
    paths = cfg.setdefault("paths", {})
    env = "colab" if in_colab() else "local"
    selected = paths.get(env, {})

    if in_colab() and selected.get("mount_drive", True):
        _mount_drive()

    paths["data_root"] = os.path.expanduser(selected.get("data_root", "data"))
    paths["results_root"] = os.path.expanduser(selected.get("results_root", "results"))
    # Cache for pre-extracted slices; defaults to a "cache" dir next to the results.
    cache_root = selected.get("cache_root") or os.path.join(paths["results_root"], "cache")
    paths["cache_root"] = os.path.expanduser(cache_root)
    os.makedirs(paths["results_root"], exist_ok=True)
    return cfg


def _mount_drive() -> None:
    """Mount Google Drive on Colab (no-op if already mounted)."""
    if os.path.ismount("/content/drive"):
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
