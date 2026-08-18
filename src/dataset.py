# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Dedicated dataset handler for the complex-valued MRI reconstruction task.

This module is purpose-built for the MRI course "Reconstruction" dataset (3D scans
stored as NumPy ``.npy`` volumes, with per-scan age/sex metadata in CSV files). It
replaces the original NIfTI/magnitude loader and implements the split mandated by the
professor's update.

Why a custom split (professor's update)
---------------------------------------
The provided ``test`` CSV lists subjects that are NOT present in the selected NumPy
directory, so the official train/val/test CSVs cannot be used as-is. We therefore:

1. Read the three provided CSVs (train/val/test metadata) and concatenate them into a
   single master metadata table.
2. Discover the ``.npy`` volumes actually present on disk.
3. Cross-reference the two, keeping only subjects that have a matching ``.npy`` file
   (this is what resolves the professor's bug).
4. Build our own explicit, reproducible **age-stratified 3-way split** (train / val /
   test) so all three splits share approximately the same age distribution -- avoiding
   an age confound between splits. The resulting split is recorded so it can be
   reported.

MRI-specific preprocessing
--------------------------
* Volumes are complex-valued; we keep both real and imaginary parts (phase matters in
  MRI and the brief asks for PSNR/SSIM on real and imaginary components separately).
* One consistent normalization (scale by the global maximum magnitude of each volume)
  is applied before slicing, so metrics are comparable across images and methods.
* Central axial slices are extracted (the brief permits using central slices as long as
  the same rule is applied to every method and split); near-empty slices are dropped.
* The dataset returns a 2-channel real tensor ``(2, H, W)`` = (real, imag); k-space
  generation and undersampling happen later in the training engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NUMPY_SUFFIXES = (".npy", ".npz")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Typed view of the ``data`` section of a config."""

    image_size: int = 128
    max_slices_per_subject: int = 32
    central_fraction: Tuple[float, float] = (0.30, 0.70)
    central_slice_only: bool = True    # keep only the single central slice per subject
    slice_axis: int = 2
    min_signal: float = 0.01
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 0                # seed for the age-stratified split (fixed across runs)
    max_train_subjects: Optional[int] = None  # optional cap on train subjects (speed lever)
    batch_size: int = 4
    # Metadata / cross-referencing
    numpy_dir: str = "selected_numpy"
    meta_csvs: Tuple[str, ...] = ("train.csv", "validation.csv", "test.csv")
    id_col: str = "subject_id"
    age_col: str = "age"
    path_col: str = ""             # optional CSV column giving each scan's file path
    npy_name_template: str = "{id}.npy"
    age_bins: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        max_train = d.get("max_train_subjects", None)
        return cls(
            image_size=d.get("image_size", 128),
            max_slices_per_subject=d.get("max_slices_per_subject", 32),
            central_fraction=tuple(d.get("central_fraction", (0.30, 0.70))),
            central_slice_only=bool(d.get("central_slice_only", True)),
            slice_axis=d.get("slice_axis", 2),
            min_signal=d.get("min_signal", 0.01),
            split=tuple(d.get("split", (0.7, 0.15, 0.15))),
            split_seed=int(d.get("split_seed", 0)),
            max_train_subjects=(int(max_train) if max_train not in (None, "", "null") else None),
            batch_size=d.get("batch_size", 4),
            numpy_dir=d.get("numpy_dir", "selected_numpy"),
            meta_csvs=tuple(d.get("meta_csvs", ("train.csv", "validation.csv", "test.csv"))),
            id_col=d.get("id_col", "subject_id"),
            age_col=d.get("age_col", "age"),
            path_col=d.get("path_col", ""),
            npy_name_template=d.get("npy_name_template", "{id}.npy"),
            age_bins=d.get("age_bins", 4),
        )


# ---------------------------------------------------------------------------
# Metadata: read CSVs, discover files, cross-reference
# ---------------------------------------------------------------------------


def _read_master_metadata(data_root: str, cfg: DataConfig):
    """Concatenate the provided CSVs into one master metadata DataFrame."""
    import pandas as pd

    frames = []
    for name in cfg.meta_csvs:
        path = os.path.join(data_root, name)
        if not os.path.exists(path):
            warnings.warn(f"metadata CSV not found, skipping: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"none of the metadata CSVs {cfg.meta_csvs} were found under {data_root}"
        )
    master = pd.concat(frames, ignore_index=True)
    for col in (cfg.id_col, cfg.age_col):
        if col not in master.columns:
            raise KeyError(
                f"expected column '{col}' in metadata CSVs; found {list(master.columns)}"
            )
    master = master.drop_duplicates(subset=[cfg.id_col]).reset_index(drop=True)
    return master


def _discover_npy(numpy_root: str) -> Dict[str, str]:
    """Map a file *stem* -> absolute path for every ``.npy``/``.npz`` on disk."""
    if not os.path.isdir(numpy_root):
        raise FileNotFoundError(f"numpy directory does not exist: {numpy_root}")
    out: Dict[str, str] = {}
    for f in sorted(os.listdir(numpy_root)):
        if f.endswith(NUMPY_SUFFIXES):
            full = os.path.join(numpy_root, f)
            # Filter out 0-byte files up front so they never enter a split.
            try:
                if os.path.getsize(full) == 0:
                    warnings.warn(f"ignoring empty (0-byte) file during discovery: {full}")
                    continue
            except OSError:
                continue
            stem = f
            for suf in NUMPY_SUFFIXES:
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            out[stem] = full
    if not out:
        raise FileNotFoundError(f"no .npy/.npz files found in {numpy_root}")
    return out


def _stem_of(name: str) -> str:
    """Strip a NumPy suffix from a filename to get its stem."""
    for suf in NUMPY_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _resolve_path(subject_id: str, available: Dict[str, str], template: str) -> Optional[str]:
    """Find the on-disk file for a subject id, tolerant to naming differences."""
    sid = str(subject_id)
    # 1. exact filename from the template
    name = template.format(id=sid)
    for stem, path in available.items():
        if os.path.basename(path) == name:
            return path
    # 2. stem equals the id
    if sid in available:
        return available[sid]
    # 3. id is a substring of the stem (handles prefixes/suffixes in filenames)
    for stem, path in available.items():
        if sid in stem:
            return path
    return None


def _resolve_from_filepath(file_path: object, available: Dict[str, str]) -> Optional[str]:
    """Match a CSV ``filePath`` value to a file present in the NumPy directory.

    We only trust the *basename* of the CSV path (the recorded path may point to an
    original location that differs from where the selected NumPy volumes now live),
    and match it against the files actually on disk.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    base = os.path.basename(file_path.strip())
    for stem, path in available.items():
        if os.path.basename(path) == base:
            return path
    stem_q = _stem_of(base)
    if stem_q in available:
        return available[stem_q]
    for stem, path in available.items():
        if stem_q and stem_q in stem:
            return path
    return None


def cross_reference(data_root: str, cfg: DataConfig):
    """Master metadata filtered to subjects that have a matching ``.npy`` on disk.

    Returns a DataFrame with an added ``path`` column. Warns about how many metadata
    rows were dropped for lacking a file (the professor's ``test_csv`` mismatch).
    """
    master = _read_master_metadata(data_root, cfg)
    available = _discover_npy(os.path.join(data_root, cfg.numpy_dir))

    if cfg.path_col and cfg.path_col in master.columns:
        # Preferred: match on the filename recorded in the CSV's path column.
        paths = [_resolve_from_filepath(fp, available) for fp in master[cfg.path_col]]
    else:
        # Fallback: reconstruct the filename from the subject id + template.
        paths = [_resolve_path(sid, available, cfg.npy_name_template)
                 for sid in master[cfg.id_col]]
    master = master.assign(path=paths)
    n_before = len(master)
    matched = master[master["path"].notna()].reset_index(drop=True)
    n_dropped = n_before - len(matched)
    if n_dropped:
        warnings.warn(
            f"cross-reference dropped {n_dropped}/{n_before} metadata rows with no "
            f"matching .npy on disk (expected, per the professor's update)."
        )
    if matched.empty:
        raise RuntimeError(
            "no metadata subjects could be matched to .npy files; check "
            "data.path_col / data.id_col / data.npy_name_template against the actual "
            "filenames in the NumPy directory."
        )
    return matched


# ---------------------------------------------------------------------------
# Age-stratified 3-way split
# ---------------------------------------------------------------------------


def stratified_split(meta, cfg: DataConfig, seed: int = 0) -> Dict[str, List[str]]:
    """Split subjects into train/val/test preserving the age distribution.

    Age is binned into ``cfg.age_bins`` quantile bins; within each bin the subjects
    are shuffled and partitioned by the configured fractions. Because every bin is
    split by the same fractions, the three splits share (approximately) the overall
    age distribution. Deterministic given ``seed``.
    """
    import pandas as pd

    if len(cfg.split) != 3 or not np.isclose(sum(cfg.split), 1.0):
        raise ValueError(f"split must be three fractions summing to 1, got {cfg.split}")

    rng = np.random.default_rng(seed)
    ages = pd.to_numeric(meta[cfg.age_col], errors="coerce")
    # Quantile bins; fall back to fewer bins if there are many duplicate ages.
    try:
        bins = pd.qcut(ages, q=min(cfg.age_bins, ages.nunique()), duplicates="drop")
    except (ValueError, IndexError):
        bins = pd.Series(["all"] * len(meta), index=meta.index)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for _, group in meta.groupby(bins, observed=True):
        idx = rng.permutation(group.index.to_numpy())
        n = len(idx)
        n_train = int(round(cfg.split[0] * n))
        n_val = int(round(cfg.split[1] * n))
        # Guard tiny bins: keep at least the test remainder non-negative.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        splits["train"].extend(meta.loc[idx[:n_train], "path"].tolist())
        splits["val"].extend(meta.loc[idx[n_train : n_train + n_val], "path"].tolist())
        splits["test"].extend(meta.loc[idx[n_train + n_val :], "path"].tolist())

    for name, paths in splits.items():
        if not paths:
            raise RuntimeError(
                f"the '{name}' split is empty after stratification; there may be too "
                f"few matched subjects. Reduce data.age_bins or add data."
            )
    return splits


def cap_train_subjects(meta, splits: Dict[str, List[str]], cfg: DataConfig,
                       seed: int = 0) -> Dict[str, List[str]]:
    """Optionally subsample the *train* split to ``cfg.max_train_subjects`` (speed lever).

    The subsample is age-stratified (same quantile bins as the split) so the reduced
    training set keeps the overall age distribution. ``val`` and ``test`` are never
    touched, so the test set remains a legitimate, representative sample.
    """
    import pandas as pd

    cap = cfg.max_train_subjects
    train_paths = splits["train"]
    if not cap or cap >= len(train_paths):
        return splits

    by_path = meta.set_index("path")
    ages = pd.to_numeric(by_path.loc[train_paths, cfg.age_col], errors="coerce")
    try:
        bins = pd.qcut(ages, q=min(cfg.age_bins, ages.nunique()), duplicates="drop")
    except (ValueError, IndexError):
        bins = pd.Series(["all"] * len(train_paths), index=ages.index)

    rng = np.random.default_rng(seed)
    frac = cap / len(train_paths)
    kept: List[str] = []
    for _, group in ages.groupby(bins, observed=True):
        idx = np.array(group.index.tolist(), dtype=object)
        idx = idx[rng.permutation(len(idx))]
        n_keep = max(1, int(round(frac * len(idx))))
        kept.extend(idx[:n_keep].tolist())
    # Correct rounding drift so we land on exactly ``cap`` subjects.
    if len(kept) > cap:
        kept = list(np.array(kept, dtype=object)[rng.permutation(len(kept))][:cap])

    out = dict(splits)
    out["train"] = kept
    return out


def describe_split(meta, splits: Dict[str, List[str]], cfg: DataConfig) -> Dict[str, Dict]:
    """Per-split age statistics, to demonstrate the distributions match (for the report)."""
    import pandas as pd

    by_path = meta.set_index("path")
    out: Dict[str, Dict] = {}
    for name, paths in splits.items():
        ages = pd.to_numeric(by_path.loc[paths, cfg.age_col], errors="coerce").dropna()
        out[name] = {
            "n_subjects": int(len(paths)),
            "age_mean": float(ages.mean()) if len(ages) else float("nan"),
            "age_std": float(ages.std()) if len(ages) else float("nan"),
            "age_min": float(ages.min()) if len(ages) else float("nan"),
            "age_max": float(ages.max()) if len(ages) else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Preprocessing primitives (pure functions)
# ---------------------------------------------------------------------------


def normalize_complex_volume(volume: np.ndarray) -> np.ndarray:
    """Scale a (possibly complex) volume so its maximum magnitude is 1 (no-op if 0)."""
    volume = volume.astype(np.complex64) if np.iscomplexobj(volume) else volume.astype(np.float32)
    max_mag = float(np.abs(volume).max())
    if max_mag > 0:
        volume = volume / max_mag
    return volume.astype(np.complex64)


def center_crop_or_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or zero-pad a 2D image to ``size`` x ``size`` (complex-safe)."""
    h, w = img.shape
    if h > size:
        top = (h - size) // 2
        img = img[top : top + size, :]
    if w > size:
        left = (w - size) // 2
        img = img[:, left : left + size]

    h, w = img.shape
    pad_h, pad_w = size - h, size - w
    return np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        mode="constant",
    )


def extract_slices(volume: np.ndarray, cfg: DataConfig) -> List[np.ndarray]:
    """Extract informative central slices from a normalized complex volume.

    When ``cfg.central_slice_only`` is set (the fast, professor-permitted mode) we keep
    exactly one slice: the anatomical middle slice (``num // 2``), walking outward to the
    nearest slice whose mean magnitude clears ``cfg.min_signal``. This makes the number of
    samples in a split equal to the number of subjects, so the test set stays a legitimate
    "few hundred" (~10%) with the split's age distribution.
    """
    volume = np.moveaxis(volume, cfg.slice_axis, -1)
    num = volume.shape[-1]

    if cfg.central_slice_only:
        mid = num // 2
        # Search outward from the middle for the first sufficiently-signalled slice.
        for offset in range(0, num):
            for idx in ({mid + offset, mid - offset} if offset else {mid}):
                if not 0 <= idx < num:
                    continue
                sl = center_crop_or_pad(volume[..., idx], cfg.image_size)
                if float(np.abs(sl).mean()) > cfg.min_signal:
                    return [sl.astype(np.complex64)]
        # Nothing cleared the threshold: fall back to the geometric middle slice.
        return [center_crop_or_pad(volume[..., mid], cfg.image_size).astype(np.complex64)]

    lo = int(cfg.central_fraction[0] * num)
    hi = int(cfg.central_fraction[1] * num)

    slices: List[np.ndarray] = []
    for i in range(lo, hi):
        sl = center_crop_or_pad(volume[..., i], cfg.image_size)
        if float(np.abs(sl).mean()) > cfg.min_signal:
            slices.append(sl.astype(np.complex64))
        if len(slices) >= cfg.max_slices_per_subject:
            break
    return slices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MRISliceDataset(Dataset):
    """Complex 2D MRI slices from a fixed list of subject ``.npy`` volumes.

    Each item is a 2-channel real tensor ``(2, H, W)`` = (real, imag). Because the
    subject list is already restricted to one split, no slice can leak across splits.
    """

    def __init__(self, paths: Sequence[str], data_cfg: DataConfig):
        self.cfg = data_cfg
        self.slices: List[np.ndarray] = []
        self.subject_of_slice: List[str] = []

        skipped = 0
        for path in paths:
            volume = _load_volume(path)
            if volume is None:  # corrupted / empty file -> skip this subject
                skipped += 1
                continue
            volume = normalize_complex_volume(volume)
            subject_slices = extract_slices(volume, data_cfg)
            self.slices.extend(subject_slices)
            self.subject_of_slice.extend([os.path.basename(path)] * len(subject_slices))
        if skipped:
            warnings.warn(f"skipped {skipped}/{len(paths)} unreadable/empty volumes in this split")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sl = self.slices[idx]
        real = torch.from_numpy(np.ascontiguousarray(sl.real)).float()
        imag = torch.from_numpy(np.ascontiguousarray(sl.imag)).float()
        return torch.stack([real, imag], dim=0)  # (2, H, W)

    def to_channel_array(self) -> Tuple[np.ndarray, List[str]]:
        """Stack all slices into a single ``(N, 2, H, W)`` float32 array for caching."""
        if not self.slices:
            size = self.cfg.image_size
            return np.zeros((0, 2, size, size), dtype=np.float32), []
        data = np.stack(
            [np.stack([sl.real, sl.imag], axis=0) for sl in self.slices], axis=0
        ).astype(np.float32)
        return data, list(self.subject_of_slice)


class CachedSliceDataset(Dataset):
    """Serve pre-extracted 2-channel slices from an in-memory ``(N, 2, H, W)`` array.

    Used when a split's slices have already been extracted and written to the on-disk
    cache, so no network volume reads are needed.
    """

    def __init__(self, data: np.ndarray, subjects: Sequence[str]):
        self.data = torch.from_numpy(np.ascontiguousarray(data)).float()
        self.subject_of_slice = list(subjects)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_volume(path: str) -> Optional[np.ndarray]:
    """Load a 3D volume from ``.npy`` or ``.npz`` and squeeze singleton dims.

    ``allow_pickle=True`` is required because the provided volumes are stored as pickled
    NumPy objects. If a file wraps the array in a 0-d object array (e.g. a dict or a
    boxed ndarray), unwrap it before returning.

    Robust to corrupted / empty files: returns ``None`` (with a warning) instead of
    raising, so a single bad ``.npy`` cannot crash the whole training run.
    """
    import pickle

    try:
        if os.path.getsize(path) == 0:
            warnings.warn(f"skipping empty (0-byte) file: {path}")
            return None
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        # A pickled object may come back as a 0-d object array wrapping the real array.
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 0:
            arr = arr.item()
        vol = np.squeeze(np.asarray(arr))
        if vol.ndim < 2 or vol.size == 0:
            warnings.warn(f"skipping file with unexpected/empty contents: {path}")
            return None
        return vol
    except (EOFError, pickle.UnpicklingError, ValueError, OSError) as exc:
        warnings.warn(f"skipping unreadable/corrupted file {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------


def build_splits(cfg: dict) -> Tuple[Dict[str, List[str]], "object", DataConfig]:
    """Cross-reference metadata with files and produce the age-stratified split.

    Returns ``(splits, meta_dataframe, data_cfg)``.
    """
    data_cfg = DataConfig.from_dict(cfg["data"])
    data_root = cfg["paths"]["data_root"]
    meta = cross_reference(data_root, data_cfg)
    # Fixed, reproducible split (independent of train.seed) so the test set is identical
    # across every run/seed -- required for consistent sample-wise scatter plots and for a
    # single, stable age-matched test set.
    splits = stratified_split(meta, data_cfg, seed=data_cfg.split_seed)
    splits = cap_train_subjects(meta, splits, data_cfg, seed=data_cfg.split_seed)
    return splits, meta, data_cfg


# ---------------------------------------------------------------------------
# On-disk slice cache
# ---------------------------------------------------------------------------


def split_fingerprint(splits: Dict[str, List[str]], data_cfg: DataConfig) -> str:
    """Stable hash of everything that determines the extracted slices.

    Includes preprocessing knobs and the exact subject membership of each split, so a
    cache is reused only when it corresponds byte-for-byte to the requested data.
    """
    payload = {
        "image_size": data_cfg.image_size,
        "slice_axis": data_cfg.slice_axis,
        "central_slice_only": data_cfg.central_slice_only,
        "central_fraction": list(data_cfg.central_fraction),
        "max_slices_per_subject": data_cfg.max_slices_per_subject,
        "min_signal": data_cfg.min_signal,
        "numpy_dir": data_cfg.numpy_dir,
        "split": list(data_cfg.split),
        "split_seed": data_cfg.split_seed,
        "max_train_subjects": data_cfg.max_train_subjects,
        "members": {k: sorted(os.path.basename(p) for p in v) for k, v in splits.items()},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _cache_root(cfg: dict) -> str:
    paths = cfg.get("paths", {})
    root = paths.get("cache_root") or os.path.join(paths.get("results_root", "results"), "cache")
    return os.path.expanduser(root)


def _cache_dir(cfg: dict, fingerprint: str) -> str:
    return os.path.join(_cache_root(cfg), fingerprint)


def _load_cached_split(cache_dir: str, name: str) -> Optional["CachedSliceDataset"]:
    path = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as npz:
        data = npz["data"]
        subjects = [str(s) for s in npz["subjects"].tolist()]
    return CachedSliceDataset(data, subjects)


def _save_cached_split(cache_dir: str, name: str, data: np.ndarray, subjects: List[str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    # np.savez keeps a filename that already ends in .npz, so the atomic rename is safe.
    tmp = os.path.join(cache_dir, f".{name}.tmp.npz")
    final = os.path.join(cache_dir, f"{name}.npz")
    np.savez(tmp, data=data.astype(np.float32),
             subjects=np.array(subjects, dtype=object))
    os.replace(tmp, final)


def build_datasets(cfg: dict, use_cache: bool = True, rebuild_cache: bool = False):
    """Build train/val/test datasets, reading from (or writing to) the slice cache.

    If a cache matching the current split/preprocessing exists, slices are loaded from it
    (no network volume reads). Otherwise volumes are read once, sliced, and the result is
    written to the cache so every subsequent run/seed loads instantly.
    """
    splits, _meta, data_cfg = build_splits(cfg)
    fingerprint = split_fingerprint(splits, data_cfg)
    cache_dir = _cache_dir(cfg, fingerprint)

    datasets: Dict[str, Dataset] = {}
    for name, paths in splits.items():
        cached = None
        if use_cache and not rebuild_cache:
            cached = _load_cached_split(cache_dir, name)
        if cached is not None:
            datasets[name] = cached
            continue
        # Cache miss: extract from volumes (the slow, one-time path) and persist.
        ds = MRISliceDataset(paths, data_cfg)
        if use_cache:
            data, subjects = ds.to_channel_array()
            _save_cached_split(cache_dir, name, data, subjects)
        datasets[name] = ds
    return datasets


def build_dataloaders(cfg: dict) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders from a full config dict."""
    data_cfg = DataConfig.from_dict(cfg["data"])
    datasets = build_datasets(cfg)
    num_workers = cfg.get("train", {}).get("num_workers", 0)
    return {
        "train": DataLoader(datasets["train"], batch_size=data_cfg.batch_size,
                            shuffle=True, num_workers=num_workers, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=data_cfg.batch_size,
                          shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=data_cfg.batch_size,
                           shuffle=False, num_workers=num_workers),
    }
