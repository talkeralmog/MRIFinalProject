# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Display conventions shared by every figure that shows a brain slice.

The cached slices are stored with the array's first axis running left-right across the
head, so plotting them directly puts the interhemispheric fissure horizontally -- a
sideways axial view. Radiological convention is anterior at the top with the fissure
vertical, so every figure transposes the array before drawing it.

That transpose has a consequence worth stating once, here, rather than in every caption:
the undersampling mask keeps whole rows of the stored array (axis 0), so in the *displayed*
images the phase-encode direction runs **left-right**, and the aliasing smears horizontally.
Nothing about the experiment changes -- only which way the picture is turned.

Also collects the two things the figure modules need from disk but that live behind the
dataset loader: the cached slices and the per-split subject lists.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Which display axis the phase-encode (undersampled) direction ends up on.
PHASE_ENCODE_AXIS = "horizontal"
PHASE_ENCODE_LABEL = "$k_y$ (phase encode)"


def to_display(image: np.ndarray) -> np.ndarray:
    """Turn a stored slice into the anatomical viewing orientation.

    Accepts a 2D array or a stack whose last two axes are the image.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"expected at least a 2D image, got shape {array.shape}")
    return np.swapaxes(array, -2, -1)


def find_cache_dir(cache_root: str = "cache") -> str:
    """The newest slice-cache fingerprint directory under ``cache_root``."""
    candidates = sorted(glob.glob(os.path.join(cache_root, "*", "test.npz")),
                        key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"no slice cache under {cache_root}/*/test.npz; run "
            "`python -m src.build_cache` on a machine that has the dataset")
    return os.path.dirname(candidates[-1])


def load_split(split: str = "test", cache_root: str = "cache"
               ) -> Tuple[np.ndarray, List[str]]:
    """Cached slices of one split as ``(N, 2, H, W)`` plus their subject filenames."""
    path = os.path.join(find_cache_dir(cache_root), f"{split}.npz")
    with np.load(path, allow_pickle=True) as npz:
        data = np.ascontiguousarray(npz["data"])
        subjects = [str(s) for s in npz["subjects"].tolist()]
    return data, subjects


def load_ages(results_root: str = "results") -> Dict[str, List[float]]:
    """Per-split subject ages, recovered from a run's ``split.json`` + the metadata CSVs.

    Falls back to an empty mapping when the metadata CSVs are not reachable (the split
    files record the subject lists and the summary statistics, but not the per-subject
    ages), so callers can degrade to plotting the summary instead.
    """
    files = sorted(glob.glob(os.path.join(results_root, "*", "split.json")))
    if not files:
        return {}
    with open(files[0]) as f:
        payload = json.load(f)
    return {name: payload["age_stats"][name] for name in payload.get("age_stats", {})}


def load_split_members(results_root: str = "results") -> Dict[str, List[str]]:
    """The subject filename lists recorded by any run's ``split.json``."""
    files = sorted(glob.glob(os.path.join(results_root, "*", "split.json")))
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f).get("subjects", {})
