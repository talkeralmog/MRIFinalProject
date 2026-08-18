# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Shared utilities: seeding, run identification, checkpoint/manifest io, FFT helpers.

These helpers are deliberately framework-light so they can be reused by the training
script, the experiment runner, and the analysis notebook alike.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device when available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def config_hash(cfg: Dict[str, Any], length: int = 10) -> str:
    """Stable short hash of a config, used to key runs uniquely.

    Paths are excluded so that the same experiment on Colab vs local maps to the
    same hash.
    """
    filtered = {k: v for k, v in cfg.items() if k != "paths"}
    payload = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def run_id(cfg: Dict[str, Any]) -> str:
    """Human-readable run id combining the experiment name and a config hash."""
    name = cfg.get("name", cfg.get("model", {}).get("name", "run"))
    return f"{name}_{config_hash(cfg)}"


def run_dir(cfg: Dict[str, Any]) -> str:
    """Directory for a run's artifacts, created under the results root."""
    path = os.path.join(cfg["paths"]["results_root"], run_id(cfg))
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(path: str, model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    """Save model weights together with the config that produced them."""
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_checkpoint(path: str, map_location: Optional[Any] = None) -> Dict[str, Any]:
    """Load a checkpoint dict ({'state_dict', 'config'})."""
    return torch.load(path, map_location=map_location)


def save_train_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    history: list,
    cfg: Dict[str, Any],
) -> None:
    """Save the full training state for crash-resumable training (the 'last' ckpt).

    Unlike ``save_checkpoint`` (weights + config, used for the best model), this
    captures everything needed to resume mid-run: optimizer state, the last
    completed epoch, the best validation loss so far, the epoch history, and the
    RNG states. The write is atomic (temp file + ``os.replace``) so a crash during
    the save itself cannot corrupt an existing checkpoint.
    """
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
        "config": cfg,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_train_state(path: str, map_location: Optional[Any] = None) -> Dict[str, Any]:
    """Load a full training-state checkpoint written by ``save_train_state``."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(rng: Dict[str, Any]) -> None:
    """Restore Python/NumPy/PyTorch RNG states captured by ``save_train_state``.

    RNG state tensors must live on the CPU regardless of how the checkpoint was
    mapped on load, so they are coerced back to CPU uint8 here.
    """
    torch_state = rng["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.cpu().to(torch.uint8)
    torch.set_rng_state(torch_state)
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def write_manifest(directory: str, cfg: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Write a JSON manifest (config + metrics + timestamp) for a run."""
    manifest = {
        "run_id": run_id(cfg),
        "config": cfg,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(directory, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# FFT helpers
#
# We follow the convention used throughout the project: ``torch.fft.fft2`` is the
# (non-centered) forward transform, and undersampling masks are stored in the same
# non-centered layout (see masks.py). Centering with fftshift is only applied for
# visualization.
# ---------------------------------------------------------------------------


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """2D FFT over the last two dims (non-centered)."""
    return torch.fft.fft2(x)


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """2D inverse FFT over the last two dims (non-centered)."""
    return torch.fft.ifft2(x)


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Take the real part of a complex tensor."""
    return torch.real(x)


# ---------------------------------------------------------------------------
# Complex <-> 2-channel real conversions
#
# MRI images are complex-valued (magnitude + phase). Throughout the pipeline we
# represent an image as a 2-channel real tensor ``(B, 2, H, W)`` -- channel 0 is
# the real part, channel 1 the imaginary part -- because PyTorch convolutions and
# the project's losses/metrics operate naturally on real tensors, and the brief
# asks for PSNR/SSIM on the real and imaginary components separately. k-space stays
# a genuine complex tensor ``(B, 1, H, W)`` for the FFT / data-consistency steps.
# ---------------------------------------------------------------------------


def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    """Complex ``(B, 1, H, W)`` -> real ``(B, 2, H, W)`` (real, imag) channels."""
    return torch.cat([x.real, x.imag], dim=1)


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Real ``(B, 2, H, W)`` (real, imag) channels -> complex ``(B, 1, H, W)``."""
    return torch.complex(x[:, 0:1].contiguous(), x[:, 1:2].contiguous())


def zero_filled_recon(undersampled_kspace: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction as a 2-channel (real, imag) image.

    The inverse FFT of the zero-filled k-space; the imaginary part is retained
    (not discarded) so downstream complex metrics are well defined.
    """
    return complex_to_chan(ifft2c(undersampled_kspace))


def append_results_csv(path: str, rows: list) -> None:
    """Append rows (list of dicts) to a long-format results CSV, creating it if new.

    Rows already logged under the same ``run_id`` are replaced rather than duplicated, so
    re-running a run (after a crash, or to regenerate its per-sample metrics) leaves one
    consistent record instead of double-counting it in every aggregate.
    """
    import pandas as pd

    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if "run_id" in existing.columns and "run_id" in df_new.columns:
            existing = existing[~existing["run_id"].isin(set(df_new["run_id"]))]
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
