# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Training entry point.

Usage::

    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/experiments/depth_8.yaml --set train.epochs=5

Reads a config, trains the selected model, checkpoints the best-validation weights,
writes a per-run manifest, and appends dataset-wide test/val metrics to a shared
long-format ``runs.csv`` keyed by the config hash.

Runtime / resumability knobs (read from the ``train`` block with code-side defaults
so they do NOT change the config hash / run_id, and therefore never invalidate runs
that were already completed):

* ``early_stop_patience`` (default 15): stop once the validation loss has not
  improved for this many epochs. ``epochs`` stays the upper bound. Set to 0 to
  disable. This is the main runtime lever -- most runs converge well before the
  100-epoch cap.
* ``early_stop_min_delta`` (default 0.0): minimum val-loss decrease to count as an
  improvement.
* ``ckpt_every`` (default 1): how often (in epochs) to write the resumable
  ``last.pth`` checkpoint.

Crash recovery: a full-state ``last.pth`` is written every ``ckpt_every`` epochs.
If a run is restarted (e.g. after a Colab disconnect) and ``last.pth`` exists, training
resumes from the next epoch with the optimizer, best-so-far, history and RNG restored.
``last.pth`` is removed once the run finishes successfully.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from typing import Dict, List

import torch

from .config import load_config
from .dataset import build_dataloaders, build_splits, describe_split
from .engine import build_model, build_optimizer, evaluate, train_one_epoch, undersample
from .losses import build_loss
from .masks import build_mask, sampling_rate
from .metrics import DATA_RANGE, MetricAccumulator, compute_metrics
from .utils import (
    append_results_csv,
    get_device,
    load_train_state,
    restore_rng,
    run_dir,
    run_id,
    save_checkpoint,
    save_train_state,
    set_seed,
    write_manifest,
    zero_filled_recon,
)

import json

# Modules whose import registers models/losses. Imported defensively so the trainer
# works as the project grows (some of these are added in later steps).
_REGISTRATION_MODULES = [
    "losses",
    "model",
    "baselines.zero_filled",
    "baselines.unet",
    "baselines.ista_net",
    "baselines.classical_cs",
]


def _ensure_registrations() -> None:
    pkg = __package__ or "src"
    for suffix in _REGISTRATION_MODULES:
        name = f"{pkg}.{suffix}"
        if importlib.util.find_spec(name) is not None:
            importlib.import_module(name)


@torch.no_grad()
def _zero_filled_metrics(loader, mask, device, data_range: float = DATA_RANGE) -> Dict[str, float]:
    """Reference metrics for the zero-filled reconstruction (no learning)."""
    acc = MetricAccumulator()
    for label in loader:
        label = label.to(device)
        y = undersample(label, mask)
        recon = zero_filled_recon(y)
        acc.update(compute_metrics(recon.clamp(-1, 1), label.clamp(-1, 1), data_range))
    return acc.summary()


@torch.no_grad()
def _per_sample_rows(model, loader, mask, device, cfg: Dict, method: str,
                     split: str = "test", data_range: float = DATA_RANGE) -> List[Dict]:
    """Per-image metric rows for the required scatter plots / Pearson correlations.

    Each test slice contributes one row of per-channel PSNR/SSIM, tagged with the
    method, sampling ratio, seed and a stable sample index (so the baseline and the
    model can be paired sample-by-sample).
    """
    rows: List[Dict] = []
    idx = 0
    for label in loader:
        label = label.to(device)
        y = undersample(label, mask)
        recon = model(y, mask)
        m = compute_metrics(recon.clamp(-1, 1), label.clamp(-1, 1), data_range)
        bs = label.size(0)
        for i in range(bs):
            rows.append({
                "run_id": run_id(cfg),
                "name": cfg.get("name"),
                "method": method,
                "mask": cfg["mask"].get("name"),
                "sampling_ratio": cfg["mask"].get("sampling_ratio"),
                "seed": cfg["train"].get("seed"),
                "split": split,
                "sample_index": idx + i,
                "psnr_real": float(m["psnr_real"][i]),
                "psnr_imag": float(m["psnr_imag"][i]),
                "ssim_real": float(m["ssim_real"][i]),
                "ssim_imag": float(m["ssim_imag"][i]),
                "nmse": float(m["nmse"][i]),
            })
        idx += bs
    return rows


def _result_rows(cfg: Dict, split: str, summary: Dict[str, float], method: str) -> List[Dict]:
    """Flatten a metric summary into long-format rows for the results CSV."""
    base = {
        "run_id": run_id(cfg),
        "name": cfg.get("name"),
        "method": method,
        "model": cfg["model"].get("name"),
        "num_stages": cfg["model"].get("num_stages"),
        "channels": cfg["model"].get("channels"),
        "share_weights": cfg["model"].get("share_weights"),
        "use_bn": cfg["model"].get("use_bn"),
        "use_residual": cfg["model"].get("use_residual"),
        "rho_init": cfg["model"].get("rho_init"),
        "mask": cfg["mask"].get("name"),
        "sampling_ratio": cfg["mask"].get("sampling_ratio"),
        "loss": cfg["loss"].get("name"),
        "epochs": cfg["train"].get("epochs"),
        "lr": cfg["train"].get("lr"),
        "optimizer": cfg["train"].get("optimizer"),
        "weight_decay": cfg["train"].get("weight_decay"),
        "seed": cfg["train"].get("seed"),
        "split": split,
    }
    rows = []
    for metric in ("psnr_real", "psnr_imag", "ssim_real", "ssim_imag", "nmse"):
        if f"{metric}_mean" in summary:
            rows.append({**base, "metric": metric,
                         "value": summary[f"{metric}_mean"],
                         "std": summary.get(f"{metric}_std")})
    return rows


def _epochs_since_best(history: List[Dict], min_delta: float) -> int:
    """Replay the val-loss history to recover the early-stopping counter on resume."""
    running_best = float("inf")
    no_improve = 0
    for record in history:
        if record["val_loss"] < running_best - min_delta:
            running_best = record["val_loss"]
            no_improve = 0
        else:
            no_improve += 1
    return no_improve


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an MRI reconstruction model.")
    p.add_argument("--config", required=True, help="Path to a YAML config.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                   help="Dotted config overrides.")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, float]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)

    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"].get("prefer_cuda", True))
    if device.type == "cuda":
        # Free speedup for fixed-size inputs (all slices are image_size x image_size).
        torch.backends.cudnn.benchmark = True
    _ensure_registrations()

    loaders = build_dataloaders(cfg)
    size = cfg["data"]["image_size"]
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    mask_params.setdefault("seed", cfg["train"]["seed"])
    mask, mask_centered = build_mask(cfg["mask"]["name"], (size, size), device, **mask_params)

    model = build_model(cfg, device)
    loss_fn = build_loss(cfg["loss"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = n_params > 0
    print(f"run_id={run_id(cfg)} | device={device} | params={n_params:,} | "
          f"trainable={trainable} | sampling={sampling_rate(mask_centered)*100:.1f}%")

    rdir = run_dir(cfg)

    # Record the custom age-stratified split (professor's update): which subjects
    # landed in each split and their age statistics, so the report can show the
    # distributions match and document that we built the split ourselves.
    try:
        splits, meta, data_cfg = build_splits(cfg)
        split_info = {
            "age_stats": describe_split(meta, splits, data_cfg),
            "subjects": {k: [os.path.basename(p) for p in v] for k, v in splits.items()},
        }
        with open(os.path.join(rdir, "split.json"), "w") as f:
            json.dump(split_info, f, indent=2)
    except Exception as exc:  # never let bookkeeping abort a training run
        print(f"warning: could not record split summary ({exc})")

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop_patience", 15))
    min_delta = float(cfg["train"].get("early_stop_min_delta", 0.0))
    ckpt_every = max(1, int(cfg["train"].get("ckpt_every", 1)))
    best_path = os.path.join(rdir, "best.pth")
    last_path = os.path.join(rdir, "last.pth")

    history, best_val = [], float("inf")
    start_epoch, epochs_no_improve = 1, 0

    # Parameter-free baselines (zero-filled, classical CS) are evaluation-only.
    if trainable:
        optimizer = build_optimizer(model, cfg["train"])

        # Resume from the last checkpoint if a previous attempt crashed mid-training.
        if os.path.exists(last_path):
            state = load_train_state(last_path, map_location=device)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            history = state.get("history", [])
            best_val = state.get("best_val", float("inf"))
            start_epoch = int(state.get("epoch", 0)) + 1
            # Reconstruct the early-stopping counter from history so that resuming
            # behaves identically to an uninterrupted run.
            epochs_no_improve = _epochs_since_best(history, min_delta)
            try:
                restore_rng(state["rng"])
            except Exception as exc:  # resume is still valid without exact RNG state
                print(f"warning: could not restore RNG state ({exc})")
            print(f"resuming from epoch {start_epoch}/{epochs} (best_val={best_val:.6f})")

        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], mask, loss_fn, optimizer, device)
            val = evaluate(model, loaders["val"], mask, loss_fn, device)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val["loss"], "val_psnr": val.get("psnr_real_mean")})

            improved = val["loss"] < best_val - min_delta
            if improved:
                best_val = val["loss"]
                epochs_no_improve = 0
                save_checkpoint(best_path, model, cfg)
            else:
                epochs_no_improve += 1

            # Periodic full-state checkpoint for crash recovery (atomic write).
            if epoch % ckpt_every == 0 or epoch == epochs:
                save_train_state(last_path, model, optimizer, epoch, best_val, history, cfg)

            print(f"epoch {epoch:3d}/{epochs} | train {train_loss:.6f} | "
                  f"val {val['loss']:.6f} | val PSNR {val.get('psnr_real_mean', float('nan')):.2f}"
                  f"{' *' if improved else ''}")

            if patience > 0 and epochs_no_improve >= patience:
                print(f"early stopping at epoch {epoch} "
                      f"(no val-loss improvement for {patience} epochs)")
                break
    else:
        print("no trainable parameters: skipping training (evaluation-only baseline)")

    # Final evaluation with the best checkpoint (if one was saved).
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    val = evaluate(model, loaders["val"], mask, loss_fn, device)
    test = evaluate(model, loaders["test"], mask, loss_fn, device)
    zf_test = _zero_filled_metrics(loaders["test"], mask, device)

    metrics = {
        "n_params": n_params,
        "sampling_rate": sampling_rate(mask_centered),
        "val": val,
        "test": test,
        "zero_filled_test": zf_test,
    }
    write_manifest(rdir, cfg, metrics)

    # Per-sample metrics on the test set (this model + the zero-filled reference) for
    # the required sample-wise scatter plots / Pearson correlations.
    sample_rows = _per_sample_rows(model, loaders["test"], mask, device, cfg,
                                   method=cfg["model"]["name"], split="test")

    class _ZF(torch.nn.Module):
        def forward(self, y, m):
            return zero_filled_recon(y)

    sample_rows += _per_sample_rows(_ZF().to(device), loaders["test"], mask, device, cfg,
                                    method="zero_filled", split="test")
    append_results_csv(os.path.join(cfg["paths"]["results_root"], "samples.csv"), sample_rows)

    # runs.csv is written LAST because the sweep runner treats a run_id appearing there as
    # proof the run finished. Writing it first would mark a run complete even if the
    # per-sample logging was interrupted, and the sweep would then never redo it.
    rows = (
        _result_rows(cfg, "val", val, method=cfg["model"]["name"])
        + _result_rows(cfg, "test", test, method=cfg["model"]["name"])
        + _result_rows(cfg, "test", zf_test, method="zero_filled")
    )
    append_results_csv(os.path.join(cfg["paths"]["results_root"], "runs.csv"), rows)

    # The run is complete and logged: drop the resume checkpoint to save space.
    # (best.pth and manifest.json remain.)
    if os.path.exists(last_path):
        os.remove(last_path)

    print(f"DONE | test PSNR(real) {test.get('psnr_real_mean'):.2f} (zero-filled "
          f"{zf_test.get('psnr_real_mean'):.2f}) | results -> {rdir}")
    return metrics


if __name__ == "__main__":
    main()
