# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""How much of the learned reconstruction is specific to one undersampling pattern?

Every training run draws a single k-space mask (seeded by ``train.seed``) and uses it for
training, validation and test. That is the standard protocol for unrolled reconstruction
networks -- the original ADMM-Net does the same -- but it means the network has optimized
against one particular sampling operator. This script quantifies the consequence: it takes
a trained checkpoint and re-evaluates it, **without any retraining**, on the same test
slices under mask realizations drawn with different seeds.

The classical baseline is evaluated under the identical masks as a control. It has no
learned parameters, so any variation it shows reflects how intrinsically hard each mask
realization is, which separates "this mask is harder" from "the network only knows its own
mask".

Results are appended to ``results/crossmask.csv`` and summarized on stdout.

Usage::

    python -m src.eval_crossmask --config configs/default.yaml
    python -m src.eval_crossmask --checkpoint results/comparison_xxx/best.pth --mask-seeds 0 1 2 3
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import torch

from .config import load_config
from .engine import build_model, evaluate
from .losses import build_loss
from .masks import build_mask
from .metrics import DATA_RANGE
from .registry import MODEL_REGISTRY
from .utils import append_results_csv, get_device, load_checkpoint

# Importing these registers every model/baseline so checkpoints can be rebuilt.
from . import model as _model  # noqa: F401
from .baselines import zero_filled as _zf, classical_cs as _cs, unet as _unet, ista_net as _ista  # noqa: F401,E501


def _admmnet_checkpoints(results_root: str, experiment: str = "comparison") -> List[str]:
    """Every ADMM-Net ``best.pth`` under the results root for a given experiment."""
    found: List[str] = []
    if not os.path.isdir(results_root):
        return found
    for entry in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, entry, "best.pth")
        if not os.path.exists(path):
            continue
        try:
            cfg = load_checkpoint(path, map_location="cpu")["config"]
        except Exception:
            continue
        model_name = str(cfg.get("model", {}).get("name", ""))
        if model_name.startswith("admmnet") and cfg.get("name") == experiment:
            found.append(path)
    return found


@torch.no_grad()
def evaluate_under_masks(checkpoint_path: str, mask_seeds: List[int],
                         baseline: Optional[str] = "classical_cs_tv",
                         device: Optional[torch.device] = None) -> List[Dict]:
    """Evaluate a checkpoint (and a control baseline) on the test split, mask by mask."""
    from .dataset import build_dataloaders

    device = device or get_device()
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    base_model = MODEL_REGISTRY.build(baseline).to(device).eval() if baseline else None
    loss_fn = build_loss(cfg["loss"])
    loader = build_dataloaders(cfg)["test"]

    size = cfg["data"]["image_size"]
    mask_params = {k: v for k, v in cfg["mask"].items() if k != "name"}
    train_seed = int(cfg["train"]["seed"])

    rows: List[Dict] = []
    for mask_seed in mask_seeds:
        mask, _ = build_mask(cfg["mask"]["name"], (size, size), device,
                            **{**mask_params, "seed": mask_seed})
        common = {
            "run_id": os.path.basename(os.path.dirname(checkpoint_path)),
            "model": cfg["model"]["name"],
            "sampling_ratio": cfg["mask"]["sampling_ratio"],
            "train_seed": train_seed,
            "mask_seed": mask_seed,
            "mask_seen_in_training": mask_seed == train_seed,
        }
        summary = evaluate(model, loader, mask, loss_fn, device, DATA_RANGE)
        rows.append({**common, "method": cfg["model"]["name"],
                     "psnr_real": summary["psnr_real_mean"],
                     "psnr_imag": summary["psnr_imag_mean"],
                     "ssim_real": summary["ssim_real_mean"],
                     "ssim_imag": summary["ssim_imag_mean"]})
        if base_model is not None:
            base_summary = evaluate(base_model, loader, mask, loss_fn, device, DATA_RANGE)
            rows.append({**common, "method": baseline,
                         "psnr_real": base_summary["psnr_real_mean"],
                         "psnr_imag": base_summary["psnr_imag_mean"],
                         "ssim_real": base_summary["ssim_real_mean"],
                         "ssim_imag": base_summary["ssim_imag_mean"]})

        print(f"  mask_seed={mask_seed}"
              f"{' (seen in training)' if mask_seed == train_seed else ''}: "
              f"PSNR(real) {summary['psnr_real_mean']:.2f} dB")
    return rows


def summarize(rows: List[Dict]) -> None:
    """Print the drop in performance when the mask was not the one trained on."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows to summarize")
        return
    for method, group in df.groupby("method"):
        seen = group[group["mask_seen_in_training"]]["psnr_real"].mean()
        unseen = group[~group["mask_seen_in_training"]]["psnr_real"].mean()
        print(f"\n{method}:")
        print(f"  trained-on mask   PSNR(real) {seen:.2f} dB")
        print(f"  unseen masks      PSNR(real) {unseen:.2f} dB")
        print(f"  generalization gap           {seen - unseen:+.2f} dB")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate trained checkpoints under unseen undersampling masks.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Config used to locate the results root.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value")
    p.add_argument("--checkpoint", default=None,
                   help="A single checkpoint (default: every comparison ADMM-Net run).")
    p.add_argument("--mask-seeds", nargs="*", type=int, default=[0, 1, 2, 7, 11],
                   help="Mask seeds to evaluate; seeds other than train.seed are unseen.")
    p.add_argument("--baseline", default="classical_cs_tv",
                   help="Control model evaluated under the same masks ('' to skip).")
    return p.parse_args(argv)


def main(argv=None) -> List[Dict]:
    args = parse_args(argv)
    cfg = load_config(args.config, cli_overrides=args.set)
    results_root = cfg["paths"]["results_root"]

    checkpoints = ([args.checkpoint] if args.checkpoint
                   else _admmnet_checkpoints(results_root))
    if not checkpoints:
        raise SystemExit(f"no ADMM-Net checkpoints found under {results_root}")

    rows: List[Dict] = []
    for path in checkpoints:
        print(f"\n{path}")
        rows += evaluate_under_masks(path, args.mask_seeds,
                                     baseline=args.baseline or None)

    append_results_csv(os.path.join(results_root, "crossmask.csv"), rows)
    summarize(rows)
    print(f"\nwrote {os.path.join(results_root, 'crossmask.csv')}")
    return rows


if __name__ == "__main__":
    main()
