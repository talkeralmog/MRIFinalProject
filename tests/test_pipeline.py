# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Tests that verify the MRI pipeline without needing the dataset.

The medical images cannot be committed, so this suite runs on a Shepp-Logan phantom (the
standard MRI test object, bundled with scikit-image) and on synthetic tensors. It checks
the properties the report actually claims: that the undersampling mask follows the course
specification, that the forward model and data-consistency step are consistent, that the
classical baseline genuinely improves on the zero-filled reconstruction, and that run ids
are reproducible.

Run from the project root::

    pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.baselines.classical_cs import ClassicalCS, ClassicalCSWaveletTV, tv_denoise
from src.masks import build_mask, gaussian1d_mask, sampling_rate
from src.metrics import DATA_RANGE, compute_metrics, convert_psnr, psnr_channel
from src.model import ADMMNet, ADMMNetPWL
from src.utils import chan_to_complex, complex_to_chan, config_hash, fft2c, ifft2c

SIZE = 128
RATIOS = (0.2, 0.3, 0.5)


@pytest.fixture(scope="module")
def phantom() -> torch.Tensor:
    """A complex-valued phantom slice, normalized exactly like the dataset pipeline."""
    img = resize(shepp_logan_phantom().astype(np.float32), (SIZE, SIZE), anti_aliasing=True)
    rows = np.arange(SIZE)[:, None]
    volume = img * np.exp(1j * 0.5 * np.sin(2 * np.pi * rows / SIZE))  # non-trivial phase
    volume = volume / np.abs(volume).max()
    return complex_to_chan(torch.from_numpy(volume.astype(np.complex64))[None, None])


def _undersample(label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask * fft2c(chan_to_complex(label))


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(-1, 1)
    return float((psnr_channel(pred[:, 0], target[:, 0])
                  + psnr_channel(pred[:, 1], target[:, 1])).item() / 2)


# ---------------------------------------------------------------------------
# Undersampling mask: the course specification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
def test_mask_keeps_the_requested_fraction_of_rows(ratio):
    mask = gaussian1d_mask((SIZE, SIZE), sampling_ratio=ratio, seed=0)
    assert mask.shape == (SIZE, SIZE)
    assert sampling_rate(mask) == pytest.approx(ratio, abs=1.5 / SIZE)


@pytest.mark.parametrize("ratio", RATIOS)
def test_mask_samples_whole_phase_encode_rows(ratio):
    """A 1D Cartesian pattern skips entire readout lines, never individual pixels."""
    mask = gaussian1d_mask((SIZE, SIZE), sampling_ratio=ratio, seed=0)
    for row in mask:
        assert row.min() == row.max()


@pytest.mark.parametrize("ratio", RATIOS)
def test_mask_rows_are_drawn_without_replacement(ratio):
    """Exactly round(ratio * rows) distinct rows, i.e. no row is drawn twice."""
    mask = gaussian1d_mask((SIZE, SIZE), sampling_ratio=ratio, seed=0)
    assert int(mask[:, 0].sum()) == int(round(ratio * SIZE))


def test_mask_is_variable_density_around_the_centre_of_kspace():
    """Low frequencies must be sampled far more often than high frequencies."""
    centre, edge = 0.0, 0.0
    realizations = 200
    for seed in range(realizations):
        mask = gaussian1d_mask((SIZE, SIZE), sampling_ratio=0.3, seed=seed)[:, 0]
        centre += mask[SIZE // 2 - 16:SIZE // 2 + 16].mean()
        edge += np.concatenate([mask[:16], mask[-16:]]).mean()
    assert centre / realizations > 2 * edge / realizations


def test_mask_is_reproducible_from_its_seed():
    a = gaussian1d_mask((SIZE, SIZE), sampling_ratio=0.3, seed=5)
    b = gaussian1d_mask((SIZE, SIZE), sampling_ratio=0.3, seed=5)
    c = gaussian1d_mask((SIZE, SIZE), sampling_ratio=0.3, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Forward model and data consistency
# ---------------------------------------------------------------------------


def test_forward_model_is_invertible_when_nothing_is_masked(phantom):
    """With a fully sampled mask the zero-filled reconstruction is the original slice."""
    mask = torch.ones(1, 1, SIZE, SIZE)
    recon = complex_to_chan(ifft2c(_undersample(phantom, mask)))
    assert torch.allclose(recon, phantom, atol=1e-5)


def test_zero_filling_leaves_the_measured_lines_untouched(phantom):
    """The acquired k-space rows must survive the forward/inverse round trip exactly."""
    mask, _ = build_mask("gaussian1d", (SIZE, SIZE), sampling_ratio=0.3, seed=0)
    y = _undersample(phantom, mask)
    recon = complex_to_chan(ifft2c(y))
    assert torch.allclose(mask * fft2c(chan_to_complex(recon)), y, atol=1e-4)


@pytest.mark.parametrize("model_cls", [ADMMNet, ADMMNetPWL])
def test_unrolled_models_return_a_two_channel_image(model_cls, phantom):
    mask, _ = build_mask("gaussian1d", (SIZE, SIZE), sampling_ratio=0.3, seed=0)
    model = model_cls(num_stages=2, channels=8).eval()
    with torch.no_grad():
        out = model(_undersample(phantom, mask), mask)
    assert out.shape == phantom.shape
    assert torch.isfinite(out).all()


def test_weight_sharing_reduces_the_parameter_count():
    independent = ADMMNet(num_stages=4, channels=16, share_weights=False)
    shared = ADMMNet(num_stages=4, channels=16, share_weights=True)
    n_independent = sum(p.numel() for p in independent.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    assert n_shared == pytest.approx(n_independent / 4, rel=0.02)


# ---------------------------------------------------------------------------
# Total-variation proximal operator
# ---------------------------------------------------------------------------


def test_tv_denoise_is_a_no_op_at_zero_weight():
    x = torch.randn(2, 2, 32, 32)
    assert torch.equal(tv_denoise(x, weight=0.0), x)


def test_tv_denoise_reduces_total_variation():
    x = torch.randn(1, 1, 64, 64) * 0.2
    out = tv_denoise(x, weight=0.05, num_iters=50)

    def total_variation(t):
        return (t[..., 1:, :] - t[..., :-1, :]).abs().sum() + \
               (t[..., 1:] - t[..., :-1]).abs().sum()

    assert total_variation(out) < total_variation(x)


def test_tv_denoise_matches_the_reference_implementation():
    """Our torch implementation must agree with skimage's Chambolle solver."""
    from skimage.restoration import denoise_tv_chambolle

    rng = np.random.default_rng(0)
    img = (rng.normal(size=(64, 64)) * 0.2).astype(np.float32)
    mine = tv_denoise(torch.from_numpy(img)[None, None], weight=0.02,
                      num_iters=200)[0, 0].numpy()
    reference = denoise_tv_chambolle(img, weight=0.02, max_num_iter=200, eps=0.0)
    assert np.abs(mine - reference).max() < 1e-5


# ---------------------------------------------------------------------------
# The classical baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
def test_tuned_baseline_improves_on_zero_filling(phantom, ratio):
    """The reported baseline must actually reconstruct, not just return its input."""
    mask, _ = build_mask("gaussian1d", (SIZE, SIZE), sampling_ratio=ratio, seed=0)
    y = _undersample(phantom, mask)
    zero_filled = complex_to_chan(ifft2c(y))
    reconstructed = ClassicalCSWaveletTV()(y, mask)
    assert _psnr(reconstructed, phantom) > _psnr(zero_filled, phantom) + 1.0


def test_single_level_wavelet_prior_is_too_weak_to_help(phantom):
    """Documents the finding behind `classical_cs_tv`.

    A one-level Haar transform only exposes the finest detail coefficients, so the
    iteration converges back onto the zero-filled solution. This is why the report's
    baseline uses a multi-level transform plus a TV prior.
    """
    mask, _ = build_mask("gaussian1d", (SIZE, SIZE), sampling_ratio=0.3, seed=0)
    y = _undersample(phantom, mask)
    zero_filled = complex_to_chan(ifft2c(y))
    naive = ClassicalCS(num_iters=50, lam=0.02, wavelet_levels=1)(y, mask)
    assert _psnr(naive, phantom) < _psnr(zero_filled, phantom) + 0.5


def test_baseline_output_is_data_consistent(phantom):
    """The final POCS step must leave the acquired k-space rows exactly as measured."""
    mask, _ = build_mask("gaussian1d", (SIZE, SIZE), sampling_ratio=0.3, seed=0)
    y = _undersample(phantom, mask)
    recon = ClassicalCSWaveletTV(num_iters=5)(y, mask)
    assert torch.allclose(mask * fft2c(chan_to_complex(recon)), y, atol=1e-3)


def test_baseline_has_no_trainable_parameters():
    assert sum(p.numel() for p in ClassicalCSWaveletTV().parameters()) == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_are_reported_per_channel(phantom):
    metrics = compute_metrics(phantom, phantom)
    assert set(metrics) == {"psnr_real", "psnr_imag", "ssim_real", "ssim_imag", "nmse"}
    assert metrics["ssim_real"].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["nmse"].item() == pytest.approx(0.0, abs=1e-9)


def test_psnr_decreases_as_the_error_grows(phantom):
    noisier = phantom + 0.05 * torch.randn_like(phantom)
    noisiest = phantom + 0.20 * torch.randn_like(phantom)
    assert _psnr(noisier, phantom) > _psnr(noisiest, phantom)


def test_psnr_convention_conversion_is_exactly_six_db():
    """Restating PSNR with peak 1.0 instead of 2.0 costs 20*log10(2) dB."""
    assert convert_psnr(40.0, from_range=DATA_RANGE, to_range=1.0) == pytest.approx(
        40.0 - 20 * math.log10(2), abs=1e-9)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_run_id_ignores_machine_specific_paths():
    """The same experiment must map to the same run id locally and on the HPC."""
    base = {"name": "x", "model": {"name": "admmnet_softthresh"}, "train": {"seed": 0}}
    local = {**base, "paths": {"data_root": "/truenas/home/user/data"}}
    colab = {**base, "paths": {"data_root": "/content/drive/data"}}
    assert config_hash(local) == config_hash(colab)


def test_run_id_changes_with_the_experiment_configuration():
    a = {"model": {"name": "admmnet_softthresh", "num_stages": 8}}
    b = {"model": {"name": "admmnet_softthresh", "num_stages": 5}}
    assert config_hash(a) != config_hash(b)


# ---------------------------------------------------------------------------
# Aggregation of the logged results
# ---------------------------------------------------------------------------


def _fake_samples() -> pd.DataFrame:
    """Per-sample rows shaped like samples.csv, including the two traps it contains."""
    rows = []
    for experiment, psnr in (("comparison", 40.0), ("depth_sweep", 20.0)):
        for sample in range(3):
            rows.append({"name": experiment, "method": "admmnet_softthresh",
                         "sampling_ratio": 0.3, "seed": 0, "split": "test",
                         "sample_index": sample, "psnr_real": psnr, "psnr_imag": psnr,
                         "ssim_real": 0.9, "ssim_imag": 0.9})
    # The zero-filled reference is logged once per run, so identical rows repeat.
    for _ in range(2):
        for sample in range(3):
            rows.append({"name": "comparison", "method": "zero_filled",
                         "sampling_ratio": 0.3, "seed": 0, "split": "test",
                         "sample_index": sample, "psnr_real": 22.0, "psnr_imag": 22.0,
                         "ssim_real": 0.4, "ssim_imag": 0.4})
    return pd.DataFrame(rows)


def test_aggregation_excludes_other_experiments():
    """Ablations also run at ratio 0.3, so they must not leak into headline numbers."""
    from src.analysis import combined_samples

    d = combined_samples(_fake_samples(), "psnr", experiment="comparison")
    model = d[d["method"] == "admmnet_softthresh"]
    assert len(model) == 3
    assert model["value"].mean() == pytest.approx(40.0)


def test_aggregation_drops_repeated_reference_rows():
    """The zero-filled reference appears once per run; duplicates must not be counted."""
    from src.analysis import select_samples

    d = select_samples(_fake_samples(), experiment="comparison")
    assert len(d[d["method"] == "zero_filled"]) == 3


def test_relogging_a_run_replaces_its_rows(tmp_path):
    """Re-running a run must leave one record, not double-count it in the aggregates."""
    from src.utils import append_results_csv

    path = str(tmp_path / "runs.csv")
    append_results_csv(path, [{"run_id": "a", "metric": "psnr_real", "value": 1.0},
                              {"run_id": "b", "metric": "psnr_real", "value": 2.0}])
    append_results_csv(path, [{"run_id": "a", "metric": "psnr_real", "value": 9.0}])

    df = pd.read_csv(path)
    assert len(df) == 2
    assert df[df["run_id"] == "a"]["value"].tolist() == [9.0]


def test_ablations_do_not_mix_training_budgets():
    """A variant re-run under a different epoch budget must not be averaged with the rest."""
    from src.analysis import comparable_runs

    df = pd.DataFrame([
        {"name": "depth_sweep", "run_id": "a", "epochs": 100, "num_stages": 1},
        {"name": "depth_sweep", "run_id": "b", "epochs": 100, "num_stages": 3},
        {"name": "depth_sweep", "run_id": "c", "epochs": 100, "num_stages": 8},
        {"name": "depth_sweep", "run_id": "stale", "epochs": 40, "num_stages": 1},
    ])
    with pytest.warns(UserWarning, match="epochs"):
        kept = comparable_runs(df, "depth_sweep")
    assert set(kept["run_id"]) == {"a", "b", "c"}


def test_a_run_missing_its_per_sample_rows_is_not_treated_as_complete(tmp_path):
    """Otherwise a run interrupted between the two logs would be skipped forever."""
    from src.run_experiments import completed_run_ids

    pd.DataFrame({"run_id": ["finished", "half_logged"]}).to_csv(
        tmp_path / "runs.csv", index=False)
    pd.DataFrame({"run_id": ["finished"]}).to_csv(tmp_path / "samples.csv", index=False)

    assert completed_run_ids(str(tmp_path / "runs.csv")) == {"finished"}
