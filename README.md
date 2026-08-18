# MRI Reconstruction from Subsampled k-space

**Course:** Magnetic Resonance Imaging (361.2.6501) - Final Project, Task 3.2
**Authors:** Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)

Reconstruct anatomically faithful MRI images from **undersampled k-space**. A main
bottleneck of MRI is scan time: acquiring a full k-space volume takes minutes. Sampling
only a fraction of the phase-encode lines accelerates acquisition, but violates the
Nyquist criterion and introduces **aliasing artifacts** in the zero-filled
reconstruction. This project restores the fully sampled image from such partial,
corrupted acquisitions and compares two reconstruction models.

## The two models

| Role | Model | Idea |
|------|-------|------|
| **Baseline** (classical, non-DL) | `classical_cs_tv` | **Compressed sensing**: iterate a multi-level Haar-wavelet soft-threshold and a total-variation proximal step (sparsity priors) with k-space **data consistency** (POCS) on the acquired lines. Physically grounded, training-free; the two regularization weights are calibrated on the validation split. |
| **Our model** (learned) | `admmnet_softthresh` | **Unrolled ADMM-Net**: the same CS optimization (sparsity prior + data consistency), but unrolled into a fixed number of stages whose proximal operator and data-consistency penalty are **learned**. A learned generalization of the baseline. |

Two references are logged in every run for context: the trivial **zero-filled**
reconstruction (inverse FFT of the masked k-space), which is the aliased model input, and
`classical_cs` — the *single-level* wavelet variant of the baseline. The latter is kept
deliberately: a one-level Haar transform is too weak a prior to reconstruct anything (it
gains +0.13 dB over zero-filling and no threshold value helps), so it serves as the report's
prior ablation and shows why the baseline needs coarser scales and a TV term.

## MRI-specific choices

- **Undersampling mask** (`gaussian1d`): a 1D variable-density Cartesian mask. Phase-encode
  lines are drawn from a **normal distribution centered on the middle of k-space**, so the
  low frequencies (most of the image energy) are densely sampled and the high frequencies
  are sparse. We keep **20% / 30% / 50%** of lines, sampled **without replacement**, seeded
  for reproducibility (`src/masks.py`).
- **Complex-valued data**: MRI images are complex (magnitude + phase). The pipeline carries
  the image as a 2-channel (real, imag) tensor end to end; the forward model and data
  consistency use a genuine complex FFT.
- **Metrics**: PSNR and SSIM computed on the **real and imaginary components separately**,
  after a consistent normalization (scale by the global maximum magnitude, so each channel
  lies in [-1, 1]). Mean and standard deviation are reported **across the test set**, as the
  brief requires, from the per-slice metrics in `samples.csv`.
- **PSNR peak convention**: PSNR uses `peak = 2.0`, the width of the [-1, 1] channel range.
  A central slice has an RMS magnitude near 0.2, so this peak is about ten times the typical
  signal and all absolute dB values sit higher than ones computed from a per-image peak. The
  convention is identical for every method and ratio, so comparisons are unaffected;
  subtracting 20*log10(2) = 6.02 dB restates any value with `peak = 1.0`
  (`src/metrics.convert_psnr`). Both appear in `report/tables.md`.

## Dataset handling (`src/dataset.py`)

The course "Reconstruction" dataset is complex-valued 3D scans in `.npy`, with per-scan
age/sex metadata in CSV files. Per the professor's update, the provided `test` CSV lists
subjects that are **not** present in the selected NumPy directory, so we build our own
split:

1. Read the three provided CSVs (train/val/test) and concatenate them into one master table.
2. Discover the `.npy` volumes actually on disk.
3. **Cross-reference**: keep only subjects that have a matching `.npy` file (this resolves
   the mismatch).
4. Build an explicit, reproducible **age-stratified 3-way split** (train/val/test) by
   binning age into quantiles and splitting within each bin, so all three splits share
   approximately the same age distribution. Each run records its split (subjects + age
   statistics) to `results/<run_id>/split.json`.

The split is **fixed** via `data.split_seed` (independent of `train.seed`), so every run
and every seed sees the **same** train/val/test partition. With `data.split: [0.8, 0.1, 0.1]`
this yields **3833 / 479 / 479 subjects**, with mean ages 26.3 / 26.3 / 26.0 years and
effectively identical ranges, and it lets the per-sample scatter plots pair the baseline and
model sample-for-sample. The three seeds vary only the k-space mask realization and the
network initialization.

**Central slice only.** By default (`data.central_slice_only: true`) exactly one slice --
the anatomical middle slice -- is kept per subject, which the course explicitly permits.
The number of samples per split therefore equals the number of subjects, keeping the test
set at a legitimate ~10% while cutting training volume ~32x. Each kept slice is
intensity-normalized by its volume's global maximum magnitude. Set
`central_slice_only: false` to fall back to a band of central slices
(`central_fraction` / `max_slices_per_subject`).

### Slice cache (run this once)

Reading ~5000 volumes from the networked filesystem on every run is the main runtime
bottleneck. Build the cache **once**; all subsequent runs load pre-extracted slices
instantly (no network reads):

```bash
python -m src.build_cache --config configs/default.yaml \
    --set paths.local.data_root=../MRI_2026_datasets/brain_age
```

The cache is written under `paths.local.cache_root` (default `./cache`) and keyed by a
fingerprint of the split + preprocessing, so it is reused only when it matches exactly.
Because the split is fixed, one cache serves the entire sweep. Missing caches are also
built automatically on the first run that needs them.

**Optional speed lever:** cap the number of training subjects (age-stratified, test/val
untouched) via `data.max_train_subjects` in `configs/default.yaml`.

## Installation

```bash
pip install -r requirements.txt
```

## Data layout

The dataset path is set in `configs/default.yaml` (`paths.local.data_root`). On the HPC
server the data lives under `MRI_2026_datasets/brain_age/`. Do **not** commit the data.

```
MRI_2026_datasets/brain_age/     # = paths.local.data_root
├── selected_npy/                # complex 3D volumes, one .npy per subject
├── student_train_metadata.csv   # metadata (subject_id, age, sex, ...)
├── student_val_metadata.csv
└── student_test_metadata.csv
```

`data_root` is a relative path; if you launch from a different working directory, point
it at the dataset without editing files:

```bash
python -m src.train --config configs/default.yaml \
    --set paths.local.data_root=/absolute/or/relative/path/to/MRI_2026_datasets/brain_age
```

The CSV column names and the id-to-filename mapping are configurable
(`data.id_col`, `data.age_col`, `data.npy_name_template`, `data.numpy_dir`, `data.meta_csvs`).

## Usage

### Runbook (the exact order to reproduce the report)

```bash
# 0. Tests: verify the pipeline without touching the dataset (seconds)
pytest -q

# 1. Build the slice cache once
python -m src.build_cache --config configs/default.yaml

# 2. Calibrate the classical baseline on the VALIDATION split (evaluation-only, minutes).
#    Then read table 4 of report/tables.md and copy the winning (lam, tv_weight) into
#    TUNED_LAM / TUNED_TV_WEIGHT at the top of src/baselines/classical_cs.py.
python -m src.run_experiments --sweep configs/experiments/baseline_tuning.yaml

# 3. Headline comparison: our model, then the calibrated baseline at the same ratios/seeds
nohup python -m src.run_experiments --sweep configs/experiments/comparison.yaml \
    > logs/run_comparison.log 2>&1 &
python -m src.run_experiments --sweep configs/experiments/comparison_baseline_tv.yaml

# 4. Appendix ablations (unrolling depth, loss, architecture, U-Net reference)
python -m src.run_experiments --sweep configs/experiments/depth_sweep.yaml
python -m src.run_experiments --sweep configs/experiments/loss_ablation.yaml
python -m src.run_experiments --sweep configs/experiments/structure_ablation.yaml
python -m src.run_experiments --sweep configs/experiments/unet_reference.yaml

# 5. Appendix: do the trained networks generalize to masks they never saw? (no retraining)
python -m src.eval_crossmask --config configs/default.yaml

# 6. Figures and tables
python -m src.figures --config configs/default.yaml            # method / data / results
python -m src.figures_mri --config configs/default.yaml        # MRI-principle figures
python -m src.figures_results --config configs/default.yaml    # required plots, per channel
python -m src.figures_code                                     # source listings for the report
python -m src.make_qualitative --ratios 0.2 0.3 0.5 --seed 0   # the four worked examples
python -m src.eval_contrast --ratios 0.2 0.3 0.5 --seed 0      # WM/GM contrast and CNR
python -m src.make_report_tables --config configs/default.yaml

# 7. The report itself (PDF + DOCX + a plain-text dump)
python -m src.build_report --github https://github.com/<user>/<repo>
```

Only the sweeps in steps 3 and 4 train; steps 2, 5 and 6 are evaluation-only. Train a single
model directly with:

```bash
python -m src.train --config configs/default.yaml \
    --set model.name=admmnet_softthresh mask.sampling_ratio=0.3
```

Each run writes a checkpoint, a `manifest.json`, its `split.json`, per-sample metrics to
`results/samples.csv`, and finally dataset-wide metrics to `results/runs.csv`. Sweeps are
resumable: a run is skipped only when its config hash appears in **both** CSVs, so a run
interrupted between the two writes is redone rather than silently left half-logged.
Re-logging a run replaces its rows instead of duplicating them.

## Tests

`tests/test_pipeline.py` runs in a few seconds and needs **no dataset** — it uses a
Shepp-Logan phantom and synthetic tensors — so the pipeline can be verified on any machine:

```bash
pytest -q
```

It checks the properties the report claims: the mask follows the brief (whole phase-encode
rows, exactly the requested fraction, drawn without replacement, variable density, seeded),
the forward model and POCS step keep the acquired lines exact, the TV proximal operator
agrees with `skimage`'s reference solver, the classical baseline genuinely improves on
zero-filling (and the single-level variant demonstrably does not), and run ids are
reproducible and path-independent.

## Figures, tables and the report

Everything the report embeds is generated by code, and the report itself is generated too:
`src/build_report.py` reads every number it quotes from the logged results, so the prose,
the tables and the figures cannot disagree or go stale.

```bash
python -m src.figures --config configs/default.yaml            # method / data figures
python -m src.figures_mri --config configs/default.yaml        # MRI-principle figures
python -m src.figures_results --config configs/default.yaml    # the brief's required plots
python -m src.figures_code                                     # syntax-highlighted listings
python -m src.make_qualitative --ratios 0.2 0.3 0.5 --seed 0   # the four worked examples
python -m src.eval_contrast --ratios 0.2 0.3 0.5 --seed 0      # WM/GM contrast and CNR
python -m src.make_report_tables --config configs/default.yaml # tables -> report/tables.md
python -m src.build_report --github https://github.com/<user>/<repo>   # PDF + DOCX + txt
```

`src/figures.py` groups its output (`--which masks pipelines results eda per_stage`):

| Group | Needs | Contents |
|-------|-------|----------|
| `masks` | nothing | Sampling patterns at 20/30/50% and the per-row acquisition probability |
| `pipelines` | nothing | Block diagrams of the baseline and of our unrolled model |
| `results` | the CSVs | PSNR/SSIM vs sampling ratio, sample-wise scatter with Pearson r, ablations |
| `eda` | the dataset | Example slices, per-split age distribution, full vs undersampled k-space |
| `per_stage` | a checkpoint | The reconstruction after each unrolled ADMM stage, with error maps |

The remaining modules need only the **slice cache** (plus a checkpoint where noted), not the
raw volume directory, so the whole report can be rebuilt on a laptop:

| Module | Needs | Contents |
|--------|-------|----------|
| `figures_mri.py` | the cache | k-space energy vs `ky`; PSF and artifact type for three sampling schemes; Hermitian symmetry and the spurious imaginary channel; the scan-time / SNR trade-off; contrast weighting; the DC-line finding |
| `figures_results.py` | the CSVs | PSNR/SSIM vs ratio and the sample-wise scatter plots, **real and imaginary channels separately** |
| `make_qualitative.py` | cache + checkpoint | The four required example categories, with k-space and error maps |
| `eval_contrast.py` | cache + checkpoint | White-matter/grey-matter contrast and CNR retention against the fully sampled reference |
| `figures_code.py` | nothing | Source listings pulled from `src/` by function name, so they cannot go stale |
| `build_report.py` | the above | `report/MRI_Final_Project_Report.{pdf,docx,txt}` |

`src/make_report_tables.py` additionally renders `report/tables.md` from `runs.csv` /
`samples.csv`.

## Reproducibility

Global seeding (Python/NumPy/PyTorch), a fixed deterministic age-stratified split
(`data.split_seed`, identical across seeds), per-run manifests (config + metrics +
timestamp), config-hash de-duplication, and idempotent result logging. All paths are
relative; the same config runs locally or on Colab (auto-detected).

## Submission checklist

- [ ] `pytest -q` passes.
- [ ] Runbook steps 1–7 completed.
- [ ] `python -m src.build_report --github <your public repo URL>` re-run, so the title page
      carries the real link; submit `report/MRI_Final_Project_Report.pdf`.
- [ ] Repository pushed to a **public** GitHub repo, with the dataset excluded (`.gitignore` covers `data/`, `*.npy`, `cache/`, checkpoints).
- [ ] Every code file starts with both authors' names and IDs.

## Repository layout

```
configs/            default.yaml + one sweep file per experiment
src/                pipeline: dataset, masks, model, baselines, engine, metrics, analysis
  baselines/        classical_cs (+ _tv), zero_filled, unet, ista_net
  figures.py        method/data/results figures
  figures_mri.py    MRI-principle figures (k-space energy, PSF, Hermitian symmetry, DC line)
  figures_results.py  the brief's required plots, real and imaginary channels separately
  figures_code.py   syntax-highlighted source listings for the report
  make_qualitative.py the four required worked examples (runs from the slice cache)
  eval_contrast.py  white-matter/grey-matter contrast and CNR retention
  eval_crossmask.py generalization to unseen undersampling masks
  make_report_tables.py   renders report/tables.md from the logged CSVs
  report_data.py    every number the report quotes, read from the logs
  report_content.py the report itself, as a list of blocks
  report_render.py  PDF (ReportLab) and DOCX (python-docx) renderers
  build_report.py   entry point: report/MRI_Final_Project_Report.{pdf,docx,txt}
tests/              dataset-free test suite (pytest)
analysis.ipynb      thin notebook that calls src/analysis.py and src/figures.py
report/             the generated report (PDF/DOCX) + auto-generated tables.md
results/            runs.csv, samples.csv, crossmask.csv, figures/, one dir per run
logs/               stdout of the sweeps that produced the results
```

## References

- Yan Yang, Jian Sun, Huibin Li, Zongben Xu. *Deep ADMM-Net for Compressive Sensing MRI.*
  NIPS, 2016.
- Michael Lustig, David Donoho, John M. Pauly. *Sparse MRI: The application of compressed
  sensing for rapid MR imaging.* Magnetic Resonance in Medicine, 2007. (Wavelet + TV prior
  used by the baseline.)
- Antonin Chambolle. *An algorithm for total variation minimization and applications.*
  Journal of Mathematical Imaging and Vision, 2004. (The TV proximal solver.)
- Jian Zhang, Bernard Ghanem. *ISTA-Net: Interpretable Optimization-Inspired Deep Network
  for Image Compressive Sensing.* CVPR, 2018. (Alternative unrolling, in `src/baselines/`.)
