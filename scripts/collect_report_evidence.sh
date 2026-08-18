#!/usr/bin/env bash
# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
#
# Collect, on the machine that has the dataset, the evidence the final report cannot
# produce from the slice cache alone. Everything it writes is a small derived artefact
# (CSV, JSON, PNG) -- no image data is copied -- so the output directory is safe to bring
# back to a laptop and safe to commit.
#
# Server layout this expects (from the file explorer):
#
#   ~/                                   home
#   |- MRI/                              this project
#   |- MRI_2026_datasets/ -> symlink
#   |  |- brain_age/
#   |  |  |- selected_npy/               the .npy volumes
#   |  |  |- student_train_metadata.csv
#   |  |  |- student_val_metadata.csv
#   |  |  \- student_test_metadata.csv
#   |  \- Brats/                         (not used by this project)
#   \- data/ -> symlink to brain_age
#
# Usage, from the project root on the server:
#
#   bash scripts/collect_report_evidence.sh                 # stages 1-4, minutes
#   bash scripts/collect_report_evidence.sh --with-heavy    # also stages 5-6, ~1-2 h
#   bash scripts/collect_report_evidence.sh --data-root /path/to/brain_age
#
# Then copy the tarball it prints back to the laptop and re-run
#   python -m src.build_report
# to fold the new figures into the PDF and DOCX.

set -uo pipefail

PYTHON="${PYTHON:-}"
DATA_ROOT=""
WITH_HEAVY=0
OUT_DIR="results/dataset_evidence"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --with-heavy) WITH_HEAVY=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '3,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Locate the dataset. Try the symlink first, then the real path.
# ---------------------------------------------------------------------------
if [[ -z "$DATA_ROOT" ]]; then
  for candidate in "$HOME/data" "$HOME/MRI_2026_datasets/brain_age"; do
    if compgen -G "$candidate/*metadata*.csv" > /dev/null 2>&1; then
      DATA_ROOT="$candidate"; break
    fi
  done
fi
if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: could not find the metadata CSVs."
  echo "  Tried \$HOME/data and \$HOME/MRI_2026_datasets/brain_age."
  echo "  Pass the directory explicitly:  --data-root /path/to/brain_age"
  exit 1
fi

# Validate whatever we ended up with, including an explicitly passed path: running the
# whole pipeline against a wrong directory wastes an hour and produces a misleading bundle.
if ! compgen -G "$DATA_ROOT/*metadata*.csv" > /dev/null 2>&1; then
  echo "ERROR: no *metadata*.csv files in: $DATA_ROOT"
  echo "  Expected student_train_metadata.csv, student_val_metadata.csv,"
  echo "  student_test_metadata.csv (as in ~/MRI_2026_datasets/brain_age)."
  exit 1
fi
if [[ ! -d "$DATA_ROOT/selected_npy" ]]; then
  echo "ERROR: no selected_npy/ directory in: $DATA_ROOT"
  exit 1
fi
if [[ ! -d src ]]; then
  echo "ERROR: run this from the project root (the directory containing src/)."
  exit 1
fi

# Pick an interpreter that can actually run the project. On some systems bare `python` is
# still Python 2, which fails with an unhelpful traceback several stages in.
pick_python () {
  local candidates=()
  [[ -n "$PYTHON" ]] && candidates+=("$PYTHON")
  candidates+=(python3 python)
  for exe in "${candidates[@]}"; do
    command -v "$exe" > /dev/null 2>&1 || continue
    if "$exe" -c 'import sys, numpy, matplotlib; assert sys.version_info >= (3, 8)' \
         > /dev/null 2>&1; then
      PYTHON="$exe"; return 0
    fi
  done
  return 1
}
if ! pick_python; then
  echo "ERROR: no usable Python found."
  echo "  Need Python >= 3.8 with numpy and matplotlib importable."
  echo "  Activate the environment you trained in, then re-run, or pass one:"
  echo "    bash scripts/collect_report_evidence.sh --python /path/to/python"
  exit 1
fi

echo "=============================================================="
echo " dataset   : $DATA_ROOT"
echo " project   : $(pwd)"
echo " python    : $($PYTHON --version 2>&1)"
echo " output    : $OUT_DIR"
echo " heavy runs: $([[ $WITH_HEAVY -eq 1 ]] && echo yes || echo 'no (pass --with-heavy)')"
echo "=============================================================="

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/collect.log"
: > "$LOG"

# Run a stage, tee its output to the log, and keep going if it fails: a later stage may
# still succeed and a partial bundle is more useful than none.
stage () {
  local name="$1"; shift
  echo ""
  echo "--------------------------------------------------------------"
  echo ">>> $name"
  echo "--------------------------------------------------------------"
  { echo ""; echo "### $name"; } >> "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then
    echo "    [ok] $name"
  else
    echo "    [FAILED] $name -- continuing; see $LOG"
  fi
}

# ---------------------------------------------------------------------------
# 1. Subject accounting, acquisition parameters, repeat sessions, slice profiles
#    Cheap, and the audit is what closes the 5,242 -> 4,791 chain in section 1.5.
# ---------------------------------------------------------------------------
stage "1/6  dataset audit, acquisition columns, repeats, slice profiles" \
  "$PYTHON" -m src.dataset_evidence --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"

# ---------------------------------------------------------------------------
# 2. The demographics figure with real per-subject ages and sex. This is the
#    histogram the report currently cannot draw.
# ---------------------------------------------------------------------------
stage "2/6  demographics figure (age histogram, sex balance, cohorts)" \
  "$PYTHON" -m src.figures_demographics --meta-dir "$DATA_ROOT"

# ---------------------------------------------------------------------------
# 3. Copy the metadata CSVs into the project so the figure can be regenerated on the
#    laptop without the server. They are small text files, but they are still dataset
#    metadata: .gitignore them rather than committing them.
# ---------------------------------------------------------------------------
stage "3/6  stage the metadata CSVs locally (gitignored)" \
  bash -c 'mkdir -p metadata_local && cp -v "$1"/*metadata*.csv metadata_local/ && \
           grep -qxF "metadata_local/" .gitignore 2>/dev/null || \
           printf "\n# dataset metadata copied from the server (not for publication)\nmetadata_local/\n" >> .gitignore' \
  _ "$DATA_ROOT"

# ---------------------------------------------------------------------------
# 4. Regenerate the figures that only need the cache, so the bundle is self-consistent
#    with whatever the audit found.
# ---------------------------------------------------------------------------
stage "4/6  regenerate the cache-only figures" \
  bash -c '"$1" -m src.figures_mri --which energy hermitian tradeoff contrast dcline sequence && \
           "$1" -m src.figures_equations' _ "$PYTHON"

if [[ $WITH_HEAVY -eq 1 ]]; then
  # -------------------------------------------------------------------------
  # 5. Build the 256x256 slice cache and test whether the classical baseline does
  #    better at a larger matrix size. This is the experiment that checks the report's
  #    own explanation for why the baseline underperforms. No retraining needed, since
  #    the baseline is training-free. Expect tens of minutes: the TV solver dominates.
  # -------------------------------------------------------------------------
  stage "5/6  resolution experiment: classical CS at 128 vs 256" \
    "$PYTHON" -m src.eval_resolution --sizes 128 256 --ratios 0.2 0.3 0.5 --limit 120

  # -------------------------------------------------------------------------
  # 6. Off-centre slices: does the result hold away from the one slice we trained on?
  #    Needs the trained checkpoints, which live under results/.
  # -------------------------------------------------------------------------
  stage "6/6  per-stage activity audit over every checkpoint" \
    "$PYTHON" -c "import sys; sys.path.insert(0,'.'); \
from src.postprocess import stage_activity_main; stage_activity_main([])"
else
  echo ""
  echo ">>> skipping stages 5-6 (the slow ones). Re-run with --with-heavy to include:"
  echo "      5) classical CS at 128 vs 256 -- tests the report's central caveat"
  echo "      6) per-stage activity over all checkpoints"
fi

# ---------------------------------------------------------------------------
# Bundle only the small derived artefacts.
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M)"
BUNDLE="report_evidence_${STAMP}.tar.gz"
tar -czf "$BUNDLE" \
  "$OUT_DIR" \
  results/figures/*.png results/figures/*.csv \
  results/figures/equations/*.png \
  2>/dev/null

echo ""
echo "=============================================================="
echo " done"
echo "=============================================================="
echo " bundle : $BUNDLE  ($(du -h "$BUNDLE" 2>/dev/null | cut -f1))"
echo " log    : $LOG"
echo ""
echo " No image data is in the bundle -- only CSV, JSON and PNG."
echo ""
echo " Next, on your laptop:"
echo "   scp <user>@<server>:$(pwd)/$BUNDLE ."
echo "   tar -xzf $BUNDLE -C /path/to/MRI"
echo "   cd /path/to/MRI && python -m src.build_report"
echo ""
echo " Read $OUT_DIR/dataset_audit.json first: if matched_with_usable_age equals"
echo " subjects_recorded_in_splits, the report's section 1.5 accounting is confirmed."
