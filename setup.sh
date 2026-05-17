#!/usr/bin/env bash
# One-shot setup for RGBench. Run from the repo root:
#
#     bash setup.sh
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. Creates a local venv at .venv/ if one isn't already active.
#   3. pip installs RGBench in editable mode with [all] extras.
#   4. Downloads SAM + GroundingDINO checkpoints into third_party/.
#   5. Downloads the smoke-test data sample from Hugging Face.
#   6. Runs the PyBullet quickstart to verify the install.
#
# Environment overrides:
#   ISAACSIM_PYTHON       — path to Isaac Sim's python.sh (optional)
#   RGBENCH_CHECKPOINT_DIR— where to drop SAM/GroundingDINO weights
#   RGBENCH_HF_DATASET    — Hugging Face dataset repo id
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- 1. Python version check ----------------------------------------------
PY=python3
if ! command -v "$PY" >/dev/null; then
  echo "error: python3 not found on PATH" >&2; exit 1
fi
MAJ_MIN=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$MAJ_MIN" in
  3.10|3.11|3.12) ;;
  *) echo "error: Python $MAJ_MIN found; RGBench requires 3.10+" >&2; exit 1 ;;
esac
echo "[1/6] Python $MAJ_MIN OK"

# --- 2. Venv (skip if already in one) -------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  if [[ ! -d .venv ]]; then
    echo "[2/6] Creating venv at .venv/"
    "$PY" -m venv .venv
  else
    echo "[2/6] Using existing .venv/"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[2/6] Reusing active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
fi

# --- 3. Editable install --------------------------------------------------
echo "[3/6] pip install -e .[all]"
pip install --upgrade pip wheel >/dev/null
pip install -e ".[all]"

# --- 4. Checkpoints -------------------------------------------------------
echo "[4/6] Downloading SAM + GroundingDINO checkpoints"
python scripts/download_checkpoints.py || {
  echo "warn: checkpoint download failed; you can re-run scripts/download_checkpoints.py later" >&2
}

# --- 5. Sample data -------------------------------------------------------
echo "[5/6] Downloading sample data from Hugging Face"
python scripts/download_data.py --sample-only || {
  echo "warn: sample download failed; verify your network and HF availability" >&2
}

# --- 6. Smoke test --------------------------------------------------------
echo "[6/6] Running PyBullet quickstart smoke test"
python examples/pybullet_quickstart.py || {
  echo "warn: quickstart failed; see docs/INSTALL.md for troubleshooting" >&2
  exit 1
}

echo
echo "RGBench is ready. Try:"
echo "    make benchmark sim=pybullet"
echo "    python scripts/run_benchmark.py params.sim_environment=pybullet"
