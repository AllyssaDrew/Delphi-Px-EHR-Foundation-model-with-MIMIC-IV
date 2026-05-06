#!/bin/bash
#SBATCH --job-name=bootstrap_ci
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_access
#SBATCH --output=logs/bootstrap_ci_%j.log
#SBATCH --error=logs/bootstrap_ci_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Bootstrap CI started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p ${PIPE}/logs

$PYTHON ${PIPE}/bootstrap_ci.py

echo "Bootstrap CI done at $(date)"
