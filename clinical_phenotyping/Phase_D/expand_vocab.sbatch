#!/bin/bash
#SBATCH --job-name=expand_vocab_v61
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/expand_vocab_%j.log
#SBATCH --error=logs/expand_vocab_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Phase D: expanding vocab v5 → v6.1 at $(date) on $(hostname)"
mkdir -p ${PIPE}/clinical_phenotyping/Phase_D/logs

$PYTHON ${PIPE}/clinical_phenotyping/Phase_D/expand_vocab_v5_to_v61.py

echo "Phase D vocab expansion done at $(date)"
