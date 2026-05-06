#!/bin/bash
#SBATCH --job-name=build_v61
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/build_v61_%j.log
#SBATCH --error=logs/build_v61_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Phase C: building v6.1 dataset at $(date) on $(hostname)"
mkdir -p ${PIPE}/clinical_phenotyping/Phase_C/logs

$PYTHON ${PIPE}/clinical_phenotyping/Phase_C/build_v61_dataset.py

echo "Phase C done at $(date)"
