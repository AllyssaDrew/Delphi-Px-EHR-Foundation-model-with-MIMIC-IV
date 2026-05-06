#!/bin/bash
#SBATCH --job-name=phenotype_extract
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/extract_%j.log
#SBATCH --error=logs/extract_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Phase B phenotype extraction started at $(date) on $(hostname)"
echo "CPUs available: $(nproc)"

mkdir -p ${PIPE}/clinical_phenotyping/Phase_B/logs

$PYTHON ${PIPE}/clinical_phenotyping/Phase_B/run_extraction.py \
    --workers 30 \
    --chunk_size 500

echo "Phase B done at $(date)"
