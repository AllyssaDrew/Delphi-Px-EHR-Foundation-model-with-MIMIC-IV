#!/bin/bash
#SBATCH --job-name=mimic_v3_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mimic_v3_prep_%j.log
#SBATCH --partition=general

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

PYBIN=${PYTHON}

mkdir -p $PIPE/logs $PIPE/data/mimic_data_v3

echo "=== MIMIC-IV v3 Preprocessing ==="
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
echo "Changes: first-occurrence dedup, freq≥25, lab tokens (8 labs)"
echo ""

cd $PIPE
$PYBIN 02_preprocess_v3.py

echo ""
echo "=== Done. Output in data/mimic_data_v3/ ==="
ls -lh $PIPE/data/mimic_data_v3/
