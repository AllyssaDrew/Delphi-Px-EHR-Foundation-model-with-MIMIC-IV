#!/bin/bash
#SBATCH --job-name=v5_phase_a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/phase_a_%j.log
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

mkdir -p $PIPE/logs

echo "=== Phase A: ICD-9 one-to-one GEM mapping statistics ==="
echo "Node: $(hostname)  Date: $(date)"

cd $PIPE
$PYBIN phase_a_icd9_stats.py 2>&1 | tee $PIPE/logs/phase_a_results.txt

echo ""
echo "=== Phase A complete. Results in logs/phase_a_results.txt ==="
