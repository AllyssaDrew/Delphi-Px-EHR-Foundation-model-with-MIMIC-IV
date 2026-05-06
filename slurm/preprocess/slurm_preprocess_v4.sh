#!/bin/bash
#SBATCH --job-name=delphi_preprocess_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/preprocess_v4_%j.log
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

echo "=== Delphi v4 Preprocessing: post-2015 ICD-10 filter ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd $PIPE
$PYBIN 02_preprocess_v4.py

echo ""
echo "=== Preprocessing complete ==="
echo "End: $(date)"
echo ""
ls -lh $PIPE/data/mimic_data_v4/

# Create symlink in Delphi data dir so train.py can find the dataset
ln -sfn $PIPE/data/mimic_data_v4 $DELPHI/data/mimic_data_v4
echo "Symlink: $DELPHI/data/mimic_data_v4 → $PIPE/data/mimic_data_v4"

echo ""
echo "Vocab size for training:"
$PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v4/meta.pkl','rb')); print('vocab_size (stored):', m['vocab_size']); print('vocab_size (model = +1):', m['vocab_size']+1)"
