#!/bin/bash
#SBATCH --job-name=delphi_preprocess_v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/preprocess_v5_%j.log
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

echo "=== Delphi v5 Preprocessing ==="
echo "Changes: pre-2015 ICD-9 one-to-one GEM, eGFR (CKD-EPI 2021, 4-level), TroponinI"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd $PIPE
$PYBIN 02_preprocess_v5.py

echo ""
echo "=== Preprocessing complete ==="
echo "End: $(date)"
echo ""
ls -lh $PIPE/data/mimic_data_v5/

# Symlink for training
ln -sfn $PIPE/data/mimic_data_v5 $DELPHI/data/mimic_data_v5
echo "Symlink: $DELPHI/data/mimic_data_v5 → $PIPE/data/mimic_data_v5"

echo ""
echo "Vocab info for finetune config:"
$PYBIN -c "
import pickle
m = pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb'))
print('vocab_size (stored):', m['vocab_size'])
print('vocab_size (model) :', m['vocab_size']+1)
print('DEATH_TOKEN        :', m['DEATH_TOKEN'])
print('LAB_TOKEN_START    :', m['LAB_TOKEN_START'])
print('N_LAB_TOKENS       :', m['N_LAB_TOKENS'])
print('ignore_tokens      :', m['ignore_tokens'][:8], '... +', len(m['ignore_tokens'])-8, 'lab tokens')
"
