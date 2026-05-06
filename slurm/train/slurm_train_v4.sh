#!/bin/bash
#SBATCH --job-name=delphi_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v4_%j.log
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

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
mkdir -p $DELPHI/checkpoints/mimic_v4

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v4/meta.pkl','rb')); print(m['vocab_size']+1)")

echo "=== Delphi v4: Global Memory, exp-Weibull, post-2015 ICD-10 ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Vocab size: $VOCAB_SIZE"
echo "Config: block_size=128, n_summary=4, n_overflow=64"
echo "LR: 6e-4 → 6e-5 cosine, 50k steps"
echo ""

cd $DELPHI
$PYBIN train.py $PIPE/config/train_delphi_mimic_v4.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "=== Training complete. Checkpoint in $DELPHI/checkpoints/mimic_v4/ ==="
ls -lh $DELPHI/checkpoints/mimic_v4/
