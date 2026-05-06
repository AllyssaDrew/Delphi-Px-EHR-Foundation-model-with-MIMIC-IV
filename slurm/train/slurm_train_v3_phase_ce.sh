#!/bin/bash
#SBATCH --job-name=delphi_v3_ce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v3_phase_ce_%j.log
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

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v3/meta.pkl','rb')); print(m['vocab_size']+1)")

echo "=== Delphi v3 Phase C+E: block_size=128 + warm restart ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Vocab size: $VOCAB_SIZE"
echo "Resuming from: $DELPHI/checkpoints/mimic_v3/ckpt.pt"
echo "block_size: 64 → 128  |  LR: 2e-4 → 2e-5 over 20k steps"
echo ""

cd $DELPHI
$PYBIN train.py $PIPE/config/train_delphi_mimic_v3_phase_ce.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "=== Phase C+E complete. Checkpoint in $DELPHI/checkpoints/mimic_v3/ ==="
ls -lh $DELPHI/checkpoints/mimic_v3/
