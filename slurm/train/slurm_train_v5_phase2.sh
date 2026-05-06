#!/bin/bash
#SBATCH --job-name=delphi_v5p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v5_phase2_%j.log
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
export PYTHONUNBUFFERED=1

mkdir -p $PIPE/logs

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb')); print(m['vocab_size']+1)")

echo "=== Delphi v5 Phase 2 finetune (10,000 steps, full model) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Resuming from Phase 1 checkpoint (iter=500)"
echo "v5 Vocab size (model): $VOCAB_SIZE"
echo "Start: $(date)"
echo ""

cd $DELPHI

$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_finetune.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Done: $(date)"
echo ""
echo "=== Phase 2 complete. Checkpoint: $DELPHI/checkpoints/mimic_v5_finetune/ ==="
ls -lh $DELPHI/checkpoints/mimic_v5_finetune/
