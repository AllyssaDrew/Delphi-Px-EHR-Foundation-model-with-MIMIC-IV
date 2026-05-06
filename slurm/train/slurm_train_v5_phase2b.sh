#!/bin/bash
#SBATCH --job-name=delphi_v5p2b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_v5_phase2b_%j.log
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

mkdir -p $PIPE/logs $DELPHI/checkpoints/mimic_v5_phase2b

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb')); print(m['vocab_size']+1)")

# Copy Phase 1b checkpoint into Phase 2b dir
SRC=$DELPHI/checkpoints/mimic_v5_phase1b/ckpt.pt
DST=$DELPHI/checkpoints/mimic_v5_phase2b/ckpt.pt
if [ ! -f "$DST" ]; then
    echo "Copying Phase 1b checkpoint → mimic_v5_phase2b/"
    cp "$SRC" "$DST"
else
    echo "Checkpoint already present in mimic_v5_phase2b/, skipping copy"
fi

echo "=== Delphi v5 Phase 2b: full model fine-tune from Phase 1b (iter 12000) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Resuming from Phase 1b checkpoint (Neoplasm=0.8007, Overall=0.8182)"
echo "Full model: freeze_backbone=False, LR=5e-6 constant, 3000 steps"
echo "v5 Vocab size (model): $VOCAB_SIZE"
echo "Start: $(date)"
echo ""

cd $DELPHI

$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_phase2b.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Done: $(date)"
echo ""
echo "=== Phase 2b complete. Checkpoint: $DELPHI/checkpoints/mimic_v5_phase2b/ ==="
ls -lh $DELPHI/checkpoints/mimic_v5_phase2b/
