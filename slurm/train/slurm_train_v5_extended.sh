#!/bin/bash
#SBATCH --job-name=delphi_v5ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v5_extended_%j.log
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

mkdir -p $PIPE/logs $DELPHI/checkpoints/mimic_v5_extended

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb')); print(m['vocab_size']+1)")

# Copy step-9000 best checkpoint into the new output dir (if not already there)
SRC=$DELPHI/checkpoints/mimic_v5_finetune/ckpt.pt
DST=$DELPHI/checkpoints/mimic_v5_extended/ckpt.pt
if [ ! -f "$DST" ]; then
    echo "Copying step-9000 best checkpoint → mimic_v5_extended/"
    cp "$SRC" "$DST"
else
    echo "Checkpoint already present in mimic_v5_extended/, skipping copy"
fi

echo "=== Delphi v5 Extended finetune (5,000 steps at LR=5e-6, from step 9000) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Resuming from step-9000 best (val 6.7770)"
echo "v5 Vocab size (model): $VOCAB_SIZE"
echo "Start: $(date)"
echo ""

cd $DELPHI

$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_extended.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Done: $(date)"
echo ""
echo "=== Extended finetune complete. Checkpoint: $DELPHI/checkpoints/mimic_v5_extended/ ==="
ls -lh $DELPHI/checkpoints/mimic_v5_extended/
