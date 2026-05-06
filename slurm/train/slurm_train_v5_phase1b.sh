#!/bin/bash
#SBATCH --job-name=delphi_v5p1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_v5_phase1b_%j.log
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

mkdir -p $PIPE/logs $DELPHI/checkpoints/mimic_v5_phase1b

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb')); print(m['vocab_size']+1)")

# Copy step-11000 extended best checkpoint into Phase 1b dir
SRC=$DELPHI/checkpoints/mimic_v5_extended/ckpt.pt
DST=$DELPHI/checkpoints/mimic_v5_phase1b/ckpt.pt
if [ ! -f "$DST" ]; then
    echo "Copying step-11000 extended checkpoint → mimic_v5_phase1b/"
    cp "$SRC" "$DST"
else
    echo "Checkpoint already present in mimic_v5_phase1b/, skipping copy"
fi

echo "=== Delphi v5 Phase 1b: targeted warm-up for 42 new ICD code rows (1469-1510) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Resuming from step-11000 extended best (val 6.7543)"
echo "Warm rows: model-space 1469-1510 (stored IDs 1468-1509, 42 new ICD codes)"
echo "LR=1e-3 constant, 1000 steps, backbone + all other vocab frozen"
echo "v5 Vocab size (model): $VOCAB_SIZE"
echo "Start: $(date)"
echo ""

cd $DELPHI

$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_phase1b.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Done: $(date)"
echo ""
echo "=== Phase 1b complete. Checkpoint: $DELPHI/checkpoints/mimic_v5_phase1b/ ==="
ls -lh $DELPHI/checkpoints/mimic_v5_phase1b/
