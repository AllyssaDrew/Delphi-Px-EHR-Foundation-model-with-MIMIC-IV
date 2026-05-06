#!/bin/bash
#SBATCH --job-name=delphi_v5ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v5_finetune_%j.log
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
mkdir -p $DELPHI/checkpoints/mimic_v5_finetune

# Read v5 vocab_size from meta.pkl
VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v5/meta.pkl','rb')); print(m['vocab_size']+1)")

echo "=== Delphi v5 Finetune: Method A (two-phase gradient masking) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "v5 Vocab size (model): $VOCAB_SIZE"
echo ""

cd $DELPHI

# ── Step 1: Expand v4 vocabulary → v5 ────────────────────────────────────────
echo "=== Step 1: Expanding vocab v4 → v5 ==="
echo "Start: $(date)"
$PYBIN expand_vocab_v4_to_v5.py
echo "Done: $(date)"
echo ""

# ── Step 2: Phase 1 — warm-up new embeddings (500 steps, backbone frozen) ────
echo "=== Step 2: Phase 1 warm-up (500 steps, freeze_backbone=True) ==="
echo "Start: $(date)"
$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_phase1.py \
    --vocab_size=$VOCAB_SIZE
echo "Done: $(date)"
echo ""

# ── Step 3: Phase 2 — full finetune (10,000 steps, all params unfrozen) ──────
echo "=== Step 3: Phase 2 finetune (10,000 steps, LR 1e-4 → 1e-5) ==="
echo "Start: $(date)"
$PYBIN train.py $PIPE/config/train_delphi_mimic_v5_finetune.py \
    --vocab_size=$VOCAB_SIZE
echo "Done: $(date)"
echo ""

echo "=== Training complete. Checkpoint in $DELPHI/checkpoints/mimic_v5_finetune/ ==="
ls -lh $DELPHI/checkpoints/mimic_v5_finetune/
