#!/bin/bash
#SBATCH --job-name=delphi_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/delphi_resume_%j.log
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access

# ── Setup ──────────────────────────────────────────────────────────────────────
# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

PYBIN=${PYTHON}

mkdir -p $PIPE/logs $PIPE/eval_output/full_allpat

echo "=== Resume Training (20k → 40k) + Full Evaluation ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Resume training from iter 20000 → 40000 ──────────────────────────
echo ""
echo "[1/2] Resuming full model training (iter 20000 → 40000) ..."
cd $DELPHI
$PYBIN train.py \
    $PIPE/config/train_delphi_mimic_full.py \
    --init_from=resume \
    --max_iters=40000 \
    --learning_rate=2e-4 \
    --min_lr=2e-5 \
    --lr_decay_iters=40000 \
    --warmup_iters=0 \
    --out_dir=$PIPE/checkpoints/mimic_full \
    --device=cuda \
    --dtype=float32

# ── Step 2: Full val evaluation (all 21k patients, no subset limit) ───────────
echo ""
echo "[2/2] Full validation evaluation (all val patients) ..."
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v2 \
    --output_path=$PIPE/eval_output/full_allpat \
    --model_ckpt_path=$PIPE/checkpoints/mimic_full/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v2/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=20

echo ""
echo "=== Done. Results in eval_output/full_allpat/ ==="
