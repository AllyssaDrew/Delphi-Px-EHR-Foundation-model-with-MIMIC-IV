#!/bin/bash
#SBATCH --job-name=delphi_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/delphi_tiny_%j.log
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

mkdir -p $PIPE/logs $PIPE/checkpoints/mimic_tiny

echo "=== Quick Validation Experiment ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Symlink data so train.py can find it ─────────────────────────────────────
ln -sfn $PIPE/data/mimic_data $DELPHI/data/mimic_data

# ── Step 1: Compute sex+age baseline AUC ─────────────────────────────────────
echo ""
echo "[1/3] Computing sex+age baseline AUC ..."
cd $PIPE
$PYBIN 03_baseline_auc.py \
    --data_dir data/mimic_data \
    --min_cases 20

# ── Step 2: Train tiny Delphi model ──────────────────────────────────────────
echo ""
echo "[2/3] Training tiny Delphi model ..."
cd $DELPHI
$PYBIN train.py \
    $PIPE/config/train_delphi_mimic_tiny.py \
    --dataset=mimic_data \
    --out_dir=$PIPE/checkpoints/mimic_tiny \
    --device=cuda \
    --dtype=float32

# ── Step 3: Evaluate model AUC ───────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating model AUC ..."
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data \
    --output_path=$PIPE/eval_output/tiny \
    --model_ckpt_path=$PIPE/checkpoints/mimic_tiny/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data/mimic_labels.csv \
    --no_event_token_rate=5 \
    --dataset_subset_size=3000 \
    --filter_min_total=20

echo ""
echo "=== Quick validation complete. Check eval_output/tiny/ for AUC results ==="
echo "Compare with baseline at eval_output/baseline_auc.csv"
