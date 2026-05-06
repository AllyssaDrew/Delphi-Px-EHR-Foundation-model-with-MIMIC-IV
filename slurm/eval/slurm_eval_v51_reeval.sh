#!/bin/bash
#SBATCH --job-name=eval_v51_reeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_v51_reeval_%j.log
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access

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

mkdir -p $PIPE/logs $PIPE/eval_output/v5_phase2b $PIPE/eval_output/v4_stable_gm

echo "=== v5.1 Re-evaluation: off-by-1 fix applied to evaluate_auc.py ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

cd $DELPHI

echo "--- [1/2] v5 Phase 2b + GM (select=right, filter>=50) ---"
echo "Start: $(date)"
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v5 \
    --output_path=$PIPE/eval_output/v5_phase2b \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v5_phase2b/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v5/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=50 \
    --gm_eval
echo "Done: $(date)"
echo ""

echo "--- [2/2] v4 Stable + GM (select=right, filter>=50) ---"
echo "Start: $(date)"
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v4 \
    --output_path=$PIPE/eval_output/v4_stable_gm \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v4_stable/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v4/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=50 \
    --gm_eval
echo "Done: $(date)"
echo ""

echo "=== AUC Summary (filter_min_total >= 50, corrected off-by-1) ==="
$PYBIN $PIPE/final_summary.py
