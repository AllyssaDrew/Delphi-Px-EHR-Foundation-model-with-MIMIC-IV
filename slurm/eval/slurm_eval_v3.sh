#!/bin/bash
#SBATCH --job-name=delphi_v3_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_v3_%j.log
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

mkdir -p $PIPE/logs $PIPE/eval_output/v3

echo "=== Delphi v3 AUC Evaluation ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: $DELPHI/checkpoints/mimic_v3/ckpt.pt"
echo ""

cd $DELPHI
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v3 \
    --output_path=$PIPE/eval_output/v3 \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v3/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v3/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=20

echo ""
echo "=== Done. Results in $PIPE/eval_output/v3/ ==="
ls -lh $PIPE/eval_output/v3/
