#!/bin/bash
#SBATCH --job-name=eval_v61
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_access
#SBATCH --output=logs/eval_v61_%j.log
#SBATCH --error=logs/eval_v61_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

RESULTS=${PIPE}/clinical_phenotyping/Phase_E/results
LABELS=${DELPHI}/delphi_labels_chapters_colours_icd.csv

echo "v6.1 evaluation started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p ${RESULTS}/v5_baseline
mkdir -p ${RESULTS}/v61_phase_f

# ── 1. v5 baseline ────────────────────────────────────────────────────────────
echo ""
echo "=== [1/2] v5 baseline ==="
$PYTHON ${DELPHI}/evaluate_auc.py \
    --input_path       ${PIPE}/data/mimic_data_v5 \
    --output_path      ${RESULTS}/v5_baseline \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v5_phase2b/ckpt.pt \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --filter_min_total 100

echo "v5 baseline done at $(date)"

# ── 2. v6.1 Phase D-F ────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] v6.1 Phase D-F ==="
$PYTHON ${DELPHI}/evaluate_auc.py \
    --input_path       ${PIPE}/data/mimic_data_v61 \
    --output_path      ${RESULTS}/v61_phase_f \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v61_phase_f/ckpt.pt \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --filter_min_total 100

echo "v6.1 Phase D-F done at $(date)"

echo ""
echo "=== Evaluation complete at $(date) ==="
echo "Results in ${RESULTS}/"
