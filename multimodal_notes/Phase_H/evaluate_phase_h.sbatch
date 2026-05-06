#!/bin/bash
#SBATCH --job-name=delphi_v6_phaseH
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_access
#SBATCH --output=logs/phase_h_%j.log
#SBATCH --error=logs/phase_h_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

RESULTS=${PIPE}/multimodal_notes/Phase_H/results
LABELS=${DELPHI}/delphi_labels_chapters_colours_icd.csv

echo "Phase H evaluation started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p ${RESULTS}/v5_baseline
mkdir -p ${RESULTS}/v6_phase_f
mkdir -p ${RESULTS}/v6_phase_f_ablation
mkdir -p ${RESULTS}/v6_phase_g

# ── 1. v5 baseline (no notes) ────────────────────────────────────────────────
echo ""
echo "=== [1/4] v5 baseline ==="
$PYTHON ${DELPHI}/evaluate_auc.py \
    --input_path       ${PIPE}/data/mimic_data_v5 \
    --output_path      ${RESULTS}/v5_baseline \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v5_phase2b/ckpt.pt \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --filter_min_total 100

echo "v5 baseline done at $(date)"

# ── 2. v6 Phase F — real note embeddings ────────────────────────────────────
echo ""
echo "=== [2/4] v6 Phase F (real notes) ==="
$PYTHON ${DELPHI}/evaluate_auc_v6.py \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v6_phase_f/ckpt.pt \
    --output_path      ${RESULTS}/v6_phase_f \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --filter_min_total 100

echo "v6 Phase F done at $(date)"

# ── 3. v6 Phase F — random embeddings at eval (content ablation) ────────────
echo ""
echo "=== [3/4] v6 Phase F + ablation (random embeddings at eval) ==="
$PYTHON ${DELPHI}/evaluate_auc_v6.py \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v6_phase_f/ckpt.pt \
    --output_path      ${RESULTS}/v6_phase_f_ablation \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --ablation_mode \
    --filter_min_total 100

echo "v6 Phase F ablation done at $(date)"

# ── 4. v6 Phase G — trained with random embeddings ──────────────────────────
echo ""
echo "=== [4/4] v6 Phase G (trained with random notes) ==="
$PYTHON ${DELPHI}/evaluate_auc_v6.py \
    --model_ckpt_path  ${DELPHI}/checkpoints/mimic_v6_phase_g/ckpt.pt \
    --output_path      ${RESULTS}/v6_phase_g \
    --no_event_token_rate 5 \
    --labels_file      ${LABELS} \
    --gm_eval \
    --ablation_mode \
    --filter_min_total 100

echo "v6 Phase G done at $(date)"

echo ""
echo "=== Phase H complete at $(date) ==="
echo "Results in ${RESULTS}/"
