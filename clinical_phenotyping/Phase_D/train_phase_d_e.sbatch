#!/bin/bash
#SBATCH --job-name=delphi_v61_phaseE
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpu_access
#SBATCH --output=logs/phase_d_e_%j.log
#SBATCH --error=logs/phase_d_e_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Phase D-E (v6.1 phenotype embedding warmup) started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p ${DELPHI}/checkpoints/mimic_v61_phase_e
cp ${DELPHI}/checkpoints/mimic_v61_init/ckpt.pt \
   ${DELPHI}/checkpoints/mimic_v61_phase_e/ckpt.pt
echo "Copied mimic_v61_init → mimic_v61_phase_e"

cd ${DELPHI}
$PYTHON train_v61.py \
    ${DELPHI}/config/train_delphi_mimic_v61_phase_e.py

echo "Phase D-E done at $(date)"
