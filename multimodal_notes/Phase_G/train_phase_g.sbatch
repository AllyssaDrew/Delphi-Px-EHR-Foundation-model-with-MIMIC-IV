#!/bin/bash
#SBATCH --job-name=delphi_v6_phaseG
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpu_access
#SBATCH --output=logs/phase_g_%j.log
#SBATCH --error=logs/phase_g_%j.log

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Phase G (ablation) started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Phase G starts from the same Phase E checkpoint as Phase F (fair ablation)
mkdir -p ${DELPHI}/checkpoints/mimic_v6_phase_g
if [ ! -f "${DELPHI}/checkpoints/mimic_v6_phase_g/ckpt.pt" ]; then
    cp ${DELPHI}/checkpoints/mimic_v6_phase_e/ckpt.pt \
       ${DELPHI}/checkpoints/mimic_v6_phase_g/ckpt.pt
    echo "Copied mimic_v6_phase_e → mimic_v6_phase_g"
fi

cd ${DELPHI}
$PYTHON train_v6.py \
    ${PIPE}/config/train_delphi_mimic_v6_phase_g.py

echo "Phase G done at $(date)"
