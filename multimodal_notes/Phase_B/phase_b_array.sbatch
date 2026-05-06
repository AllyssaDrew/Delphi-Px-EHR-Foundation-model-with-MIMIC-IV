#!/bin/bash
#SBATCH --job-name=delphi_v6_phaseB
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --qos=gpu_access
#SBATCH --output=logs/shard_%a_%j.log
#SBATCH --error=logs/shard_%a_%j.log

export HF_HOME=${HF_HOME:-${DELPHI_PROJECT_ROOT}/huggingface}

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────


echo "Shard ${SLURM_ARRAY_TASK_ID}/8 started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$PYTHON ${PIPE}/multimodal_notes/Phase_B/run_phase_b_shard.py \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --n_shards 8

echo "Shard ${SLURM_ARRAY_TASK_ID} done at $(date)"
