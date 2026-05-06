#!/bin/bash
# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

# Submit Phase B as 8-way parallel SLURM array (one GPU per shard).
# Usage: bash submit_phase_b.sh [--n_shards 8]

N_SHARDS=${2:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=delphi_v6_phaseB
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-$((N_SHARDS-1))
#SBATCH --output=${LOG_DIR}/shard_%a_%j.log
#SBATCH --error=${LOG_DIR}/shard_%a_%j.log

# source ~/.bashrc  # uncomment if conda is not on PATH
conda activate dl_env

echo "Shard \$SLURM_ARRAY_TASK_ID / ${N_SHARDS} started at \$(date) on \$(hostname)"

python ${SCRIPT_DIR}/run_phase_b_shard.py \\
    --shard \$SLURM_ARRAY_TASK_ID \\
    --n_shards ${N_SHARDS}

echo "Shard \$SLURM_ARRAY_TASK_ID done at \$(date)"
EOF
)

echo "Submitted array job: $JOB"
echo "Monitor: squeue -j $JOB"
echo "Logs:    $LOG_DIR/shard_*_${JOB}.log"
