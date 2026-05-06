#!/bin/bash
# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

# Usage:
#   ./submit.sh <script.py> [--gpu] [--hours N] [--mem N] [--cpus N] [--name JOBNAME]
#
# Examples:
#   ./submit.sh E3_shap_chapter_matrix/run_e3_shap.py --gpu
#   ./submit.sh E4_temporal_decay/run_e4_temporal_decay.py --hours 4
#   ./submit.sh E3_shap_chapter_matrix/run_e3_extend1000.py --gpu --hours 3

set -e

SCRIPT=""
USE_GPU=0
HOURS=12
MEM=32
CPUS=8
JOBNAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)    USE_GPU=1; shift ;;
        --hours)  HOURS=$2; shift 2 ;;
        --mem)    MEM=$2; shift 2 ;;
        --cpus)   CPUS=$2; shift 2 ;;
        --name)   JOBNAME=$2; shift 2 ;;
        *.py)     SCRIPT=$1; shift ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCRIPT" ]]; then
    echo "Error: no .py script specified"
    echo "Usage: ./submit.sh <script.py> [--gpu] [--hours N] [--mem N] [--cpus N] [--name NAME]"
    exit 1
fi

# Resolve absolute path
if [[ "$SCRIPT" != /* ]]; then
    SCRIPT_ABS="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
else
    SCRIPT_ABS="$SCRIPT"
fi

if [[ ! -f "$SCRIPT_ABS" ]]; then
    echo "Error: script not found: $SCRIPT_ABS"
    exit 1
fi

SCRIPT_DIR="$(dirname "$SCRIPT_ABS")"
SCRIPT_BASE="$(basename "$SCRIPT_ABS" .py)"

if [[ -z "$JOBNAME" ]]; then
    JOBNAME="$SCRIPT_BASE"
fi

LOG="$SCRIPT_DIR/${SCRIPT_BASE}_slurm_%j.log"

if [[ $USE_GPU -eq 1 ]]; then
    PARTITION="l40-gpu"
    GPU_LINE="#SBATCH --gres=gpu:1"
    # GPU nodes have fewer CPUs — cap at 4
    [[ $CPUS -gt 4 ]] && CPUS=4
else
    PARTITION="general"
    GPU_LINE=""
fi

SBATCH_CONTENT="#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${HOURS}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}
${GPU_LINE}

# source ~/.bashrc  # uncomment if conda is not on PATH
conda activate dl_env

cd ${DELPHI}

echo \"Job \$SLURM_JOB_ID started at \$(date)\"
echo \"Script: ${SCRIPT_ABS}\"
echo \"Node: \$(hostname)\"
[[ -n \"\$CUDA_VISIBLE_DEVICES\" ]] && echo \"GPU: \$CUDA_VISIBLE_DEVICES\"

python ${SCRIPT_ABS}

echo \"Job finished at \$(date)\"
"

TMPFILE=$(mktemp /tmp/slurm_XXXXXX.sbatch)
echo "$SBATCH_CONTENT" > "$TMPFILE"

echo "Submitting: $SCRIPT_ABS"
echo "Partition:  $PARTITION  |  Time: ${HOURS}h  |  Mem: ${MEM}G  |  CPUs: ${CPUS}  |  GPU: $([[ $USE_GPU -eq 1 ]] && echo yes || echo no)"
echo "Log:        ${SCRIPT_DIR}/${SCRIPT_BASE}_slurm_<JOBID>.log"
echo ""

JOB_OUTPUT=$(sbatch "$TMPFILE")
rm "$TMPFILE"

echo "$JOB_OUTPUT"
JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
echo ""
echo "Monitor:  squeue -j $JOB_ID"
echo "Log:      tail -f ${SCRIPT_DIR}/${SCRIPT_BASE}_slurm_${JOB_ID}.log"
