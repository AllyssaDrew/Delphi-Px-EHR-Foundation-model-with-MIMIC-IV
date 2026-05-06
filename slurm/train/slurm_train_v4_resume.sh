#!/bin/bash
#SBATCH --job-name=delphi_v4r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v4_resume_%j.log
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

PYBIN=${PYTHON}

mkdir -p $PIPE/logs
mkdir -p $DELPHI/checkpoints/mimic_v4

VOCAB_SIZE=$($PYBIN -c "import pickle; m=pickle.load(open('$PIPE/data/mimic_data_v4/meta.pkl','rb')); print(m['vocab_size']+1)")

echo "=== Delphi v4 RESUME: fixed Weibull NLL (log-space hazard) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Vocab size: $VOCAB_SIZE"
echo "Resuming from checkpoints/mimic_v4/ckpt.pt (step 9000, val 8.7657)"
echo ""

cd $DELPHI
$PYBIN train.py $PIPE/config/train_delphi_mimic_v4_resume.py \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "=== Training complete. Checkpoint in $DELPHI/checkpoints/mimic_v4/ ==="
ls -lh $DELPHI/checkpoints/mimic_v4/
