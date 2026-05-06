#!/bin/bash
#SBATCH --job-name=delphi_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/delphi_full_%j.log
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access

# ── Setup ──────────────────────────────────────────────────────────────────────
# ── Portable path configuration ──────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory containing mimic_pipeline/ and
# Delphi/Delphi-main/ as siblings, e.g.:
#   export DELPHI_PROJECT_ROOT=/your/project/root
PYTHON=${PYTHON:-python}            # override with: export PYTHON=/path/to/envs/delphi/bin/python
DELPHI=${DELPHI_DIR:-${DELPHI_PROJECT_ROOT}/Delphi/Delphi-main}
PIPE=${MIMIC_PIPELINE_DIR:-${DELPHI_PROJECT_ROOT}/mimic_pipeline}
# ─────────────────────────────────────────────────────────────────────────────

PYBIN=${PYTHON}

mkdir -p $PIPE/logs $PIPE/checkpoints/mimic_full

echo "=== Full Training (B+D data) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 0: Run B+D preprocessing (if not already done) ──────────────────────
if [ ! -f "$PIPE/data/mimic_data_v2/train.bin" ]; then
    echo "[0/3] Running B+D preprocessing ..."
    cd $PIPE
    $PYBIN 02_preprocess_v2.py
    echo "Preprocessing complete."
else
    echo "[0/3] B+D data already exists, skipping preprocessing."
fi

# ── Symlink data ──────────────────────────────────────────────────────────────
ln -sfn $PIPE/data/mimic_data_v2 $DELPHI/data/mimic_data_v2

# ── Read vocab_size from meta.pkl and patch the config ───────────────────────
VOCAB_SIZE=$($PYBIN -c "
import pickle
with open('$PIPE/data/mimic_data_v2/meta.pkl','rb') as f:
    m = pickle.load(f)
print(m['vocab_size'] + 1)  # +1: get_batch shifts tokens by 1, embedding needs one extra slot
")
echo "Detected vocab_size=$VOCAB_SIZE — patching config ..."
sed -i "s/^vocab_size = .*/vocab_size = $VOCAB_SIZE/" $PIPE/config/train_delphi_mimic_full.py

# ── Step 1: Train full model ──────────────────────────────────────────────────
echo ""
echo "[1/2] Training full Delphi-MIMIC model ..."
cd $DELPHI
$PYBIN train.py \
    $PIPE/config/train_delphi_mimic_full.py \
    --dataset=mimic_data_v2 \
    --out_dir=$PIPE/checkpoints/mimic_full \
    --device=cuda \
    --dtype=float32

# ── Step 2: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[2/2] Evaluating full model AUC ..."
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v2 \
    --output_path=$PIPE/eval_output/full \
    --model_ckpt_path=$PIPE/checkpoints/mimic_full/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v2/mimic_labels.csv \
    --no_event_token_rate=5 \
    --dataset_subset_size=10000 \
    --filter_min_total=20

echo "=== Full training complete. Results in eval_output/full/ ==="
