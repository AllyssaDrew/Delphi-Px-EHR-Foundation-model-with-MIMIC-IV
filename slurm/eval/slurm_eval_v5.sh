#!/bin/bash
#SBATCH --job-name=delphi_v5_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_v5_%j.log
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

mkdir -p $PIPE/logs $PIPE/eval_output/v5_finetune $PIPE/eval_output/v4_stable_gm

echo "=== Delphi v5 AUC Evaluation ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd $DELPHI

# ── Eval 1: v5 model with GM activated (select='right') ──────────────────────
echo "=== Eval 1: v5 finetune + GM (select=right) ==="
echo "Start: $(date)"
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v5 \
    --output_path=$PIPE/eval_output/v5_finetune \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v5_finetune/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v5/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=50 \
    --gm_eval
echo "Done: $(date)"
echo ""

# ── Eval 2: v4 stable baseline with GM activated (same protocol, for fair comparison) ──
echo "=== Eval 2: v4 stable + GM (select=right, same protocol as v5) ==="
echo "Start: $(date)"
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v4 \
    --output_path=$PIPE/eval_output/v4_stable_gm \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v4_stable/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v4/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=50 \
    --gm_eval
echo "Done: $(date)"
echo ""

echo "=== Evaluation complete ==="
echo "v5 results:          $PIPE/eval_output/v5_finetune/"
echo "v4 baseline (GM):    $PIPE/eval_output/v4_stable_gm/"
echo ""

# Quick AUC summary
$PYBIN -c "
import pandas as pd, warnings; warnings.filterwarnings('ignore')

for label, path in [('v5 finetune', '$PIPE/eval_output/v5_finetune'),
                    ('v4 stable (GM protocol)', '$PIPE/eval_output/v4_stable_gm')]:
    try:
        df = pd.read_parquet(f'{path}/df_both.parquet')
        matched = df[df['ICD-10 Chapter (short)'].str.contains('IX\.|II\. N', regex=True)]
        ch9 = df[df['ICD-10 Chapter (short)'].str.startswith('IX.')]
        neo = df[df['ICD-10 Chapter (short)'].str.startswith('II.')]
        overall = df[~df['ICD-10 Chapter (short)'].isin(['Technical','Lab Values','Death'])]
        print(f'{label}:')
        print(f'  Overall matched AUC: {overall[\"auc\"].mean():.4f}')
        print(f'  Chapter IX (Circulatory): {ch9[\"auc\"].mean():.4f}')
        print(f'  Chapter II (Neoplasms):   {neo[\"auc\"].mean():.4f}')
        print()
    except Exception as e:
        print(f'{label}: {e}')
"
