#!/bin/bash
#SBATCH --job-name=eval_v5p1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_v5_phase1b_%j.log
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
export PYTHONUNBUFFERED=1

mkdir -p $PIPE/logs $PIPE/eval_output/v5_phase1b

echo "=== Delphi v5 Phase 1b AUC Evaluation ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

cd $DELPHI

echo "=== v5 phase1b + GM (select=right) ==="
echo "Start: $(date)"
$PYBIN evaluate_auc.py \
    --input_path=$PIPE/data/mimic_data_v5 \
    --output_path=$PIPE/eval_output/v5_phase1b \
    --model_ckpt_path=$DELPHI/checkpoints/mimic_v5_phase1b/ckpt.pt \
    --labels_file=$PIPE/data/mimic_data_v5/mimic_labels.csv \
    --no_event_token_rate=5 \
    --filter_min_total=50 \
    --gm_eval
echo "Done: $(date)"
echo ""

echo "=== AUC Summary ==="
$PYBIN -c "
import pandas as pd, warnings; warnings.filterwarnings('ignore')

for label, path in [
    ('v5 phase1b   (targeted warm-up)', '$PIPE/eval_output/v5_phase1b'),
    ('v5 extended  (step-11000)',        '$PIPE/eval_output/v5_extended'),
    ('v5 finetune  (step-9000)',         '$PIPE/eval_output/v5_finetune'),
    ('v4 stable    (baseline)',          '$PIPE/eval_output/v4_stable_gm'),
]:
    try:
        df = pd.read_parquet(f'{path}/df_both.parquet')
        excl = ['Technical', 'Lab Values', 'Death']
        overall = df[~df['ICD-10 Chapter (short)'].isin(excl)]
        ch9  = df[df['ICD-10 Chapter (short)'].str.startswith('IX.')]
        neo  = df[df['ICD-10 Chapter (short)'].str.startswith('II.')]
        print(f'{label}:  Overall={overall[\"auc\"].mean():.4f}  IX={ch9[\"auc\"].mean():.4f}  Neo={neo[\"auc\"].mean():.4f}')
        # Show bottom-5 neoplasm
        neo_sorted = neo[['name','auc']].sort_values('auc')
        worst = '  |  '.join(f'{r[\"name\"].split()[0]}: {r[\"auc\"]:.3f}' for _, r in neo_sorted.head(5).iterrows())
        print(f'  worst Neo: {worst}')
    except Exception as e:
        print(f'{label}: {e}')
"
