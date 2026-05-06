"""
DelphiV5 Phase 2 finetune config — full model finetune (10,000 steps).

Resumes from checkpoints/mimic_v5_finetune/ckpt.pt (output of Phase 1).
All parameters unfrozen; standard cosine LR schedule 1e-4 → 1e-5.

v5 changes vs v4:
  - eGFR (4-level, CKD-EPI 2021) replaces Creatinine (3-level): +1 new token row
  - TroponinI replaces TroponinT (same token IDs)
  - Pre-2015 ICD-9 one-to-one GEM events (longer sequences)
  - Lab tokens in ignore_tokens (not predicted by CE loss)
  - Any new ICD codes from pre-2015 appended as additional token rows
"""
import os
from pathlib import Path

# ── Portable path configuration ────────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory that contains both
# mimic_pipeline/ and Delphi/Delphi-main/ as siblings.
#   export DELPHI_PROJECT_ROOT=/your/project/root
# Alternatively MIMIC_PIPELINE_DIR and DELPHI_DIR can be set individually.
_ROOT        = Path(os.environ.get('DELPHI_PROJECT_ROOT',
                                    Path(__file__).resolve().parents[2]))
PIPELINE_DIR = Path(os.environ.get('MIMIC_PIPELINE_DIR',
                                    _ROOT / 'mimic_pipeline'))
DELPHI_DIR   = Path(os.environ.get('DELPHI_DIR',
                                    _ROOT / 'Delphi' / 'Delphi-main'))
# ──────────────────────────────────────────────────────────────────────────────

import time
import pickle, os

out_dir = 'checkpoints/mimic_v5_finetune'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v5_ft2_' + str(int(time.time()))

dataset    = 'mimic_data_v5'
batch_size = 128
gradient_accumulation_steps = 1
block_size = 128
data_fraction = 1.0

model_version = 4
n_layer       = 12
n_head        = 12
n_embd        = 192
n_summary     = 4
n_overflow    = 64
dropout       = 0.0
bias          = False
time_loss_weight = 1.0

init_from = 'resume'   # resumes from out_dir/ckpt.pt (Phase 1 output)

learning_rate  = 1e-4
min_lr         = 1e-5
warmup_iters   = 500    # no warmup needed (resumes past it); kept for LR schedule math
max_iters      = 10500  # 500 (Phase 1) + 10000 full-model steps
lr_decay_iters = 10500
weight_decay   = 2e-1
beta2          = 0.99
grad_clip      = 1.0

token_dropout       = 0.0
t_min               = 0.1
mask_ties           = True
no_event_token_rate = 5

freeze_backbone   = False
freeze_vocab_rows = 0

# ignore_tokens: lab tokens (model-space) excluded from CE loss.
# Read from v5 meta.pkl for accuracy; fallback assumes DEATH=1468, 25 lab tokens.
_PIPE   = '${MIMIC_PIPELINE_DIR}'
_V5META = os.path.join(_PIPE, 'data/mimic_data_v5/meta.pkl')
try:
    with open(_V5META, 'rb') as _f:
        _meta = pickle.load(_f)
    ignore_tokens = _meta['ignore_tokens']
except FileNotFoundError:
    ignore_tokens = [0, 2, 3, 4, 5, 6, 7] + list(range(1470, 1495))

device  = 'cuda'
dtype   = 'float32'
compile = False
