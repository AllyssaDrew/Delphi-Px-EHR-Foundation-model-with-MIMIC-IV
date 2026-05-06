"""
DelphiV5 extended finetune — 5,000 more steps at constant LR=5e-6.

Resumes from checkpoints/mimic_v5_extended/ckpt.pt
(which is a copy of the step-9000 best checkpoint from Phase 2).

Rationale: Phase 2 val loss was still slowly converging at step 9000
(slope -0.0054/500 steps, small train-val gap 0.045). Neoplasm AUC
regressed vs v4; low-frequency chapters may need more training.
Constant LR avoids over-decaying into a worse local basin.
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

out_dir = 'checkpoints/mimic_v5_extended'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v5_ext_' + str(int(time.time()))

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

init_from = 'resume'   # resumes from out_dir/ckpt.pt (step-9000 best)

# Constant LR: set learning_rate == min_lr so cosine schedule is flat
learning_rate  = 5e-6
min_lr         = 5e-6
warmup_iters   = 0
max_iters      = 14000   # 9000 (resume iter) + 5000 extended steps
lr_decay_iters = 14000
weight_decay   = 2e-1
beta2          = 0.99
grad_clip      = 1.0

token_dropout       = 0.0
t_min               = 0.1
mask_ties           = True
no_event_token_rate = 5

freeze_backbone   = False
freeze_vocab_rows = 0

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
