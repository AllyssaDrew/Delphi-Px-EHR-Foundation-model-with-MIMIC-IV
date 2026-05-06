"""
DelphiV5 Phase 1 finetune config — vocab warm-up (500 steps).

Purpose: train only the new embedding rows (v5 vocab extension) while
  freezing the backbone and masking gradients for existing vocab rows.
  This lets the new tokens converge from random init before full finetune.

Settings:
  freeze_backbone   = True   → backbone (all non-wte params) frozen
  freeze_vocab_rows = 1494   → wte rows 0..1493 get zero grad after backward
                               (= v4 vocab_size; only rows 1494+ update)
  learning_rate     = 1e-3   → high LR since new rows start from random init
  max_iters         = 500    → short warm-up

After this run, checkpoints/mimic_v5_finetune/ckpt.pt is ready for Phase 2.
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

out_dir = 'checkpoints/mimic_v5_finetune'
eval_interval = 500   # eval once at the end
eval_iters    = 25
log_interval  = 50
always_save_checkpoint = True   # save even if val doesn't improve

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v5_phase1_' + str(int(time.time()))

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

init_from = 'resume'   # loads checkpoints/mimic_v5_finetune/ckpt.pt (expand_vocab output)

learning_rate  = 1e-3
min_lr         = 1e-3   # constant LR (no decay during warm-up)
warmup_iters   = 0
max_iters      = 500
lr_decay_iters = 500
weight_decay   = 0.0    # no decay for embeddings
beta2          = 0.99
grad_clip      = 1.0

token_dropout       = 0.0
t_min               = 0.1
mask_ties           = True
no_event_token_rate = 5

# Vocab extension freeze settings (v4 vocab_size = 1494 model-space rows)
freeze_backbone   = True
freeze_vocab_rows = 1494

import pickle as _pickle, os as _os
_PIPE   = '${MIMIC_PIPELINE_DIR}'
_V5META = _os.path.join(_PIPE, 'data/mimic_data_v5/meta.pkl')
try:
    with open(_V5META, 'rb') as _f:
        ignore_tokens = _pickle.load(_f)['ignore_tokens']
except FileNotFoundError:
    ignore_tokens = [0, 2, 3, 4, 5, 6, 7] + list(range(1512, 1537))

device  = 'cuda'
dtype   = 'float32'
compile = False
