"""
DelphiV5 Phase 1b — targeted warm-up for the 42 new ICD code positions.

Problem: the 42 new ICD codes (stored IDs 1468-1509, model space 1469-1510)
landed in the "frozen range" of Phase 1 (freeze_vocab_rows=1494). Their wte
and lm_head rows were never warmed during Phase 1 — only 10k+5k steps of
full fine-tuning with 29-91 noisy GEM-mapped samples each.
Result: 6 Neoplasm codes are actively anti-predictive (AUC 0.03-0.36).

Fix: warm ONLY rows 1469-1510 in wte + lm_head, keep the backbone frozen.
This lets the embeddings and output projections for the 42 new codes converge
from the backbone's existing hidden representations, without disturbing the
well-trained codes or the backbone itself.

Resumes from checkpoints/mimic_v5_phase1b/ckpt.pt
(which is a copy of the step-11000 extended best checkpoint).
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

out_dir = 'checkpoints/mimic_v5_phase1b'
eval_interval = 250
eval_iters    = 50
log_interval  = 50
always_save_checkpoint = True   # save each eval even if val doesn't improve

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v5_p1b_' + str(int(time.time()))

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

init_from = 'resume'   # resumes from out_dir/ckpt.pt (copy of v5_extended step-11000)

# High constant LR — new rows start from poorly-initialized v4 state
learning_rate  = 1e-3
min_lr         = 1e-3
warmup_iters   = 0
max_iters      = 12000   # 11000 (resume iter) + 1000 warm-up steps
lr_decay_iters = 12000
weight_decay   = 0.0    # no decay for embedding warm-up
beta2          = 0.99
grad_clip      = 1.0

token_dropout       = 0.0
t_min               = 0.1
mask_ties           = True
no_event_token_rate = 5

# Phase 1b: freeze backbone + mask lm_head, update ONLY rows 1469-1510 (42 new ICD codes)
freeze_backbone   = True
freeze_vocab_rows = 0        # disabled — use warm_vocab_rows instead
warm_vocab_rows   = list(range(1469, 1511))   # model-space rows for stored IDs 1468-1509

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
