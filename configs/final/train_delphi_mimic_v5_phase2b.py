"""
DelphiV5 Phase 2b — full model fine-tune from Phase 1b checkpoint.

Phase 1b warmed up wte+lm_head rows 1469-1510 (42 new ICD codes) with
LR=1e-3 for 1000 steps while keeping the backbone frozen. This improved
Neoplasm AUC from 0.7845 → 0.8007 but caused a small Chapter IX regression
(0.7817 → 0.7733) due to high-LR perturbation of the embedding space.

Phase 2b: unfreeze the backbone and fine-tune at LR=5e-6 (constant) so the
backbone can adapt to the new embeddings, recover Chapter IX, and continue
pushing Neoplasm toward 0.81.

Resumes from checkpoints/mimic_v5_phase2b/ckpt.pt
(which is a copy of the Phase 1b checkpoint at iter 12000).
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

out_dir = 'checkpoints/mimic_v5_phase2b'
eval_interval = 250
eval_iters    = 50
log_interval  = 50
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v5_p2b_' + str(int(time.time()))

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

init_from = 'resume'   # resumes from out_dir/ckpt.pt (Phase 1b iter 12000)

# Constant LR: conservative to preserve Phase 1b gains while allowing backbone adaptation
learning_rate  = 5e-6
min_lr         = 5e-6
warmup_iters   = 0
max_iters      = 15000   # 12000 (resume iter) + 3000 steps
lr_decay_iters = 15000
weight_decay   = 2e-1
beta2          = 0.99
grad_clip      = 1.0

token_dropout       = 0.0
t_min               = 0.1
mask_ties           = True
no_event_token_rate = 5

# Full model — no freezing
freeze_backbone   = False
freeze_vocab_rows = 0
warm_vocab_rows   = []

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
