"""
DelphiV4 training config — MIMIC-IV (stable run)
  - Full 50k steps from scratch with numerically-stable Weibull NLL
  - Fixes vs prior runs:
      1. log_hazard clamped at max=80 (prevents float32 overflow → NaN)
      2. per-token Weibull NLL clamped at max=50 nats (prevents batch mean explosion)
  - All other hyperparams unchanged from v4 scratch run
"""
import time

out_dir     = 'checkpoints/mimic_v4_stable'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v4_stable_' + str(int(time.time()))

dataset    = 'mimic_data_v4'
batch_size = 128
gradient_accumulation_steps = 1
block_size = 128
data_fraction = 1.0

model_version    = 4
n_layer          = 12
n_head           = 12
n_embd           = 192
n_summary        = 4
n_overflow       = 64
dropout          = 0.0
bias             = False
time_loss_weight = 1.0

init_from      = 'scratch'
learning_rate  = 6e-4
min_lr         = 6e-5
warmup_iters   = 2000
max_iters      = 50000
lr_decay_iters = 50000
weight_decay   = 2e-1
beta2          = 0.99
grad_clip      = 1.0

token_dropout      = 0.0
t_min              = 0.1
mask_ties          = True
no_event_token_rate = 5
ignore_tokens = [0, 2, 3, 4, 5, 6, 7]

device  = 'cuda'
dtype   = 'float32'
compile = False
