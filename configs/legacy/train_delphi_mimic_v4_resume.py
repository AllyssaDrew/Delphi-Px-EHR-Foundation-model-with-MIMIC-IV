"""
DelphiV4 training config — MIMIC-IV (resume from step 9000)
  - Resumes ckpt.pt (step 9000, val 8.7657) with fixed Weibull NLL
  - Fix: log-space hazard computation prevents float32 overflow / NaN
  - Same hyperparams as v4 scratch run
"""
import time

out_dir     = 'checkpoints/mimic_v4'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v4_resume_' + str(int(time.time()))

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

init_from      = 'resume'
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
