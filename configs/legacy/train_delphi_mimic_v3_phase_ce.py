"""
Phase C+E config — block_size 64→128 + warm restart from v3 40k checkpoint
  - block_size = 128 (Phase C: context extension)
  - init_from = 'resume' from checkpoints/mimic_v3/ckpt.pt
  - learning_rate = 2e-4, warmup_iters = 40000 → LR peaks at resume point
  - max_iters = 60000 → 20k additional steps with cosine decay to 2e-5
"""
import time

out_dir     = 'checkpoints/mimic_v3'   # continue in same dir
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v3_phase_ce_' + str(int(time.time()))

dataset    = 'mimic_data_v3'
batch_size = 128
block_size = 128           # Phase C: extended context
data_fraction = 1.0

model_version    = 3
n_layer          = 12
n_head           = 12
n_embd           = 192
dropout          = 0.1
bias             = False
bmi_aux_weight   = 0.15
time_loss_weight = 1.0

# Phase E warm restart: resume from iter 40000
# warmup_iters=40000 → at iter 40000 LR = learning_rate (peak), then cosine decay to min_lr
init_from      = 'resume'
learning_rate  = 2e-4
min_lr         = 2e-5
warmup_iters   = 40000
max_iters      = 60000
lr_decay_iters = 60000
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
