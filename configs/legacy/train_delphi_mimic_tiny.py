"""
Quick-validation config — Option A data (data/mimic_data)
Tiny model (2 layers, embd=64) to verify AUC > sex+age baseline before full run.
Expected training time: ~15 min on 1 GPU.
"""
import time

out_dir     = 'checkpoints/mimic_tiny'
eval_interval = 200
eval_iters    = 30
log_interval  = 50
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_tiny_' + str(int(time.time()))

# Data — Option A (full, unfiltered)
dataset    = 'mimic_data'
batch_size = 256
block_size = 48
data_fraction = 1.0

# Tiny model
n_layer  = 2
n_head   = 4
n_embd   = 64
dropout  = 0.1
bias     = False
vocab_size = 2010          # vocab_size+1: get_batch shifts all tokens +1, so embedding needs one extra slot

# Optimizer
learning_rate  = 3e-3
max_iters      = 3000
lr_decay_iters = 3000
min_lr         = 3e-4
weight_decay   = 1e-1
beta2          = 0.99
warmup_iters   = 300
grad_clip      = 1.0

# Delphi-specific
token_dropout      = 0.0
t_min              = 0.1
mask_ties          = True
no_event_token_rate = 5
# After get_batch +1 shift: 0=out-of-mask, 2=No_event, 3=Female, 4=Male, 5-7=BMI
ignore_tokens = [0, 2, 3, 4, 5, 6, 7]

device = 'cuda'
dtype  = 'float32'
compile = False
