"""
Full training config — Option B+D data (data/mimic_data_v2)
Comparable scale to Delphi-2M.  Set vocab_size after running 02_preprocess_v2.py.
Expected training time: ~2-4 h on 1 × A100 / V100.
"""
import time

out_dir     = 'checkpoints/mimic_full'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_full_' + str(int(time.time()))

# Data — Option B+D (filtered + enriched)
dataset    = 'mimic_data_v2'
batch_size = 128
block_size = 64          # longer than tiny: intra-admission sequences are longer
data_fraction = 1.0

# Model — medium scale (matches original Delphi-2M depth, smaller width)
n_layer  = 12
n_head   = 12
n_embd   = 192
dropout  = 0.1
bias     = False
vocab_size = 2007

# Optimizer
learning_rate  = 2e-3
max_iters      = 20000
lr_decay_iters = 20000
min_lr         = 2e-4
weight_decay   = 2e-1
beta2          = 0.99
warmup_iters   = 1000
grad_clip      = 1.0

# Delphi-specific
token_dropout      = 0.0
t_min              = 0.1
mask_ties          = True
no_event_token_rate = 5
# After get_batch +1 shift: 0=out-of-mask, 2=No_event, 3=Female, 4=Male, 5-7=BMI
# ICU(raw 7→model 8) and ED(raw 8→model 9) are real events — DO NOT ignore them
ignore_tokens = [0, 2, 3, 4, 5, 6, 7]

device  = 'cuda'
dtype   = 'float32'
compile = False
