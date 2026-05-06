"""
v3 training config — mimic_data_v3 dataset
Changes vs v2:
  - model_version=3: DelphiV3 (Weibull time head + BMI aux loss)
  - vocab_size passed via --vocab_size=N on command line (from meta.pkl after preprocessing)
  - block_size 64
  - ignore_tokens: 0,2,3,4,5-7 (lab tokens predicted, not ignored)
  - learning_rate tuned for fresh training
"""
import time

out_dir     = 'checkpoints/mimic_v3'
eval_interval = 500
eval_iters    = 50
log_interval  = 100
always_save_checkpoint = False

wandb_log      = False
wandb_project  = 'delphi_mimic'
wandb_run_name = 'mimic_v3_' + str(int(time.time()))

# Data — v3 dataset (first-occurrence dedup, freq≥25, lab tokens)
dataset    = 'mimic_data_v3'
batch_size = 128
block_size = 64
data_fraction = 1.0

# Model version — uses DelphiV3 (Weibull + BMI aux)
model_version    = 3
n_layer          = 12
n_head           = 12
n_embd           = 192
dropout          = 0.1
bias             = False
bmi_aux_weight   = 0.15
time_loss_weight = 1.0

# vocab_size is NOT set here — pass --vocab_size=N on the command line:
#   python -c "import pickle; m=pickle.load(open('data/mimic_data_v3/meta.pkl','rb')); print(m['vocab_size']+1)"

# Optimizer — same schedule as v2
learning_rate  = 2e-3
max_iters      = 40000
lr_decay_iters = 40000
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

# ignore_tokens (shifted IDs after get_batch +1):
#   0=out-of-mask, 2=No_event, 3=Female, 4=Male, 5-7=BMI
# ICU(8), ED(9), ICD codes, lab tokens: all predicted
ignore_tokens = [0, 2, 3, 4, 5, 6, 7]

device  = 'cuda'
dtype   = 'float32'
compile = False
