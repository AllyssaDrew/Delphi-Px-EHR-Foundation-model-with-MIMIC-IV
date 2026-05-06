"""
Phase E v2: NoteProjector warmup (1000 steps) with diversity regularisation.
Starts fresh from checkpoints/mimic_v6_init/ckpt.pt.
lambda_div=1.0 penalises near-constant projection direction.
"""
import time

out_dir       = 'checkpoints/mimic_v6_phase_e_v2'
eval_interval = 250
eval_iters    = 50
log_interval  = 50
always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'delphi_v6'
wandb_run_name = 'v6_phase_e_v2_' + str(int(time.time()))

phase               = 'E'
max_iters           = 1000
learning_rate       = 1e-3

weight_decay = 1e-1
beta2        = 0.99
grad_clip    = 1.0

batch_size   = 128
block_size   = 128
dtype        = 'float32'
device       = 'cuda'
compile      = False

lambda_div   = 1.0
