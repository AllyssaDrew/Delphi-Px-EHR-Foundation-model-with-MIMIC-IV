"""
Phase F v2: Joint training (5000 steps) with diversity regularisation.
Resumes from checkpoints/mimic_v6_phase_e_v3/ckpt.pt (lambda_div=5.0 warmup).
"""
import time

out_dir       = 'checkpoints/mimic_v6_phase_f_v2'
eval_interval = 250
eval_iters    = 50
log_interval  = 50
always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'delphi_v6'
wandb_run_name = 'v6_phase_f_v2_' + str(int(time.time()))

phase               = 'F'
max_iters           = 5000
learning_rate       = 5e-6
note_projector_lr   = 1e-4

weight_decay = 2e-1
beta2        = 0.99
grad_clip    = 1.0

batch_size   = 128
block_size   = 128
dtype        = 'float32'
device       = 'cuda'
compile      = False

lambda_div   = 5.0
