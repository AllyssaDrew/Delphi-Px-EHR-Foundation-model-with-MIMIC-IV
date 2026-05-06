"""
Phase E: NoteProjector warmup (1000 steps, backbone frozen).
Resumes from checkpoints/mimic_v6_init/ckpt.pt.
"""
import time

out_dir       = 'checkpoints/mimic_v6_phase_e'
eval_interval = 200
eval_iters    = 50
log_interval  = 20
always_save_checkpoint = True   # save on every improvement

wandb_log      = False
wandb_project  = 'delphi_v6'
wandb_run_name = 'v6_phase_e_' + str(int(time.time()))

phase        = 'E'
max_iters    = 1000
learning_rate = 1e-3   # NoteProjector LR
note_projector_lr = None  # unused in Phase E (single optimizer group)

weight_decay = 1e-1
beta2        = 0.99
grad_clip    = 1.0

batch_size   = 128
block_size   = 128
dtype        = 'float32'
device       = 'cuda'
compile      = False
