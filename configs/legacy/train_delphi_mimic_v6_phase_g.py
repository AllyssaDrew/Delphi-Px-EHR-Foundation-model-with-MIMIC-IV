"""
Phase G: Ablation — joint training (5000 steps) with random-patient note embeddings.
Starts from checkpoints/mimic_v6_phase_e/ckpt.pt (same init as Phase F).
The only difference from Phase F: ablation_mean_patient=True.
"""
import time

out_dir       = 'checkpoints/mimic_v6_phase_g'
eval_interval = 250
eval_iters    = 50
log_interval  = 50
always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'delphi_v6'
wandb_run_name = 'v6_phase_g_' + str(int(time.time()))

phase               = 'G'
max_iters           = 5000
learning_rate       = 5e-6    # backbone LR (matches Phase F)
note_projector_lr   = 1e-4    # NoteProjector LR (matches Phase F)

weight_decay = 2e-1
beta2        = 0.99
grad_clip    = 1.0

batch_size   = 128
block_size   = 128
dtype        = 'float32'
device       = 'cuda'
compile      = False

ablation_mean_patient = True
