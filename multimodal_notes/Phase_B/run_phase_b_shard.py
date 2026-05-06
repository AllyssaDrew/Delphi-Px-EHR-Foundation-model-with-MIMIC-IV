#!${PYTHON}
import os
from pathlib import Path

# ── Portable path configuration ────────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory that contains both
# mimic_pipeline/ and Delphi/Delphi-main/ as siblings.
#   export DELPHI_PROJECT_ROOT=/your/project/root
# Alternatively MIMIC_PIPELINE_DIR and DELPHI_DIR can be set individually.
_ROOT        = Path(os.environ.get('DELPHI_PROJECT_ROOT',
                                    Path(__file__).resolve().parents[2]))
PIPELINE_DIR = Path(os.environ.get('MIMIC_PIPELINE_DIR',
                                    _ROOT / 'mimic_pipeline'))
DELPHI_DIR   = Path(os.environ.get('DELPHI_DIR',
                                    _ROOT / 'Delphi' / 'Delphi-main'))
# ──────────────────────────────────────────────────────────────────────────────

"""
Phase B: Clinical-Longformer inference for discharge summaries.
Processes one shard (subject_id % N_SHARDS == SHARD_ID).

Usage (via SLURM array or direct):
  python run_phase_b_shard.py --shard 0 --n_shards 8
  python run_phase_b_shard.py --shard 3 --n_shards 8

Output:
  Phase_B/embeddings_shard_{SHARD_ID}.h5  — keys: "{subject_id}_{hadm_id}"
                                             values: float32 (768,)
"""
import argparse, os, sys, time

import numpy as np
import pandas as pd
import h5py
import torch
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--shard',    type=int, required=True)
parser.add_argument('--n_shards', type=int, default=8)
args = parser.parse_args()

SHARD_ID  = args.shard
N_SHARDS  = args.n_shards

PIPE  = PIPELINE_DIR
OUT   = PIPE / 'multimodal_notes/Phase_B'
OUT.mkdir(parents=True, exist_ok=True)

LINKAGE_CSV  = PIPE / 'multimodal_notes/Phase_A/hadm_linkage_map.csv'
DISC_CSV     = Path(os.environ.get('MIMIC_NOTE_DIR', _ROOT / 'Data')) / 'note' / 'discharge.csv'
MODEL_NAME   = 'yikuan8/Clinical-Longformer'
MAX_LENGTH   = 4096
BATCH_SIZE   = 4
OUT_HDF5     = OUT / f'embeddings_shard_{SHARD_ID}.h5'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}  |  Shard {SHARD_ID}/{N_SHARDS}", flush=True)

# ── Load linkage map (only v5 patients) ──────────────────────────────────────
print("Loading linkage map...", flush=True)
linkage = pd.read_csv(LINKAGE_CSV, usecols=['subject_id', 'hadm_id'])
shard_mask = (linkage['subject_id'] % N_SHARDS) == SHARD_ID
shard_df = linkage[shard_mask].reset_index(drop=True)
print(f"  This shard: {len(shard_df):,} hadm_ids", flush=True)

# ── Load discharge texts ──────────────────────────────────────────────────────
print("Loading discharge texts...", flush=True)
disc = pd.read_csv(DISC_CSV, usecols=['subject_id', 'hadm_id', 'text'])
disc = disc[disc['hadm_id'].isin(shard_df['hadm_id'])].reset_index(drop=True)
print(f"  Matched texts: {len(disc):,}", flush=True)

# ── Check already-done keys (resume support) ─────────────────────────────────
done_keys = set()
if OUT_HDF5.exists():
    with h5py.File(OUT_HDF5, 'r') as f:
        done_keys = set(f.keys())
    print(f"  Resuming: {len(done_keys):,} already done", flush=True)

todo = disc[~disc.apply(lambda r: f"{r['subject_id']}_{r['hadm_id']}" in done_keys, axis=1)]
print(f"  To process: {len(todo):,}", flush=True)

if len(todo) == 0:
    print("Shard already complete.", flush=True)
    sys.exit(0)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)
print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

# ── Inference ─────────────────────────────────────────────────────────────────
t0 = time.time()
n_done = 0

with h5py.File(OUT_HDF5, 'a') as hf:
    rows = todo.to_dict('records')
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start: batch_start + BATCH_SIZE]
        texts = [str(r['text']) for r in batch]

        t1 = time.time()
        enc = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='longest',
            return_tensors='pt'
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Longformer: set global attention on [CLS] token
        if 'global_attention_mask' not in enc:
            ga = torch.zeros_like(enc['input_ids'])
            ga[:, 0] = 1  # [CLS] gets global attention
            enc['global_attention_mask'] = ga

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=(device=='cuda')):
            out = model(**enc)

        # [CLS] embedding: last_hidden_state[:, 0, :]
        cls_emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()  # (bs, 768)

        for i, r in enumerate(batch):
            key = f"{r['subject_id']}_{r['hadm_id']}"
            hf.create_dataset(key, data=cls_emb[i], dtype='float32')

        n_done += len(batch)
        elapsed = time.time() - t1
        total_elapsed = time.time() - t0
        rate = n_done / total_elapsed
        remaining = (len(todo) - n_done) / rate if rate > 0 else 0

        if n_done % 100 == 0 or n_done <= 5:
            print(f"  [{n_done}/{len(todo)}] batch in {elapsed:.1f}s | "
                  f"rate={rate:.1f}/s | ETA={remaining/3600:.1f}h", flush=True)

print(f"\nShard {SHARD_ID} complete: {n_done} embeddings in {(time.time()-t0)/60:.1f} min", flush=True)
