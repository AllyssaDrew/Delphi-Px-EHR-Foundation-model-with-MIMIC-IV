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
Merge Phase B shard HDF5 files into a single note_embeddings.h5.
Run after all SLURM array jobs finish.

Usage: python merge_shards.py [--n_shards 8]
"""
import argparse, time
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--n_shards', type=int, default=8)
args = parser.parse_args()

OUT = PIPELINE_DIR / 'multimodal_notes/Phase_B'
FINAL = OUT / 'note_embeddings.h5'

shard_files = [OUT / f'embeddings_shard_{i}.h5' for i in range(args.n_shards)]
missing = [f for f in shard_files if not f.exists()]
if missing:
    print(f"Missing shards: {missing}")
    print("Run Phase B SLURM jobs first.")
    exit(1)

print(f"Merging {args.n_shards} shards → {FINAL}", flush=True)
t0 = time.time()
total = 0

with h5py.File(FINAL, 'w') as out_f:
    for sf in shard_files:
        with h5py.File(sf, 'r') as in_f:
            n = len(in_f.keys())
            print(f"  {sf.name}: {n:,} keys", flush=True)
            for key in in_f.keys():
                out_f.create_dataset(key, data=in_f[key][:], dtype='float32')
            total += n

print(f"\nTotal embeddings: {total:,}  ({(time.time()-t0)/60:.1f} min)", flush=True)

# Sanity check: embedding norm distribution
print("\nSanity check (random 1000 embeddings):", flush=True)
with h5py.File(FINAL, 'r') as f:
    keys = list(f.keys())
    sample_keys = keys[:1000]
    norms = [np.linalg.norm(f[k][:]) for k in sample_keys]
norms = np.array(norms)
print(f"  norm: mean={norms.mean():.3f}  std={norms.std():.3f}  "
      f"min={norms.min():.3f}  max={norms.max():.3f}", flush=True)

import pandas as pd
stats = pd.DataFrame({'norm': norms})
stats.to_csv(OUT / 'embedding_stats.csv', index=False)
print(f"  embedding_stats.csv saved.", flush=True)
print("Merge complete.", flush=True)
