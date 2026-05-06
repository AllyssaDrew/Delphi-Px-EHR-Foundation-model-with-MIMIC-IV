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
Phase C: Rebuild v5 binary sequences → v6 with NOTE tokens inserted.

For each admission that has a discharge note (from hadm_linkage_map.csv),
inserts NOTE token (stored=1536) at age_days_at_disch.
Follows same temporal logic as v5 (test cutoff preserved).

Outputs:
  data/mimic_data_v6/train.bin
  data/mimic_data_v6/val.bin
  data/mimic_data_v6/test_input.bin
  data/mimic_data_v6/test_future.bin
  data/mimic_data_v6/meta_v6.pkl
  data/mimic_data_v6/patient_splits.csv  (copy from v5, unchanged)
  data/mimic_data_v6/test_cutoffs.csv    (recomputed for new patient_idx)
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd

PIPE    = PIPELINE_DIR
V5_DIR  = PIPE / 'data/mimic_data_v5'
V6_DIR  = PIPE / 'data/mimic_data_v6'
V6_DIR.mkdir(parents=True, exist_ok=True)

NOTE_TOKEN_STORED = 1536   # model ID = 1537, new vocab_size stored = 1537

t0 = time.time()

# ── Load v5 meta and patient splits ───────────────────────────────────────────
print("Loading v5 meta and splits...", flush=True)
with open(V5_DIR / 'meta.pkl', 'rb') as f:
    meta_v5 = pickle.load(f)

splits = pd.read_csv(V5_DIR / 'patient_splits.csv')

# For each split, build sorted subject_id list → patient_idx mapping
# (matches build_bin_array in 02_preprocess_v5.py: np.unique sorts ascending)
def make_pid_maps(split_name):
    sids = sorted(splits[splits['split'] == split_name]['subject_id'].tolist())
    sid_to_pidx = {sid: i for i, sid in enumerate(sids)}
    pidx_to_sid = {i: sid for i, sid in enumerate(sids)}
    return sid_to_pidx, pidx_to_sid

train_s2p, train_p2s = make_pid_maps('train')
val_s2p,   val_p2s   = make_pid_maps('val')
test_s2p,  test_p2s  = make_pid_maps('test')

# ── Load hadm linkage map (NOTE events) ───────────────────────────────────────
print("Loading hadm linkage map...", flush=True)
linkage = pd.read_csv(PIPE / 'multimodal_notes/Phase_A/hadm_linkage_map.csv')
print(f"  {len(linkage):,} NOTE events total", flush=True)

def note_events_for_split(split_name):
    sub = linkage[linkage['split'] == split_name][['subject_id', 'age_days_at_disch', 'hadm_id']].copy()
    sub = sub.rename(columns={'age_days_at_disch': 'age_days'})
    sub['token_id'] = np.uint32(NOTE_TOKEN_STORED)
    return sub[['subject_id', 'age_days', 'token_id', 'hadm_id']]

# ── Helper: load bin → DataFrame with subject_id ─────────────────────────────
def load_bin_with_sids(bin_path, pidx_to_sid):
    arr = np.memmap(str(bin_path), dtype=np.int32, mode='r').reshape(-1, 3)
    df = pd.DataFrame({
        'subject_id': [pidx_to_sid[int(x)] for x in arr[:, 0]],
        'age_days':   arr[:, 1].astype(np.int64),
        'token_id':   arr[:, 2].astype(np.uint32),
    })
    return df

# ── Helper: build binary array (same as v5) ───────────────────────────────────
def build_bin_array(df):
    df = df[['subject_id', 'age_days', 'token_id']].copy()
    df = df.sort_values(['subject_id', 'age_days']).reset_index(drop=True)
    uid, inv = np.unique(df['subject_id'].values, return_inverse=True)
    arr = np.stack([
        inv.astype(np.uint32),
        df['age_days'].clip(lower=0).values.astype(np.uint32),
        df['token_id'].values.astype(np.uint32),
    ], axis=1).astype(np.uint32)
    return arr, uid

# ══════════════════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcessing train split...", flush=True)
train_v5 = load_bin_with_sids(V5_DIR / 'train.bin', train_p2s)
print(f"  v5 train events: {len(train_v5):,}", flush=True)

note_train = note_events_for_split('train')[['subject_id', 'age_days', 'token_id']]
print(f"  NOTE events to add: {len(note_train):,}", flush=True)

train_v6 = pd.concat([train_v5, note_train], ignore_index=True)
train_arr, train_uids = build_bin_array(train_v6)
train_arr.tofile(str(V6_DIR / 'train.bin'))
print(f"  v6 train.bin: {len(train_arr):,} events  {train_arr.nbytes/1e6:.1f} MB", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Val
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcessing val split...", flush=True)
val_v5 = load_bin_with_sids(V5_DIR / 'val.bin', val_p2s)
note_val = note_events_for_split('val')[['subject_id', 'age_days', 'token_id']]
print(f"  v5 val: {len(val_v5):,}  +NOTE: {len(note_val):,}", flush=True)

val_v6 = pd.concat([val_v5, note_val], ignore_index=True)
val_arr, val_uids = build_bin_array(val_v6)
val_arr.tofile(str(V6_DIR / 'val.bin'))
print(f"  v6 val.bin: {len(val_arr):,} events  {val_arr.nbytes/1e6:.1f} MB", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Test (preserve temporal cutoff logic from v5)
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcessing test split...", flush=True)
test_in_v5  = load_bin_with_sids(V5_DIR / 'test_input.bin', test_p2s)
test_fut_v5 = load_bin_with_sids(V5_DIR / 'test_future.bin', test_p2s)

# Load cutoffs to apply same split to NOTE tokens
cutoffs = pd.read_csv(V5_DIR / 'test_cutoffs.csv')
# cutoffs has: subject_id, last_adm_age, cutoff_age, patient_idx (v5 idx)
cutoff_map = cutoffs.set_index('subject_id')['cutoff_age'].to_dict()

note_test = note_events_for_split('test')[['subject_id', 'age_days', 'token_id']].copy()
note_test['cutoff'] = note_test['subject_id'].map(cutoff_map).fillna(0)
note_test_in  = note_test[note_test['age_days'] <  note_test['cutoff']][['subject_id','age_days','token_id']]
note_test_fut = note_test[note_test['age_days'] >= note_test['cutoff']][['subject_id','age_days','token_id']]
print(f"  NOTE test_input: {len(note_test_in):,}  test_future: {len(note_test_fut):,}", flush=True)

test_in_v6  = pd.concat([test_in_v5,  note_test_in],  ignore_index=True)
test_fut_v6 = pd.concat([test_fut_v5, note_test_fut], ignore_index=True)

test_in_arr,  test_in_uids  = build_bin_array(test_in_v6)
test_fut_arr, test_fut_uids = build_bin_array(test_fut_v6)

test_in_arr.tofile(str(V6_DIR / 'test_input.bin'))
test_fut_arr.tofile(str(V6_DIR / 'test_future.bin'))
print(f"  v6 test_input.bin:  {len(test_in_arr):,} events", flush=True)
print(f"  v6 test_future.bin: {len(test_fut_arr):,} events", flush=True)

# Rewrite test_cutoffs.csv with v6 patient_idx
test_in_pid_map = {pid: idx for idx, pid in enumerate(test_in_uids)}
cutoffs_v6 = cutoffs[['subject_id', 'last_adm_age', 'cutoff_age']].copy()
cutoffs_v6['patient_idx'] = cutoffs_v6['subject_id'].map(test_in_pid_map)
cutoffs_v6.to_csv(V6_DIR / 'test_cutoffs.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Meta v6
# ══════════════════════════════════════════════════════════════════════════════
print("\nWriting meta_v6.pkl...", flush=True)
import copy
meta_v6 = copy.deepcopy(meta_v5)

meta_v6['vocab_size']    = meta_v5['vocab_size'] + 1   # 1536 → 1537 (stored)
meta_v6['NOTE_TOKEN']    = NOTE_TOKEN_STORED            # 1536
# Add NOTE model-space ID (1537) to ignore_tokens
meta_v6['ignore_tokens'] = meta_v5['ignore_tokens'] + [NOTE_TOKEN_STORED + 1]
meta_v6['events'] = {
    'train':       len(train_arr),
    'val':         len(val_arr),
    'test_input':  len(test_in_arr),
    'test_future': len(test_fut_arr),
}
meta_v6['v6_note_coverage'] = {
    'train_note_events': len(note_train),
    'val_note_events':   len(note_val),
    'hadm_coverage_pct': 63.6,
}

with open(V6_DIR / 'meta_v6.pkl', 'wb') as f:
    pickle.dump(meta_v6, f)

# Copy other files from v5
import shutil
for fname in ['patient_splits.csv', 'mimic_labels.csv', 'vocab_stats.csv']:
    shutil.copy(V5_DIR / fname, V6_DIR / fname)
print("  Copied patient_splits.csv, mimic_labels.csv, vocab_stats.csv", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}", flush=True)
print(f"Phase C complete in {elapsed:.1f} min", flush=True)
print(f"  v5 train events:  {len(train_v5):,}  →  v6: {len(train_arr):,}  (+{len(note_train):,} NOTE)", flush=True)
print(f"  v5 val events:    {len(val_v5):,}  →  v6: {len(val_arr):,}  (+{len(note_val):,} NOTE)", flush=True)
print(f"  v5 vocab_size (stored): {meta_v5['vocab_size']}  →  v6: {meta_v6['vocab_size']}", flush=True)
print(f"  NOTE_TOKEN stored={NOTE_TOKEN_STORED}, model={NOTE_TOKEN_STORED+1}", flush=True)
print(f"  Output: {V6_DIR}", flush=True)
