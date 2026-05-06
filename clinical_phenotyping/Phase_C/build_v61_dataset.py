"""
Phase C: Build Delphi v6.1 binary dataset.

For each admission with extracted phenotype tokens:
  - Insert one synthetic event per token at age_days_at_disch
  - Multiple tokens from the same admission → same timestamp, sorted by token ID

The v5 binary files store (patient_idx, age_days, token_id) as uint32 triples.
Phenotype tokens use their stored IDs (1536–1568).

Outputs:
  mimic_pipeline/data/mimic_data_v61/{train,val,test_input,test_future}.bin
  mimic_pipeline/data/mimic_data_v61/meta_v61.pkl
  mimic_pipeline/data/mimic_data_v61/patient_splits.csv   (copy)
  mimic_pipeline/data/mimic_data_v61/mimic_labels.csv     (copy)
  mimic_pipeline/data/mimic_data_v61/test_cutoffs.csv     (copy)
"""
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


import sys, pickle, shutil
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'Phase_A'))
from phenotype_dict import (
    PHENOTYPE_TOKENS, TOKEN_NAME_TO_ID,
    NEW_VOCAB_SIZE_STORED, FIRST_STORED_ID,
)

PIPE      = PIPELINE_DIR
V5_DIR    = PIPE / 'data/mimic_data_v5'
V61_DIR   = PIPE / 'data/mimic_data_v61'
PHENO_CSV = PIPE / 'clinical_phenotyping/Phase_B/phenotype_tokens.csv'
LINKAGE   = PIPE / 'multimodal_notes/Phase_A/hadm_linkage_map.csv'

V61_DIR.mkdir(parents=True, exist_ok=True)


def load_phenotype_events() -> dict:
    """
    Returns: {(subject_id, hadm_id): [stored_token_id, ...]}
    Only admissions with ≥1 token are included.
    """
    df = pd.read_csv(PHENO_CSV, dtype={'subject_id': int, 'hadm_id': int, 'tokens': str})
    df = df[df['tokens'].notna() & (df['tokens'].str.strip() != '')]

    events = {}
    for _, row in df.iterrows():
        token_ids = [int(t) for t in row['tokens'].split()]
        if token_ids:
            events[(int(row['subject_id']), int(row['hadm_id']))] = sorted(token_ids)
    print(f"  Loaded phenotype events for {len(events):,} admissions", flush=True)
    return events


def build_split(split_name: str, pheno_events: dict, linkage_df: pd.DataFrame):
    """Insert phenotype tokens into one split's binary file."""
    if split_name in ('train', 'val'):
        src = V5_DIR / f'{split_name}.bin'
        dst = V61_DIR / f'{split_name}.bin'
    elif split_name == 'test_input':
        src = V5_DIR / 'test_input.bin'
        dst = V61_DIR / 'test_input.bin'
    elif split_name == 'test_future':
        src = V5_DIR / 'test_future.bin'
        dst = V61_DIR / 'test_future.bin'
    else:
        raise ValueError(split_name)

    data = np.fromfile(str(src), dtype=np.uint32).reshape(-1, 3)
    print(f"\n[{split_name}] {len(data):,} events loaded from {src.name}", flush=True)

    # Build lookup: patient_idx → subject_id (from existing data)
    # patient_idx is stored in col 0; we need to map it to subject_id
    # Use the splits CSV + patient ordering to reconstruct the mapping
    splits_df = pd.read_csv(V5_DIR / 'patient_splits.csv')
    split_label = 'test' if split_name.startswith('test') else split_name
    sids_sorted = sorted(
        splits_df[splits_df['split'] == split_label]['subject_id'].tolist()
    )
    pidx_to_sid = {i: sid for i, sid in enumerate(sids_sorted)}

    # Get linkage for this split's admissions
    # linkage_df has (subject_id, hadm_id, age_days_at_disch, split)
    split_linkage = linkage_df[linkage_df['split'] == split_label]

    # Build: (subject_id, hadm_id) → age_days_at_disch
    hadm_to_age = {
        (int(r.subject_id), int(r.hadm_id)): int(r.age_days_at_disch)
        for r in split_linkage.itertuples()
    }

    # Build new synthetic events to insert
    new_rows = []
    n_tokens_added = 0
    for pidx, sid in pidx_to_sid.items():
        # Find all admissions for this patient in this split
        for (s_id, h_id), token_ids in pheno_events.items():
            if s_id != sid:
                continue
            age_d = hadm_to_age.get((s_id, h_id))
            if age_d is None:
                continue
            # test_future: skip — phenotype tokens are INPUT only
            if split_name == 'test_future':
                continue
            for tok_id in token_ids:
                new_rows.append([pidx, age_d, tok_id])
                n_tokens_added += 1

    print(f"  Inserting {n_tokens_added:,} phenotype token events", flush=True)

    if new_rows:
        new_arr = np.array(new_rows, dtype=np.uint32)
        # Concatenate and re-sort by (patient_idx, age_days)
        combined = np.vstack([data, new_arr])
    else:
        combined = data

    # Sort: primary by patient_idx (col 0), secondary by age_days (col 1)
    order = np.lexsort((combined[:, 1], combined[:, 0]))
    combined = combined[order]

    combined.tofile(str(dst))
    print(f"  Saved {len(combined):,} events → {dst.name}", flush=True)
    return len(combined)


def main():
    print("Phase C: Building v6.1 dataset\n", flush=True)

    print("Loading phenotype events...", flush=True)
    pheno_events = load_phenotype_events()

    print("Loading linkage map...", flush=True)
    linkage_df = pd.read_csv(LINKAGE)

    for split in ('train', 'val', 'test_input', 'test_future'):
        build_split(split, pheno_events, linkage_df)

    # ── Meta file ─────────────────────────────────────────────────────────────
    print("\nBuilding meta_v61.pkl ...", flush=True)
    with open(V5_DIR / 'meta.pkl', 'rb') as f:
        meta_v5 = pickle.load(f)

    meta_v61 = dict(meta_v5)
    meta_v61['vocab_size']         = NEW_VOCAB_SIZE_STORED   # 1569
    meta_v61['PHENOTYPE_TOKENS']   = {name: sid for name, sid, _ in PHENOTYPE_TOKENS}
    meta_v61['FIRST_PHENO_TOKEN']  = FIRST_STORED_ID          # 1536
    meta_v61['N_PHENO_TOKENS']     = len(PHENOTYPE_TOKENS)    # 33

    # Phenotype tokens are NOT in ignore_tokens (we want to predict them)
    # but do keep the existing ignore list
    with open(V61_DIR / 'meta_v61.pkl', 'wb') as f:
        pickle.dump(meta_v61, f)
    print(f"  vocab_size (stored) = {meta_v61['vocab_size']}", flush=True)

    # ── Copy auxiliary files ───────────────────────────────────────────────────
    for fname in ('patient_splits.csv', 'mimic_labels.csv', 'test_cutoffs.csv'):
        src = V5_DIR / fname
        if src.exists():
            shutil.copy(src, V61_DIR / fname)
    print("  Copied auxiliary CSVs", flush=True)

    print("\nPhase C complete.", flush=True)
    print(f"Dataset in: {V61_DIR}", flush=True)


if __name__ == '__main__':
    main()
