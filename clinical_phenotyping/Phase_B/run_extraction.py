"""
Phase B: Full phenotype extraction over all 254k discharge notes.

Reads  : mimic_pipeline/multimodal_notes/Phase_A/merged_texts.csv
Writes : mimic_pipeline/clinical_phenotyping/Phase_B/phenotype_tokens.csv
         columns: subject_id, hadm_id, tokens   (tokens = space-separated stored IDs)

Uses Python multiprocessing; defaults to (cpu_count - 2) workers.
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


import sys
import time
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Add Phase_A to path so workers can import phenotype_dict / extract_phenotypes
PHASE_A = Path(__file__).parent.parent / 'Phase_A'
sys.path.insert(0, str(PHASE_A))

from extract_phenotypes import extract_phenotype_tokens


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of rows. Returns DataFrame with tokens column."""
    rows = []
    for _, row in chunk_df.iterrows():
        text = str(row['text']) if pd.notna(row['text']) else ''
        token_ids = extract_phenotype_tokens(text)
        rows.append({
            'subject_id': int(row['subject_id']),
            'hadm_id':    int(row['hadm_id']),
            'tokens':     ' '.join(str(t) for t in sorted(token_ids)),
        })
    return pd.DataFrame(rows)


def process_chunk_wrapper(args):
    chunk_df, chunk_idx, total = args
    t0 = time.time()
    result = process_chunk(chunk_df)
    n_with = (result['tokens'] != '').sum()
    elapsed = time.time() - t0
    print(f"  chunk {chunk_idx+1:4d}/{total}  "
          f"{len(chunk_df):5d} notes  "
          f"{n_with:4d} with tokens  "
          f"{elapsed:.1f}s", flush=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    default=None, help='Path to merged_texts.csv')
    parser.add_argument('--output',   default=None, help='Path for phenotype_tokens.csv')
    parser.add_argument('--workers',  type=int, default=0,
                        help='Number of worker processes (0=auto)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Notes per chunk (default 500)')
    args = parser.parse_args()

    PIPE = PIPELINE_DIR
    input_path  = args.input  or str(PIPE / 'multimodal_notes/Phase_A/merged_texts.csv')
    output_path = args.output or str(PIPE / 'clinical_phenotyping/Phase_B/phenotype_tokens.csv')
    n_workers   = args.workers or max(1, cpu_count() - 2)

    print(f"Reading {input_path} ...", flush=True)
    t_read = time.time()
    df = pd.read_csv(input_path, dtype={'subject_id': int, 'hadm_id': int, 'text': str})
    print(f"  {len(df):,} notes loaded in {time.time()-t_read:.1f}s", flush=True)

    # Split into chunks
    chunks = [df.iloc[i:i+args.chunk_size] for i in range(0, len(df), args.chunk_size)]
    total  = len(chunks)
    work   = [(c, i, total) for i, c in enumerate(chunks)]

    print(f"Processing {len(df):,} notes in {total} chunks "
          f"using {n_workers} workers ...", flush=True)
    t0 = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(process_chunk_wrapper, work)

    df_out = pd.concat(results, ignore_index=True)

    # Statistics
    n_with = (df_out['tokens'] != '').sum()
    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)
    print(f"  Notes with ≥1 token: {n_with:,} / {len(df_out):,} "
          f"({100*n_with/len(df_out):.1f}%)", flush=True)

    # Per-token counts
    from phenotype_dict import TOKEN_ID_TO_NAME
    all_ids = []
    for tok_str in df_out['tokens']:
        if tok_str:
            all_ids.extend(int(x) for x in tok_str.split())
    from collections import Counter
    counts = Counter(all_ids)
    print("\nPer-token counts:", flush=True)
    for tid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {TOKEN_ID_TO_NAME.get(tid, tid):<30} {cnt:>8,}", flush=True)

    df_out.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}", flush=True)


if __name__ == '__main__':
    main()
