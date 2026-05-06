"""
Phase 4 — Build MIMIC-compatible delphi_labels CSV
Creates the labels file that evaluate_auc.py expects:
  columns: index, name, count, ICD-10 Chapter, ICD-10 Chapter (short), color

Run once per preprocessed dataset (v1 or v2). Outputs to the data directory.
Also symlinks the file to the Delphi working directory for easy use.

Usage:
  python3 04_make_mimic_labels.py --data_dir data/mimic_data
  python3 04_make_mimic_labels.py --data_dir data/mimic_data_v2
"""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd

BASE   = os.path.dirname(os.path.abspath(__file__))
DELPHI = os.path.join(BASE, '../Delphi/Delphi-main')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/mimic_data_v2')
args = parser.parse_args()

DATA_DIR = os.path.join(BASE, args.data_dir)

# ── ICD-10 chapter mapping ──────────────────────────────────────────────────
ICD10_CHAPTERS = [
    ('A', 'B',  'I. Infectious Diseases',           '#1f77b4'),
    ('C', 'D4', 'II. Neoplasms',                    '#ff7f0e'),
    ('D5','D8', 'III. Blood & Immune Disorders',    '#2ca02c'),
    ('E', 'E',  'IV. Metabolic Diseases',           '#d62728'),
    ('F', 'F',  'V. Mental Disorders',              '#9467bd'),
    ('G', 'G',  'VI. Nervous System Diseases',      '#8c564b'),
    ('H0','H5', 'VII. Eye Diseases',                '#e377c2'),
    ('H6','H9', 'VIII. Ear Diseases',               '#7f7f7f'),
    ('I', 'I',  'IX. Circulatory Diseases',         '#bcbd22'),
    ('J', 'J',  'X. Respiratory Diseases',          '#17becf'),
    ('K', 'K',  'XI. Digestive Diseases',           '#aec7e8'),
    ('L', 'L',  'XII. Skin Diseases',               '#ffbb78'),
    ('M', 'M',  'XIII. Musculoskeletal Diseases',   '#98df8a'),
    ('N', 'N',  'XIV. Genitourinary Diseases',      '#ff9896'),
    ('O', 'O',  'XV. Pregnancy & Childbirth',       '#e377c2'),
    ('P', 'P',  'XVI. Perinatal Conditions',        '#c5b0d5'),
    ('Q', 'Q',  'XVII. Congenital Abnormalities',   '#c49c94'),
    ('R', 'R',  'XVIII. Symptoms & Signs',          '#f7b6d2'),
    ('S', 'T',  'XIX. Injury & Poisoning',          '#dbdb8d'),
    ('V', 'Y',  'XX. External Causes',              '#9edae5'),
    ('Z', 'Z',  'XXI. Factors Affecting Health',    '#393b79'),
]

# Build prefix-to-chapter mapping (using first char or first two chars)
def get_icd10_chapter(code3):
    """Return (long_chapter, short_chapter, color) for a 3-char ICD-10-CM or ICD-10-PCS code."""
    code3 = str(code3).upper().strip()
    if not code3:
        return ('Unknown', 'Unknown', '#cccccc')
    first   = code3[0]
    prefix2 = code3[:2]

    # ICD-10-PCS procedure codes start with a digit (0-9)
    if first.isdigit():
        return ('Procedures (ICD-10-PCS)', 'Procedures', '#aaaaaa')

    for lo, hi, chapter, color in ICD10_CHAPTERS:
        if len(lo) == 1 and len(hi) == 1:
            if lo <= first <= hi:
                return (chapter, chapter, color)
        elif len(lo) == 1 and len(hi) == 2:
            if first == lo:
                return (chapter, chapter, color)
            if first == hi[0] and prefix2 <= hi:
                return (chapter, chapter, color)
        elif len(lo) == 2 and len(hi) == 2:
            if first == lo[0] == hi[0] and lo <= prefix2 <= hi:
                return (chapter, chapter, color)

    return ('Unknown', 'Unknown', '#cccccc')


# ── Load meta and labels ───────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size  = meta['vocab_size']
RESERVED    = meta['RESERVED']
DEATH_TOKEN = meta['DEATH_TOKEN']
code2token  = meta['code2token']
token2code  = {v: k for k, v in code2token.items()}
vocab_stats = pd.read_csv(os.path.join(DATA_DIR, 'vocab_stats.csv'))
count_map   = vocab_stats.set_index('token_id')['count'].to_dict()

labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))

# ── Build output dataframe ─────────────────────────────────────────────────
rows = []
for idx, row in labels_df.iterrows():
    name = row['event_name']
    token_id = idx   # row index = token_id (labels.csv is indexed by token)

    if token_id < RESERVED:
        # Special tokens (Padding, No_event, sex, lifestyle, ICU, ED, ...)
        chapter_long  = 'Technical'
        chapter_short = 'Technical'
        color = '#2a52be'
        count = None
    elif token_id == DEATH_TOKEN:
        chapter_long  = 'Death'
        chapter_short = 'Death'
        color = '#000a35'
        count = int(count_map.get(token_id, 0)) if token_id in count_map else None
    else:
        code3 = token2code.get(token_id, '')
        chapter_long, chapter_short, color = get_icd10_chapter(code3)
        count = int(count_map.get(token_id, 0)) if token_id in count_map else None

    rows.append({
        'index': token_id,
        'name':  name,
        'count': count,
        'ICD-10 Chapter': chapter_long,
        'ICD-10 Chapter (short)': chapter_short,
        'color': color,
    })

out_df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'mimic_labels.csv')
out_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(out_df)} rows)")

# Chapter distribution summary
print("\nChapter distribution:")
print(out_df[out_df['index'] >= RESERVED]['ICD-10 Chapter (short)'].value_counts().to_string())

# Symlink to Delphi directory so evaluate_auc.py can find it
link_path = os.path.join(DELPHI, 'mimic_labels.csv')
if os.path.islink(link_path):
    os.remove(link_path)
os.symlink(out_path, link_path)
print(f"\nSymlinked to: {link_path}")
print("\nPhase 4 complete.")
print(f"To use with evaluate_auc.py, pass --labels_path={out_path}")
print("Or load directly in evaluate_delphi.ipynb")
