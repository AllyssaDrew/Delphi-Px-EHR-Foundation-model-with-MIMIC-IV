"""
Phase A: ICD-9 one-to-one GEM mapping statistics.
Answers: is adding pre-2015 ICD-9 data (via one-to-one GEM) worth implementing in v5?
Gate: Neoplasm patients gain ≥ 2 additional tokens on average after filtering to one-to-one mappings.
"""

import os, sys, pickle
import pandas as pd
import numpy as np
from collections import defaultdict

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, '../Data')
GEM_FILE = os.path.join(BASE, 'reference/2018_I9gem.txt')
V4_META  = os.path.join(BASE, 'data/mimic_data_v4/meta.pkl')

def data_path(name):
    return os.path.join(DATA, name, name)

ICD10_CUTOFF    = pd.Timestamp('2015-10-01')
MIN_CODE_COUNT  = 25   # must match v4 preprocessing

print("=" * 60)
print("Phase A: ICD-9 one-to-one GEM mapping statistics")
print("=" * 60)

# ── Load GEM ──────────────────────────────────────────────────────────────────
print("\n1. Loading GEM file...")
gem_rows = []
with open(GEM_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            gem_rows.append((parts[0], parts[1], parts[2]))

gem_df = pd.DataFrame(gem_rows, columns=['icd9', 'icd10', 'flag'])
# combination flag: flag[0]=='1' means the mapping is a combination entry
# (requires two ICD-10 codes together to represent the ICD-9)
gem_df['is_combo'] = gem_df['flag'].str[0] == '1'
gem_df_nc = gem_df[~gem_df['is_combo']]

# One-to-one: ICD-9 code maps to exactly one ICD-10 code (non-combination)
icd10_per_icd9 = gem_df_nc.groupby('icd9')['icd10'].nunique()
one2one_icd9   = set(icd10_per_icd9[icd10_per_icd9 == 1].index)
# Build mapping
one2one_map    = (gem_df_nc[gem_df_nc['icd9'].isin(one2one_icd9)]
                  .drop_duplicates('icd9')
                  .set_index('icd9')['icd10'].to_dict())

print(f"   GEM total entries:          {len(gem_df):>8,}")
print(f"   Non-combination entries:    {len(gem_df_nc):>8,}")
print(f"   Unique ICD-9 codes (nc):    {len(icd10_per_icd9):>8,}")
print(f"   One-to-one ICD-9 codes:     {len(one2one_icd9):>8,} ({100*len(one2one_icd9)/len(icd10_per_icd9):.1f}%)")

# ── Load MIMIC tables ─────────────────────────────────────────────────────────
print("\n2. Loading MIMIC tables...")
patients   = pd.read_csv(data_path('patients.csv'),
    usecols=['subject_id', 'anchor_age', 'anchor_year'])
admissions = pd.read_csv(data_path('admissions.csv'),
    usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_days']  = ((admissions['dischtime'] - admissions['admittime'])
                           .dt.total_seconds() / 86400)

diagnoses  = pd.read_csv(data_path('diagnoses_icd.csv'),
    usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])

print(f"   patients: {len(patients):,}  admissions: {len(admissions):,}  diagnoses: {len(diagnoses):,}")

# ── Apply Option B patient filter (same as preprocess_v4) ────────────────────
print("\n3. Applying Option B patient filter...")
adm_s = admissions.groupby('subject_id').agg(
    n_adm    = ('hadm_id',   'count'),
    span_days= ('admittime', lambda x: (x.max() - x.min()).total_seconds() / 86400),
    max_los  = ('los_days',  'max'),
).reset_index()
crit_multi  = (adm_s['n_adm'] >= 2) & (adm_s['span_days'] > 30)
crit_single = (adm_s['n_adm'] == 1) & (adm_s['max_los'] > 7)
valid_pats  = set(adm_s[crit_multi | crit_single]['subject_id'])
print(f"   Valid patients: {len(valid_pats):,}")

# ── ICD-9 diagnoses for valid patients ────────────────────────────────────────
# MIMIC-IV dates are shifted to 2100s, so admittime < 2015-10-01 returns nothing.
# ICD version is recorded faithfully: icd_version=9 identifies pre-ICD10 records.
print("\n4. Extracting ICD-9 diagnoses (icd_version=9)...")
adm_icd9_hadm = set(diagnoses[diagnoses['icd_version'] == 9]['hadm_id'])
adm_icd9 = admissions[
    admissions['subject_id'].isin(valid_pats) &
    admissions['hadm_id'].isin(adm_icd9_hadm)
]
diag9 = diagnoses[
    (diagnoses['icd_version'] == 9) &
    (diagnoses['subject_id'].isin(valid_pats))
].copy()

diag9['icd9_3'] = diag9['icd_code'].str[:3]
print(f"   Admissions with ICD-9 coding: {adm_icd9['hadm_id'].nunique():,}")
print(f"   ICD-9 records (valid patients): {len(diag9):,}")
print(f"   Unique ICD-9 3-char codes: {diag9['icd9_3'].nunique():,}")

# ── Map to ICD-10 (3-char), one-to-one only ───────────────────────────────────
print("\n5. Applying one-to-one GEM filter...")

# Try full-code match first; fall back to 3-char prefix match
diag9['icd10_full'] = diag9['icd_code'].map(one2one_map)
one2one_3char = {k: v for k, v in one2one_map.items() if len(k) == 3}
no_full = diag9['icd10_full'].isna()
diag9.loc[no_full, 'icd10_full'] = diag9.loc[no_full, 'icd9_3'].map(one2one_3char)

diag9_mapped = diag9[diag9['icd10_full'].notna()].copy()
diag9_mapped['code3_icd10'] = diag9_mapped['icd10_full'].str[:3].str.upper()

n_total_icd9 = len(diag9)
n_mapped     = len(diag9_mapped)
print(f"   Records with one-to-one mapping: {n_mapped:,} / {n_total_icd9:,} ({100*n_mapped/n_total_icd9:.1f}%)")

# ── Filter by v4 vocabulary (only codes in v4 vocab count) ───────────────────
print("\n6. Checking v4 vocabulary coverage...")
with open(V4_META, 'rb') as f:
    v4meta = pickle.load(f)
v4_codes = set(v4meta['code2token'].keys())   # ICD-10 3-char codes in v4 vocab

diag9_in_vocab = diag9_mapped[diag9_mapped['code3_icd10'].isin(v4_codes)].copy()
print(f"   Mapped codes in v4 vocab: {diag9_in_vocab['code3_icd10'].nunique():,} unique codes")
print(f"   Records in v4 vocab: {len(diag9_in_vocab):,} ({100*len(diag9_in_vocab)/n_total_icd9:.1f}% of all ICD-9)")

# First-occurrence dedup: per patient, keep earliest occurrence of each code
diag9_in_vocab = diag9_in_vocab.merge(
    adm_icd9[['hadm_id', 'admittime']], on='hadm_id', how='inner')
diag9_in_vocab = diag9_in_vocab.sort_values('admittime')
diag9_fo = diag9_in_vocab.drop_duplicates(subset=['subject_id', 'code3_icd10'], keep='first')
print(f"   After first-occurrence dedup: {len(diag9_fo):,} (patient, code) pairs")

# ── Chapter-level analysis ─────────────────────────────────────────────────────
print("\n7. Chapter-level breakdown...")

def icd10_chapter(code3):
    c = str(code3).upper()
    if c[0] in ('A','B'): return 'I. Infectious Diseases'
    if c[0] == 'C' or (c[0]=='D' and c[1:3]<='48'): return 'II. Neoplasms'
    if c[0] == 'D': return 'III. Blood & Immune'
    if c[0] == 'E': return 'IV. Metabolic'
    if c[0] == 'F': return 'V. Mental'
    if c[0] == 'G': return 'VI. Nervous System'
    if c[0] == 'H' and c[1:3] <= '59': return 'VII. Eye'
    if c[0] == 'H': return 'VIII. Ear'
    if c[0] == 'I': return 'IX. Circulatory'
    if c[0] == 'J': return 'X. Respiratory'
    if c[0] == 'K': return 'XI. Digestive'
    if c[0] == 'L': return 'XII. Skin'
    if c[0] == 'M': return 'XIII. Musculoskeletal'
    if c[0] == 'N': return 'XIV. Genitourinary'
    if c[0] == 'O': return 'XV. Pregnancy'
    if c[0] == 'P': return 'XVI. Perinatal'
    if c[0] == 'Q': return 'XVII. Congenital'
    if c[0] == 'R': return 'XVIII. Symptoms'
    if c[0] in ('S','T'): return 'XIX. Injury'
    if c[0] in ('V','W','X','Y'): return 'XX. External'
    if c[0] == 'Z': return 'XXI. Health Factors'
    return 'Procedures'

diag9_fo['chapter'] = diag9_fo['code3_icd10'].apply(icd10_chapter)
chapter_stats = (diag9_fo.groupby('chapter')
                 .agg(n_records=('subject_id','count'),
                      n_patients=('subject_id','nunique'),
                      n_codes=('code3_icd10','nunique'))
                 .sort_values('n_records', ascending=False))

print(f"\n{'Chapter':<35} {'Records':>8} {'Patients':>9} {'Codes':>6}")
print("-" * 62)
for ch, row in chapter_stats.iterrows():
    print(f"{ch:<35} {row['n_records']:>8,} {row['n_patients']:>9,} {row['n_codes']:>6,}")

# ── Neoplasm gate check ────────────────────────────────────────────────────────
print("\n8. Neoplasm gate check (≥2 additional tokens per patient)...")

# Get patients currently in v4 TRAIN set (approximate: all valid patients)
# We check the average number of new tokens added per Neoplasm patient
neo_patients = diag9_fo[diag9_fo['chapter'] == 'II. Neoplasms']['subject_id'].unique()
neo_tokens_per_pat = (diag9_fo[diag9_fo['chapter'] == 'II. Neoplasms']
                      .groupby('subject_id').size())

print(f"   Neoplasm patients gaining ≥1 new token: {len(neo_patients):,}")
print(f"   Mean new tokens per Neoplasm patient:   {neo_tokens_per_pat.mean():.2f}")
print(f"   Median new tokens per Neoplasm patient: {neo_tokens_per_pat.median():.2f}")
print(f"   Patients gaining ≥2 new tokens:         {(neo_tokens_per_pat >= 2).sum():,} ({100*(neo_tokens_per_pat>=2).mean():.1f}%)")

gate_pass = neo_tokens_per_pat.mean() >= 2.0
print(f"\n   GATE: mean ≥ 2 tokens → {'PASS ✓' if gate_pass else 'FAIL ✗'}")
print(f"   {'→ Proceed with pre-2015 ICD-9 one-to-one mapping in Phase B.' if gate_pass else '→ Skip pre-2015 mapping; benefit too small.'}")

# ── Overall new events per patient ───────────────────────────────────────────
print("\n9. Overall new tokens per patient...")
all_tokens_per_pat = diag9_fo.groupby('subject_id').size()
print(f"   Patients gaining ≥1 new token: {len(all_tokens_per_pat):,} / {len(valid_pats):,} ({100*len(all_tokens_per_pat)/len(valid_pats):.1f}%)")
print(f"   Mean new tokens per patient (all): {all_tokens_per_pat.mean():.2f}")
print(f"   Median: {all_tokens_per_pat.median():.1f}  P90: {all_tokens_per_pat.quantile(0.9):.1f}  Max: {all_tokens_per_pat.max()}")

# ── ICD-9 code-level mapping summary ─────────────────────────────────────────
print("\n10. ICD-9 one-to-one mapping rate by ICD-9 chapter...")
def icd9_chapter(code3):
    try:
        c = int(code3)
        if 1 <= c <= 139: return 'I. Infectious'
        if 140 <= c <= 239: return 'II. Neoplasms'
        if 240 <= c <= 279: return 'III. Endocrine/Metabolic'
        if 280 <= c <= 289: return 'IV. Blood'
        if 290 <= c <= 319: return 'V. Mental'
        if 320 <= c <= 389: return 'VI. Nervous/Sense'
        if 390 <= c <= 459: return 'VII. Circulatory'
        if 460 <= c <= 519: return 'VIII. Respiratory'
        if 520 <= c <= 579: return 'IX. Digestive'
        if 580 <= c <= 629: return 'X. Genitourinary'
        if 630 <= c <= 679: return 'XI. Pregnancy'
        if 680 <= c <= 709: return 'XII. Skin'
        if 710 <= c <= 739: return 'XIII. Musculoskeletal'
        if 740 <= c <= 759: return 'XIV. Congenital'
        if 760 <= c <= 779: return 'XV. Perinatal'
        if 780 <= c <= 799: return 'XVI. Symptoms'
        if 800 <= c <= 999: return 'XVII. Injury'
    except ValueError:
        return 'Procedures (V/E codes)'
    return 'Other'

diag9['icd9_ch'] = diag9['icd9_3'].apply(icd9_chapter)
diag9['has_mapping'] = diag9['icd_code'].isin(one2one_icd9) | diag9['icd9_3'].isin(one2one_3char)
ch_map = (diag9.groupby('icd9_ch')
          .agg(total=('icd_code','count'), mapped=('has_mapping','sum'))
          .assign(pct=lambda x: 100*x['mapped']/x['total'])
          .sort_values('total', ascending=False))

print(f"\n{'ICD-9 Chapter':<30} {'Total':>8} {'Mapped':>8} {'Map%':>6}")
print("-" * 56)
for ch, row in ch_map.iterrows():
    print(f"{ch:<30} {row['total']:>8,} {row['mapped']:>8,} {row['pct']:>5.1f}%")

print("\n" + "=" * 60)
print("Phase A COMPLETE")
print("=" * 60)
