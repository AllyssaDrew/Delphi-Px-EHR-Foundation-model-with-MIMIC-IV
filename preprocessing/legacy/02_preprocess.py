"""
Phase 2 - MIMIC-IV → Delphi binary format
Strategy B: ICD-9 codes are mapped to ICD-10 via CMS 2018 GEMs crosswalk.

Outputs (written to data/mimic_data/):
  train.bin       - uint32 array (patient_idx, age_days, token_id) — training set
  val.bin         - uint32 array — validation set
  labels.csv      - token_id → event_name (row N+1 = token N)
  meta.pkl        - vocabulary metadata
  vocab_stats.csv - per-token counts for reference

Token layout:
  0  = Padding
  1  = No event  (inserted by DataLoader, not stored in .bin)
  2  = Female
  3  = Male
  4  = BMI_low
  5  = BMI_mid
  6  = BMI_high
  7+ = ICD-10 3-char disease codes (sorted)
  last = Death

ICD-9 → ICD-10 mapping strategy (GEMs):
  - Exact non-combination, non-approximate mappings are preferred.
  - If only approximate or combination entries exist, take the first entry
    where no_map flag is 0.
  - A single ICD-9 code that maps to multiple ICD-10 codes will produce
    multiple event rows (all retained) — this is intentional: a diagnosis
    chapter classification is still recovered even if the exact code is
    uncertain.
  - ICD-9 codes with no GEMs mapping (no_map=1 or absent) are dropped.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, '../Data')
GEMS_FILE = os.path.join(BASE, 'reference/2018_I9gem.txt')
OUT_DIR   = os.path.join(BASE, 'data/mimic_data')
os.makedirs(OUT_DIR, exist_ok=True)

def data_path(name):
    return os.path.join(DATA, name, name)

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_CODE_COUNT = 10          # drop codes appearing < this many times
MIN_ICD10_EVENTS = 3         # drop patients with fewer ICD-10 events after mapping
VAL_FRACTION   = 0.20        # fraction of patients reserved for validation
RANDOM_SEED    = 42

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build ICD-9 → ICD-10 mapping from GEMs
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Building GEMs ICD-9 → ICD-10 mapping ...")

gems_raw = pd.read_csv(GEMS_FILE, sep=r'\s+', header=None,
                       names=['icd9', 'icd10', 'flags'], dtype=str)

# Parse flag bits (5-digit string):
#  [0] approximate  [1] no_map  [2] combination  [3] scenario  [4] choice_list
gems_raw['approx']  = gems_raw['flags'].str[0].astype(int)
gems_raw['no_map']  = gems_raw['flags'].str[1].astype(int)
gems_raw['combo']   = gems_raw['flags'].str[2].astype(int)
gems_raw['scenario']= gems_raw['flags'].str[3].astype(int)
gems_raw['choice']  = gems_raw['flags'].str[4].astype(int)

# Drop rows where no ICD-10 equivalent exists
gems_valid = gems_raw[gems_raw['no_map'] == 0].copy()

# For preference scoring: exact non-combo non-approx = best
gems_valid['priority'] = (gems_valid['approx'] + gems_valid['combo']).astype(int)

# Build mapping: icd9 → list of (icd10_3char, priority)
# We keep ALL valid ICD-10 mappings per ICD-9 code (many-to-many retained as
# separate events, then de-duplicated at the 3-char level per admission).
gems_valid['icd10_3char'] = gems_valid['icd10'].str[:3].str.upper()
gems_valid['icd9_clean']  = gems_valid['icd9'].str.strip().str.upper()

# Keep the best (lowest priority) mapping per (icd9, icd10_3char) pair
gems_best = (gems_valid.sort_values('priority')
                       .drop_duplicates(subset=['icd9_clean', 'icd10_3char']))

# Build dict: icd9 → list of icd10_3char codes
icd9_to_icd10 = {}
for icd9, grp in gems_best.groupby('icd9_clean'):
    icd9_to_icd10[icd9] = grp['icd10_3char'].tolist()

n_icd9_mapped = len(icd9_to_icd10)
print(f"  GEMs: {n_icd9_mapped:,} unique ICD-9 codes have valid ICD-10 mapping(s)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load MIMIC tables
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Loading MIMIC-IV tables ...")

patients = pd.read_csv(data_path('patients.csv'),
                       usecols=['subject_id','gender','anchor_age','anchor_year','dod'])
admissions = pd.read_csv(data_path('admissions.csv'),
                         usecols=['subject_id','hadm_id','admittime'])
diagnoses  = pd.read_csv(data_path('diagnoses_icd.csv'),
                         usecols=['subject_id','hadm_id','icd_code','icd_version'])
procedures = pd.read_csv(data_path('procedures_icd.csv'),
                         usecols=['subject_id','hadm_id','chartdate','icd_code','icd_version'])
omr        = pd.read_csv(data_path('omr.csv'),
                         usecols=['subject_id','chartdate','result_name','result_value'])

print(f"  patients:   {len(patients):,}")
print(f"  admissions: {len(admissions):,}")
print(f"  diagnoses:  {len(diagnoses):,}")
print(f"  procedures: {len(procedures):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute patient age at each event (in days)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Computing patient ages ...")

# Birth year: anchor_year - anchor_age (integer year precision)
patients['birth_year'] = patients['anchor_year'] - patients['anchor_age']

# Parse admission dates
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['admit_year'] = admissions['admittime'].dt.year
admissions['admit_doy']  = admissions['admittime'].dt.day_of_year  # 1-365

adm = admissions.merge(patients[['subject_id','birth_year','gender','dod']], on='subject_id')

# Age at admission in fractional days
adm['age_days'] = (
    (adm['admit_year'] - adm['birth_year']) * 365.25
    + adm['admit_doy']
).clip(lower=0).round().astype(np.int64)

adm_age = adm[['subject_id','hadm_id','age_days']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Expand ICD-9 codes to ICD-10 via GEMs; keep ICD-10 native codes
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: ICD-9 → ICD-10 conversion (Strategy B — GEMs) ...")

def clean_icd9(code):
    return str(code).replace('.','').strip().upper()

# ── 4a: ICD-10 native diagnoses (keep as-is)
d10 = diagnoses[diagnoses['icd_version'] == 10].copy()
d10['code3'] = d10['icd_code'].str[:3].str.upper()
diag_events_10 = d10[['subject_id','hadm_id','code3']].copy()

# ── 4b: ICD-9 diagnoses → expand via GEMs
d9 = diagnoses[diagnoses['icd_version'] == 9].copy()
d9['icd9_clean'] = d9['icd_code'].apply(clean_icd9)

print(f"  ICD-10 diagnosis records: {len(diag_events_10):,}")
print(f"  ICD-9  diagnosis records: {len(d9):,}")

# Explode: each ICD-9 row → one row per mapped ICD-10 3-char code
rows_9 = []
unmapped_9 = 0
for _, row in tqdm(d9.iterrows(), total=len(d9), desc="  Mapping ICD-9→10", mininterval=5):
    mapped = icd9_to_icd10.get(row['icd9_clean'])
    if mapped is None:
        unmapped_9 += 1
        continue
    for c3 in mapped:
        rows_9.append({'subject_id': row['subject_id'],
                       'hadm_id':    row['hadm_id'],
                       'code3':      c3})

diag_events_9 = pd.DataFrame(rows_9)
print(f"  ICD-9 records unmapped (dropped): {unmapped_9:,} ({100*unmapped_9/len(d9):.1f}%)")
print(f"  ICD-9 records mapped to ICD-10:   {len(diag_events_9):,}")

# ── 4c: ICD-10 procedures (use as additional event tokens)
p10 = procedures[procedures['icd_version'] == 10].copy()
p10['code3'] = p10['icd_code'].str[:3].str.upper()
# Procedures have chartdate not hadm_id-based timing; we map via hadm → admittime
proc_events_10 = p10[['subject_id','hadm_id','code3']].copy()

# ── 4d: Combine all disease events
all_diag = pd.concat([diag_events_10, diag_events_9, proc_events_10], ignore_index=True)
# De-duplicate: same patient × admission × code3 (keeps first occurrence)
all_diag = all_diag.drop_duplicates(subset=['subject_id','hadm_id','code3'])
print(f"\n  Total unique (patient, admission, code3) events: {len(all_diag):,}")

# Attach ages
all_diag = all_diag.merge(adm_age, on=['subject_id','hadm_id'], how='inner')
print(f"  After age join (admissions with timing): {len(all_diag):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Build vocabulary
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Building vocabulary ...")

code_counts = all_diag['code3'].value_counts()
codes_kept  = sorted(code_counts[code_counts >= MIN_CODE_COUNT].index.tolist())

print(f"  Total unique 3-char ICD-10 codes: {code_counts.shape[0]:,}")
print(f"  Codes with ≥{MIN_CODE_COUNT} occurrences (kept): {len(codes_kept):,}")

# Token layout
# 0=Padding 1=No_event 2=Female 3=Male 4=BMI_low 5=BMI_mid 6=BMI_high
# 7 ... 7+len(codes_kept)-1 = disease codes
# 7+len(codes_kept) = Death
RESERVED    = 7
DEATH_TOKEN = RESERVED + len(codes_kept)
vocab_size  = DEATH_TOKEN + 1

code2token  = {c: (i + RESERVED) for i, c in enumerate(codes_kept)}

print(f"  Reserved tokens (0-{RESERVED-1}): Padding, No_event, Female, Male, BMI_low, BMI_mid, BMI_high")
print(f"  Disease tokens:  {RESERVED} – {DEATH_TOKEN-1}  ({len(codes_kept)} codes)")
print(f"  Death token:     {DEATH_TOKEN}")
print(f"  Vocabulary size: {vocab_size}")

# Filter out disease events with dropped codes
all_diag = all_diag[all_diag['code3'].isin(code2token)].copy()
all_diag['token_id'] = all_diag['code3'].map(code2token).astype(np.uint32)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — BMI lifestyle tokens from OMR
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: Extracting BMI lifestyle tokens ...")

bmi_raw = omr[omr['result_name'] == 'BMI (kg/m2)'].copy()
bmi_raw['bmi_val'] = pd.to_numeric(bmi_raw['result_value'], errors='coerce')
bmi_raw = bmi_raw.dropna(subset=['bmi_val'])
bmi_raw = bmi_raw[(bmi_raw['bmi_val'] >= 10) & (bmi_raw['bmi_val'] <= 80)]  # sanity
print(f"  BMI records: {len(bmi_raw):,}")

# Compute global tertile thresholds
q33, q67 = bmi_raw['bmi_val'].quantile([0.333, 0.667])
print(f"  BMI tertile cuts: <{q33:.1f} (low) / {q33:.1f}–{q67:.1f} (mid) / >{q67:.1f} (high)")

def bmi_token(val):
    if val < q33:   return 4  # BMI_low
    elif val < q67: return 5  # BMI_mid
    else:           return 6  # BMI_high

bmi_raw['token_id'] = bmi_raw['bmi_val'].apply(bmi_token).astype(np.uint32)

# Attach patient age (use chartdate → birth_year)
bmi_raw['chartdate'] = pd.to_datetime(bmi_raw['chartdate'])
bmi_raw['chart_year'] = bmi_raw['chartdate'].dt.year
bmi_raw['chart_doy']  = bmi_raw['chartdate'].dt.day_of_year
bmi_raw = bmi_raw.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
bmi_raw['age_days'] = (
    (bmi_raw['chart_year'] - bmi_raw['birth_year']) * 365.25
    + bmi_raw['chart_doy']
).clip(lower=0).round().astype(np.int64)

# Keep one BMI record per (subject_id, year, token_id) to avoid redundancy
bmi_raw['chart_year_'] = bmi_raw['chartdate'].dt.year
bmi_events = bmi_raw.drop_duplicates(subset=['subject_id','chart_year_','token_id'])
bmi_events  = bmi_events[['subject_id','age_days','token_id']].copy()
print(f"  De-duplicated BMI events retained: {len(bmi_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Sex tokens (emitted once at age = birth ~ day 0)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Creating sex tokens ...")

sex_events = patients[['subject_id','gender']].copy()
sex_events['token_id'] = sex_events['gender'].map({'F': 2, 'M': 3})
sex_events = sex_events.dropna(subset=['token_id'])
sex_events['token_id'] = sex_events['token_id'].astype(np.uint32)
sex_events['age_days'] = np.int64(0)
sex_events = sex_events[['subject_id','age_days','token_id']]
print(f"  Sex tokens: {len(sex_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Death tokens
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 8: Creating death tokens ...")

dead = patients[patients['dod'].notna()].copy()
dead['dod'] = pd.to_datetime(dead['dod'])
dead['dod_year'] = dead['dod'].dt.year
dead['dod_doy']  = dead['dod'].dt.day_of_year
dead['age_days'] = (
    (dead['dod_year'] - dead['birth_year']) * 365.25
    + dead['dod_doy']
).clip(lower=0).round().astype(np.int64)

death_events = dead[['subject_id','age_days']].copy()
death_events['token_id'] = np.uint32(DEATH_TOKEN)
print(f"  Death tokens: {len(death_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Combine all events; filter patients with too few events
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: Combining and filtering ...")

disease_slim = all_diag[['subject_id','age_days','token_id']].copy()

combined = pd.concat([
    sex_events,
    bmi_events,
    disease_slim,
    death_events,
], ignore_index=True)

combined['age_days']  = combined['age_days'].astype(np.int64)
combined['token_id']  = combined['token_id'].astype(np.uint32)
combined['subject_id']= combined['subject_id'].astype(np.int64)

# Filter: keep patients who have >= MIN_ICD10_EVENTS disease events
disease_ct = disease_slim.groupby('subject_id')['token_id'].count()
valid_pats  = disease_ct[disease_ct >= MIN_ICD10_EVENTS].index
combined    = combined[combined['subject_id'].isin(valid_pats)]
print(f"  Patients with ≥{MIN_ICD10_EVENTS} disease events: {len(valid_pats):,}")
print(f"  Total events after filter: {len(combined):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Train / validation split
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 10: Train / val split ...")

rng = np.random.default_rng(RANDOM_SEED)
all_pids = np.array(sorted(valid_pats))
n_val    = int(len(all_pids) * VAL_FRACTION)
val_pids = set(rng.choice(all_pids, size=n_val, replace=False).tolist())
train_pids = set(all_pids.tolist()) - val_pids

print(f"  Train patients: {len(train_pids):,}")
print(f"  Val   patients: {len(val_pids):,}")

train_df = combined[combined['subject_id'].isin(train_pids)].copy()
val_df   = combined[combined['subject_id'].isin(val_pids)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 11 — Build consecutive patient-indexed uint32 arrays and save
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 11: Building binary arrays ...")

def build_bin_array(df):
    """Sort by patient then by age; remap patient IDs to consecutive integers."""
    df = df.sort_values(['subject_id', 'age_days']).reset_index(drop=True)
    uid, inv = np.unique(df['subject_id'].values, return_inverse=True)
    patient_idx = inv.astype(np.uint32)
    age_days    = df['age_days'].clip(lower=0).values.astype(np.uint32)
    token_id    = df['token_id'].values.astype(np.uint32)
    arr = np.stack([patient_idx, age_days, token_id], axis=1).astype(np.uint32)
    return arr

train_arr = build_bin_array(train_df)
val_arr   = build_bin_array(val_df)

print(f"  train.bin shape: {train_arr.shape}  "
      f"patients: {train_arr[:,0].max()+1:,}")
print(f"  val.bin   shape: {val_arr.shape}  "
      f"patients: {val_arr[:,0].max()+1:,}")

train_arr.tofile(os.path.join(OUT_DIR, 'train.bin'))
val_arr.tofile(os.path.join(OUT_DIR, 'val.bin'))
print(f"  Saved: {OUT_DIR}/train.bin  ({train_arr.nbytes/1e6:.1f} MB)")
print(f"  Saved: {OUT_DIR}/val.bin    ({val_arr.nbytes/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 12 — labels.csv (row N+1 = token N)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 12: Writing labels.csv ...")

# Load ICD-10 descriptions for readable names
diag_dict = pd.read_csv(data_path('d_icd_diagnoses.csv'),
                        usecols=['icd_code','icd_version','long_title'])
icd10_desc = (diag_dict[diag_dict['icd_version']==10]
              .assign(code3=lambda x: x['icd_code'].str[:3].str.upper())
              .drop_duplicates('code3')
              .set_index('code3')['long_title']
              .to_dict())

label_rows = [
    'Padding',
    'No event',
    'Female',
    'Male',
    f'BMI_low (BMI<{q33:.1f})',
    f'BMI_mid (BMI {q33:.1f}-{q67:.1f})',
    f'BMI_high (BMI>{q67:.1f})',
]
for code in codes_kept:
    desc = icd10_desc.get(code, '')
    label_rows.append(f'{code} {desc}'.strip())
label_rows.append('Death')

assert len(label_rows) == vocab_size, \
    f"labels length {len(label_rows)} != vocab_size {vocab_size}"

labels_df = pd.DataFrame({'event_name': label_rows})
labels_df.to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)
print(f"  Saved: {OUT_DIR}/labels.csv  ({vocab_size} entries)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 13 — meta.pkl
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 13: Writing meta.pkl ...")

meta = {
    'vocab_size':     vocab_size,
    'code2token':     code2token,
    'DEATH_TOKEN':    DEATH_TOKEN,
    'RESERVED':       RESERVED,
    # Tokens to ignore in training loss (Padding + sex + BMI lifestyle)
    'ignore_tokens':  [0, 2, 3, 4, 5, 6],
    'bmi_thresholds': (float(q33), float(q67)),
    'min_code_count': MIN_CODE_COUNT,
    'random_seed':    RANDOM_SEED,
    'train_patients': len(train_pids),
    'val_patients':   len(val_pids),
    'total_events_train': len(train_arr),
    'total_events_val':   len(val_arr),
}

with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print(f"  Saved: {OUT_DIR}/meta.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# Step 14 — Vocabulary stats CSV (for reference)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 14: Writing vocab_stats.csv ...")

vocab_stats = pd.DataFrame({
    'token_id':   [code2token[c] for c in codes_kept],
    'code3':      codes_kept,
    'count':      [int(code_counts.get(c, 0)) for c in codes_kept],
    'description':[icd10_desc.get(c, '') for c in codes_kept],
})
vocab_stats.to_csv(os.path.join(OUT_DIR, 'vocab_stats.csv'), index=False)
print(f"  Saved: {OUT_DIR}/vocab_stats.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Phase 2 COMPLETE — Summary")
print("="*60)
print(f"  Vocabulary size:          {vocab_size}")
print(f"  Disease tokens:           {len(codes_kept)}")
print(f"  Train patients / events:  {len(train_pids):,} / {len(train_arr):,}")
print(f"  Val   patients / events:  {len(val_pids):,}  / {len(val_arr):,}")
print(f"  Output directory:         {OUT_DIR}")
print(f"\nNext step: update config/train_delphi_mimic.py with vocab_size={vocab_size}")
print("  ignore_tokens should be:", meta['ignore_tokens'])
