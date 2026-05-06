"""
Phase 2 (v3) - MIMIC-IV → Delphi binary format
Improvements over v2:
  2.3  First-occurrence deduplication: per patient, keep only the earliest
       occurrence of each ICD-10 3-char code.
  2.5  Code frequency threshold raised 10 → 25.
  2.1  Lab event tokens: 8 labs × 3 discretised levels = 24 new tokens
       appended after DEATH_TOKEN.

Token layout:
  0          Padding
  1          No event
  2          Female
  3          Male
  4–6        BMI (low / mid / high)
  7          ICU_admission
  8          ED_admission
  9..D-1     ICD-10 3-char codes  (sorted, ≥MIN_CODE_COUNT; first-occurrence only)
  D          Death  (DEATH_TOKEN)
  D+1..D+24  Lab tokens (8 labs × 3 levels)
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, '../Data')
GEMS_FILE = os.path.join(BASE, 'reference/2018_I9gem.txt')
OUT_DIR   = os.path.join(BASE, 'data/mimic_data_v3')
os.makedirs(OUT_DIR, exist_ok=True)

def data_path(name):
    return os.path.join(DATA, name, name)

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_CODE_COUNT     = 25      # raised from 10
MIN_DISEASE_EVENTS = 3
TEST_CUTOFF_DAYS   = 180
TRAIN_FRAC         = 0.70
VAL_FRAC           = 0.20
RANDOM_SEED        = 42

ICU_CAREUNIT_RE = r'Intensive Care|MICU|SICU|TSICU|CVICU|CCU|Neuro SICU'
ED_CAREUNIT     = 'Emergency Department'

# ── Lab configuration ──────────────────────────────────────────────────────────
# Corrected itemids (verified against d_labitems.csv):
#   LDL:      50905 (Calculated) primary, 50906 (Measured) fallback
#   Troponin: 51003 (Troponin T) — 51002 absent in this MIMIC extract
LAB_ITEMS = [
    # (itemid_list,  name,         (low_thresh, high_thresh),  unit)
    ([50852],        'HbA1c',      (5.7,  6.5),                '%'),
    ([50912],        'Creatinine', (0.9,  1.3),                'mg/dL'),
    ([51003],        'TroponinT',  (0.01, 0.10),               'ng/mL'),
    ([50905, 50906], 'LDL',        (100., 160.),               'mg/dL'),
    ([50963],        'NTproBNP',   (125., 900.),               'pg/mL'),
    ([51222],        'Hemoglobin', (10.,  12.),                'g/dL'),
    ([51265],        'Platelet',   (100., 400.),               'x10^9/L'),
    ([50861],        'ALT',        (40.,  120.),               'U/L'),
]
N_LABS   = len(LAB_ITEMS)
N_LEVELS = 3   # LOW / MID / HIGH per lab
N_LAB_TOKENS = N_LABS * N_LEVELS   # = 24

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — GEMs ICD-9 → ICD-10 mapping
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Building GEMs ICD-9 → ICD-10 mapping ...")

gems_raw = pd.read_csv(GEMS_FILE, sep=r'\s+', header=None,
                       names=['icd9','icd10','flags'], dtype=str)
gems_raw['no_map'] = gems_raw['flags'].str[1].astype(int)
gems_raw['approx'] = gems_raw['flags'].str[0].astype(int)
gems_raw['combo']  = gems_raw['flags'].str[2].astype(int)
gems_valid = gems_raw[gems_raw['no_map'] == 0].copy()
gems_valid['priority']   = gems_valid['approx'] + gems_valid['combo']
gems_valid['icd10_3char'] = gems_valid['icd10'].str[:3].str.upper()
gems_valid['icd9_clean']  = gems_valid['icd9'].str.strip().str.upper()
gems_best = (gems_valid.sort_values('priority')
                       .drop_duplicates(subset=['icd9_clean','icd10_3char']))
icd9_to_icd10 = {}
for icd9, grp in gems_best.groupby('icd9_clean'):
    icd9_to_icd10[icd9] = grp['icd10_3char'].tolist()
print(f"  {len(icd9_to_icd10):,} ICD-9 codes mapped")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load MIMIC tables
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Loading MIMIC-IV tables ...")

patients = pd.read_csv(data_path('patients.csv'),
    usecols=['subject_id','gender','anchor_age','anchor_year','dod'])
patients['birth_year'] = patients['anchor_year'] - patients['anchor_age']

admissions = pd.read_csv(data_path('admissions.csv'),
    usecols=['subject_id','hadm_id','admittime','dischtime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_days']  = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400

diagnoses  = pd.read_csv(data_path('diagnoses_icd.csv'),
    usecols=['subject_id','hadm_id','icd_code','icd_version'])
procedures = pd.read_csv(data_path('procedures_icd.csv'),
    usecols=['subject_id','hadm_id','chartdate','icd_code','icd_version'])
transfers  = pd.read_csv(data_path('transfers.csv'),
    usecols=['subject_id','hadm_id','eventtype','careunit','intime'])
omr = pd.read_csv(data_path('omr.csv'),
    usecols=['subject_id','chartdate','result_name','result_value'])

print(f"  patients:{len(patients):>9,}  admissions:{len(admissions):>8,}")
print(f"  diagnoses:{len(diagnoses):>8,}  procedures:{len(procedures):>8,}")
print(f"  transfers:{len(transfers):>8,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Patient filtering (Option B)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Applying Option B patient filter ...")

adm_stats = admissions.groupby('subject_id').agg(
    n_adm    = ('hadm_id',   'count'),
    first_adm= ('admittime', 'min'),
    last_adm = ('admittime', 'max'),
    max_los  = ('los_days',  'max'),
).reset_index()
adm_stats['span_days'] = (adm_stats['last_adm'] - adm_stats['first_adm']).dt.total_seconds() / 86400

crit_multi  = (adm_stats['n_adm'] >= 2) & (adm_stats['span_days'] > 30)
crit_single = (adm_stats['n_adm'] == 1) & (adm_stats['max_los'] > 7)
valid_pats_b = adm_stats[crit_multi | crit_single]['subject_id'].values

print(f"  Patients passing filter B: {len(valid_pats_b):,} / {len(admissions['subject_id'].unique()):,}")

patients   = patients[patients['subject_id'].isin(valid_pats_b)].copy()
admissions = admissions[admissions['subject_id'].isin(valid_pats_b)].copy()
diagnoses  = diagnoses[diagnoses['subject_id'].isin(valid_pats_b)].copy()
procedures = procedures[procedures['subject_id'].isin(valid_pats_b)].copy()
transfers  = transfers[transfers['subject_id'].isin(valid_pats_b)].copy()
omr        = omr[omr['subject_id'].isin(valid_pats_b)].copy()

def age_days_from_year_doy(year_col, doy_col, birth_year_col):
    return ((year_col - birth_year_col) * 365.25 + doy_col).clip(lower=0).round().astype(np.int64)

adm = admissions.merge(patients[['subject_id','birth_year']], on='subject_id')
adm['admit_year'] = adm['admittime'].dt.year
adm['admit_doy']  = adm['admittime'].dt.day_of_year
adm['age_days']   = age_days_from_year_doy(adm['admit_year'], adm['admit_doy'], adm['birth_year'])
adm_age = adm[['subject_id','hadm_id','age_days','admittime']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — ICD code events (diagnoses + procedures, with GEMs for ICD-9)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Building ICD code events ...")

def clean_icd9(c):
    return str(c).replace('.','').strip().upper()

d10 = diagnoses[diagnoses['icd_version']==10].copy()
d10['code3'] = d10['icd_code'].str[:3].str.upper()
diag10 = d10.merge(adm_age[['subject_id','hadm_id','age_days']], on=['subject_id','hadm_id'], how='inner')
diag10 = diag10[['subject_id','age_days','code3']]

d9 = diagnoses[diagnoses['icd_version']==9].copy()
d9['icd9_clean'] = d9['icd_code'].apply(clean_icd9)
d9['icd10_list'] = d9['icd9_clean'].map(icd9_to_icd10)
d9 = d9.dropna(subset=['icd10_list'])
d9 = d9.explode('icd10_list').rename(columns={'icd10_list':'code3'})
d9 = d9.merge(adm_age[['subject_id','hadm_id','age_days']], on=['subject_id','hadm_id'], how='inner')
d9 = d9[['subject_id','age_days','code3']]

p10 = procedures[procedures['icd_version']==10].copy()
p10['code3'] = p10['icd_code'].str[:3].str.upper()
p10['chartdate'] = pd.to_datetime(p10['chartdate'])
p10['chart_year'] = p10['chartdate'].dt.year
p10['chart_doy']  = p10['chartdate'].dt.day_of_year
p10 = p10.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
p10['age_days'] = age_days_from_year_doy(p10['chart_year'], p10['chart_doy'], p10['birth_year'])
p10 = p10[['subject_id','age_days','code3']]

p9 = procedures[procedures['icd_version']==9].copy()
p9['icd9_clean'] = p9['icd_code'].apply(clean_icd9)
p9['icd10_list'] = p9['icd9_clean'].map(icd9_to_icd10)
p9 = p9.dropna(subset=['icd10_list'])
p9 = p9.explode('icd10_list').rename(columns={'icd10_list':'code3'})
p9['chartdate'] = pd.to_datetime(p9['chartdate'])
p9['chart_year'] = p9['chartdate'].dt.year
p9['chart_doy']  = p9['chartdate'].dt.day_of_year
p9 = p9.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
p9['age_days'] = age_days_from_year_doy(p9['chart_year'], p9['chart_doy'], p9['birth_year'])
p9 = p9[['subject_id','age_days','code3']]

all_disease = pd.concat([diag10, d9, p10, p9], ignore_index=True)
# Deduplicate same (patient, day, code3) first
all_disease = all_disease.drop_duplicates(subset=['subject_id','age_days','code3'])
print(f"  (patient, day, code3) events before first-occurrence dedup: {len(all_disease):,}")

# ── 2.3 First-occurrence deduplication ──────────────────────────────────────
# For each (patient, code3), keep only the row with the earliest age_days.
all_disease = all_disease.sort_values(['subject_id','age_days'])
all_disease = all_disease.drop_duplicates(subset=['subject_id','code3'], keep='first')
print(f"  After first-occurrence dedup (one row per patient per code): {len(all_disease):,}")

# ── 2.5 Vocabulary: frequency threshold 25 ───────────────────────────────────
code_counts = all_disease['code3'].value_counts()
codes_kept  = sorted(code_counts[code_counts >= MIN_CODE_COUNT].index.tolist())
print(f"  Codes with ≥{MIN_CODE_COUNT} occurrences: {len(codes_kept):,}  "
      f"(was {(code_counts >= 10).sum()} at threshold 10)")

RESERVED    = 9
DEATH_TOKEN = RESERVED + len(codes_kept)
vocab_size_icd = DEATH_TOKEN + 1          # ICD vocab including Death
code2token  = {c: (i + RESERVED) for i, c in enumerate(codes_kept)}

all_disease = all_disease[all_disease['code3'].isin(code2token)].copy()
all_disease['token_id'] = all_disease['code3'].map(code2token).astype(np.uint32)
all_disease = all_disease[['subject_id','age_days','token_id']]
print(f"  Vocab (ICD only): {vocab_size_icd}  "
      f"(reserved:{RESERVED}, ICD:{len(codes_kept)}, death:1)")

# ── Lab token IDs follow immediately after DEATH_TOKEN ────────────────────────
LAB_TOKEN_START = DEATH_TOKEN + 1          # first lab token ID
vocab_size = LAB_TOKEN_START + N_LAB_TOKENS  # final vocab size

def lab_token_id(lab_idx, level):
    """lab_idx: 0..7, level: 0=LOW, 1=MID, 2=HIGH  → token ID"""
    return LAB_TOKEN_START + lab_idx * N_LEVELS + level

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — ICU and ED tokens from transfers
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Building ICU / ED events ...")

transfers['intime'] = pd.to_datetime(transfers['intime'])
transfers = transfers.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
transfers['in_year'] = transfers['intime'].dt.year
transfers['in_doy']  = transfers['intime'].dt.day_of_year
transfers['age_days'] = age_days_from_year_doy(
    transfers['in_year'], transfers['in_doy'], transfers['birth_year'])

icu_mask = transfers['careunit'].str.contains(ICU_CAREUNIT_RE, case=False, na=False)
icu_events = transfers[icu_mask][['subject_id','age_days']].copy()
icu_events['token_id'] = np.uint32(7)
icu_events = icu_events.drop_duplicates(subset=['subject_id','age_days'])

ed_mask = transfers['careunit'].str.contains(ED_CAREUNIT, case=False, na=False)
ed_events = transfers[ed_mask][['subject_id','age_days']].copy()
ed_events['token_id'] = np.uint32(8)
ed_events = ed_events.drop_duplicates(subset=['subject_id','age_days'])
print(f"  ICU: {len(icu_events):,}  ED: {len(ed_events):,}")

clinical_events = pd.concat([icu_events, ed_events], ignore_index=True)
clinical_events = clinical_events[['subject_id','age_days','token_id']]

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — BMI, Sex, Death tokens
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: Building demographic / death tokens ...")

bmi_raw = omr[omr['result_name'] == 'BMI (kg/m2)'].copy()
bmi_raw['bmi_val'] = pd.to_numeric(bmi_raw['result_value'], errors='coerce')
bmi_raw = bmi_raw.dropna(subset=['bmi_val'])
bmi_raw = bmi_raw[(bmi_raw['bmi_val'] >= 10) & (bmi_raw['bmi_val'] <= 80)]
q33, q67 = bmi_raw['bmi_val'].quantile([0.333, 0.667])

def bmi_token(v):
    return 4 if v < q33 else (5 if v < q67 else 6)

bmi_raw['token_id'] = bmi_raw['bmi_val'].apply(bmi_token).astype(np.uint32)
bmi_raw['chartdate'] = pd.to_datetime(bmi_raw['chartdate'])
bmi_raw['chart_year'] = bmi_raw['chartdate'].dt.year
bmi_raw['chart_doy']  = bmi_raw['chartdate'].dt.day_of_year
bmi_raw = bmi_raw.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
bmi_raw['age_days'] = age_days_from_year_doy(
    bmi_raw['chart_year'], bmi_raw['chart_doy'], bmi_raw['birth_year'])
bmi_raw['year_'] = bmi_raw['chartdate'].dt.year
bmi_events = bmi_raw.drop_duplicates(subset=['subject_id','year_','token_id'])
bmi_events = bmi_events[['subject_id','age_days','token_id']]
print(f"  BMI events: {len(bmi_events):,}  (q33={q33:.1f} q67={q67:.1f})")

sex_events = patients[['subject_id','gender']].copy()
sex_events['token_id'] = sex_events['gender'].map({'F': 2, 'M': 3}).astype('Int64')
sex_events = sex_events.dropna(subset=['token_id'])
sex_events['token_id'] = sex_events['token_id'].astype(np.uint32)
sex_events['age_days'] = np.int64(0)
sex_events = sex_events[['subject_id','age_days','token_id']]

dead = patients[patients['dod'].notna()].copy()
dead['dod'] = pd.to_datetime(dead['dod'])
dead['dod_year'] = dead['dod'].dt.year
dead['dod_doy']  = dead['dod'].dt.day_of_year
dead['age_days'] = age_days_from_year_doy(
    dead['dod_year'], dead['dod_doy'], dead['birth_year'])
death_events = dead[['subject_id','age_days']].copy()
death_events['token_id'] = np.uint32(DEATH_TOKEN)
print(f"  Sex events: {len(sex_events):,}  Death events: {len(death_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Lab event tokens (Phase B: 2.1)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Building lab event tokens ...")

valid_sids = set(valid_pats_b.tolist())

print("  Reading labevents.csv in chunks (18 GB) ...")
all_itemids = set()
for items, _, _, _ in LAB_ITEMS:
    all_itemids.update(items)

lab_chunks = []
for chunk in pd.read_csv(
        data_path('labevents.csv'),
        usecols=['subject_id','itemid','charttime','valuenum'],
        chunksize=1_000_000):
    sub = chunk[
        chunk['itemid'].isin(all_itemids) &
        chunk['subject_id'].isin(valid_sids) &
        chunk['valuenum'].notna()
    ]
    if len(sub):
        lab_chunks.append(sub)

labs_raw = pd.concat(lab_chunks, ignore_index=True)
labs_raw['charttime'] = pd.to_datetime(labs_raw['charttime'])
labs_raw = labs_raw.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
labs_raw['chart_year'] = labs_raw['charttime'].dt.year
labs_raw['chart_doy']  = labs_raw['charttime'].dt.day_of_year
labs_raw['age_days']   = age_days_from_year_doy(
    labs_raw['chart_year'], labs_raw['chart_doy'], labs_raw['birth_year'])

lab_event_list = []
for lab_idx, (itemids, name, (thresh_lo, thresh_hi), unit) in enumerate(LAB_ITEMS):
    sub = labs_raw[labs_raw['itemid'].isin(itemids)].copy()

    # First occurrence per patient for this lab
    sub = sub.sort_values('age_days')
    sub = sub.drop_duplicates(subset=['subject_id'], keep='first')

    def discretise(v):
        if v < thresh_lo:  return 0  # LOW
        if v < thresh_hi:  return 1  # MID
        return 2                     # HIGH

    sub['level']    = sub['valuenum'].apply(discretise)
    sub['token_id'] = sub.apply(
        lambda r: lab_token_id(lab_idx, int(r['level'])), axis=1
    ).astype(np.uint32)

    events = sub[['subject_id','age_days','token_id']]
    lab_event_list.append(events)
    pats_covered = len(sub)
    print(f"  {name:<12}  token {lab_token_id(lab_idx,0)}–{lab_token_id(lab_idx,2)}"
          f"  {pats_covered:>7,} patients")

lab_events = pd.concat(lab_event_list, ignore_index=True)
lab_events = lab_events[['subject_id','age_days','token_id']]
print(f"  Total lab events: {len(lab_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Combine, filter patients with too few disease events
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 8: Combining all events and filtering patients ...")

combined = pd.concat([
    sex_events, bmi_events, clinical_events, all_disease, death_events, lab_events,
], ignore_index=True)
combined['age_days']   = combined['age_days'].astype(np.int64)
combined['token_id']   = combined['token_id'].astype(np.uint32)
combined['subject_id'] = combined['subject_id'].astype(np.int64)

disease_ct = all_disease.groupby('subject_id').size()
valid_pids  = disease_ct[disease_ct >= MIN_DISEASE_EVENTS].index
combined    = combined[combined['subject_id'].isin(valid_pids)]
print(f"  Patients with ≥{MIN_DISEASE_EVENTS} unique disease codes: {len(valid_pids):,}")
print(f"  Total events after filter: {len(combined):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — 70 / 20 / 10 patient split
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: 70/20/10 patient split ...")

rng      = np.random.default_rng(RANDOM_SEED)
all_pids = np.array(sorted(valid_pids))
rng.shuffle(all_pids)

n_total = len(all_pids)
n_val   = int(n_total * VAL_FRAC)
n_test  = int(n_total * (1 - TRAIN_FRAC - VAL_FRAC))
n_train = n_total - n_val - n_test

train_pids = set(all_pids[:n_train].tolist())
val_pids   = set(all_pids[n_train:n_train + n_val].tolist())
test_pids  = set(all_pids[n_train + n_val:].tolist())

print(f"  Train: {len(train_pids):,}  Val: {len(val_pids):,}  Test: {len(test_pids):,}")

train_df = combined[combined['subject_id'].isin(train_pids)].copy()
val_df   = combined[combined['subject_id'].isin(val_pids)].copy()
test_df  = combined[combined['subject_id'].isin(test_pids)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Test set temporal cutoff
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 10: 180-day test cutoff ...")

adm_test  = adm_age[adm_age['subject_id'].isin(test_pids)].copy()
last_adm  = adm_test.groupby('subject_id')['age_days'].max().rename('last_adm_age')
cutoff_map = (last_adm - TEST_CUTOFF_DAYS).rename('cutoff_age')
cutoff_df  = pd.concat([last_adm, cutoff_map], axis=1).reset_index()

test_with_cut = test_df.merge(
    cutoff_df[['subject_id','cutoff_age']], on='subject_id', how='left')
test_with_cut['cutoff_age'] = test_with_cut['cutoff_age'].fillna(0)

test_input  = test_with_cut[test_with_cut['age_days'] <  test_with_cut['cutoff_age']].copy()
test_future = test_with_cut[test_with_cut['age_days'] >= test_with_cut['cutoff_age']].copy()

print(f"  Test input events:  {len(test_input):,}")
print(f"  Test future events: {len(test_future):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 11 — Build binary arrays and save
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 11: Building binary arrays ...")

def build_bin_array(df):
    df = df[['subject_id','age_days','token_id']].copy()
    df = df.sort_values(['subject_id','age_days']).reset_index(drop=True)
    uid, inv = np.unique(df['subject_id'].values, return_inverse=True)
    arr = np.stack([
        inv.astype(np.uint32),
        df['age_days'].clip(lower=0).values.astype(np.uint32),
        df['token_id'].values.astype(np.uint32),
    ], axis=1).astype(np.uint32)
    return arr, uid

train_arr,    train_uids    = build_bin_array(train_df)
val_arr,      val_uids      = build_bin_array(val_df)
test_in_arr,  test_in_uids  = build_bin_array(test_input)
test_fut_arr, test_fut_uids = build_bin_array(test_future)

for fname, arr in [('train.bin', train_arr), ('val.bin', val_arr),
                   ('test_input.bin', test_in_arr), ('test_future.bin', test_fut_arr)]:
    arr.tofile(os.path.join(OUT_DIR, fname))
    print(f"  {fname:<22}: {arr.shape[0]:>9,} events  "
          f"{arr[:,0].max()+1:>8,} patients  {arr.nbytes/1e6:.1f} MB")

test_in_pid_map = {pid: idx for idx, pid in enumerate(test_in_uids)}
cutoff_out = cutoff_df[cutoff_df['subject_id'].isin(test_pids)].copy()
cutoff_out['patient_idx'] = cutoff_out['subject_id'].map(test_in_pid_map)
cutoff_out.to_csv(os.path.join(OUT_DIR, 'test_cutoffs.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 12 — Labels CSV  (mimic_labels.csv format for evaluate_auc.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 12: Writing labels ...")

diag_dict = pd.read_csv(data_path('d_icd_diagnoses.csv'),
    usecols=['icd_code','icd_version','long_title'])
icd10_desc = (diag_dict[diag_dict['icd_version']==10]
              .assign(code3=lambda x: x['icd_code'].str[:3].str.upper())
              .drop_duplicates('code3').set_index('code3')['long_title'].to_dict())

label_names = [
    'Padding', 'No event', 'Female', 'Male',
    f'BMI_low (BMI<{q33:.1f})', f'BMI_mid ({q33:.1f}-{q67:.1f})',
    f'BMI_high (BMI>{q67:.1f})', 'ICU_admission', 'ED_admission',
]
for code in codes_kept:
    desc = icd10_desc.get(code, '')
    label_names.append(f'{code} {desc}'.strip())
label_names.append('Death')

# Lab token labels
LAB_LEVELS = ['LOW', 'MID', 'HIGH']
for lab_idx, (_, name, (lo, hi), unit) in enumerate(LAB_ITEMS):
    label_names.append(f'{name}_LOW  (<{lo} {unit})')
    label_names.append(f'{name}_MID  ({lo}–{hi} {unit})')
    label_names.append(f'{name}_HIGH (≥{hi} {unit})')

assert len(label_names) == vocab_size, \
    f"Label count {len(label_names)} != vocab_size {vocab_size}"

# Counts (for ICD codes and lab tokens)
label_counts = [None] * vocab_size
for code, tok in code2token.items():
    label_counts[tok] = int(code_counts.get(code, 0))

# Build the full labels dataframe
# For evaluate_auc.py compatibility we need: index, name, count, ICD-10 Chapter, ICD-10 Chapter (short), color
# Load the delphi chapter/color mapping from the Delphi repo
delphi_label_file = os.path.join(BASE, '../Delphi/Delphi-main/delphi_labels_chapters_colours_icd.csv')
if os.path.exists(delphi_label_file):
    delphi_ch = pd.read_csv(delphi_label_file)
    # Build code3 → chapter mapping from the Delphi reference
    # (The Delphi file has 'icd10' column with 3-char codes and chapter info)
    # Columns vary by version; attempt common column names
    possible_code_cols = ['icd10', 'code', 'ICD-10', 'icd_code']
    code_col = next((c for c in possible_code_cols if c in delphi_ch.columns), None)
    if code_col:
        code_to_chapter = delphi_ch.set_index(code_col)[['ICD-10 Chapter','ICD-10 Chapter (short)','color']].to_dict('index')
    else:
        code_to_chapter = {}
else:
    code_to_chapter = {}

# Fallback chapter assignment via ICD-10 prefix rules
def icd10_chapter(code3):
    c = str(code3).upper()
    if c[0] == 'A' or c[0] == 'B': return ('I. Infectious Diseases',        '#1f77b4')
    if c[0] == 'C' or (c[0]=='D' and c[1:3]<='48'): return ('II. Neoplasms', '#ff7f0e')
    if c[0] == 'D': return ('III. Blood & Immune Disorders',       '#2ca02c')
    if c[0] == 'E': return ('IV. Metabolic Diseases',              '#d62728')
    if c[0] == 'F': return ('V. Mental Disorders',                 '#9467bd')
    if c[0] == 'G': return ('VI. Nervous System Diseases',         '#8c564b')
    if c[0] == 'H' and c[1:3] <= '59': return ('VII. Eye Diseases','#e377c2')
    if c[0] == 'H': return ('VIII. Ear Diseases',                  '#7f7f7f')
    if c[0] == 'I': return ('IX. Circulatory Diseases',            '#bcbd22')
    if c[0] == 'J': return ('X. Respiratory Diseases',             '#17becf')
    if c[0] == 'K': return ('XI. Digestive Diseases',              '#aec7e8')
    if c[0] == 'L': return ('XII. Skin Diseases',                  '#ffbb78')
    if c[0] == 'M': return ('XIII. Musculoskeletal Diseases',      '#98df8a')
    if c[0] == 'N': return ('XIV. Genitourinary Diseases',         '#ff9896')
    if c[0] == 'O': return ('XV. Pregnancy & Childbirth',          '#c5b0d5')
    if c[0] == 'P': return ('XVI. Perinatal Conditions',           '#c49c94')
    if c[0] == 'Q': return ('XVII. Congenital Abnormalities',      '#f7b6d2')
    if c[0] == 'R': return ('XVIII. Symptoms & Signs',             '#dbdb8d')
    if c[0] == 'S' or c[0] == 'T': return ('XIX. Injury & Poisoning', '#9edae5')
    if c[0] in ('V','W','X','Y'): return ('XX. External Causes',   '#ad494a')
    if c[0] == 'Z': return ('XXI. Health Factors',                 '#8c6d31')
    return ('Procedures (ICD-10-PCS)', '#aaaaaa')

rows = []
for idx, name in enumerate(label_names):
    row = {'index': idx, 'name': name, 'count': label_counts[idx]}
    code = None
    if RESERVED <= idx < DEATH_TOKEN:
        code = codes_kept[idx - RESERVED]

    if code and code in code_to_chapter:
        info = code_to_chapter[code]
        row['ICD-10 Chapter']        = info.get('ICD-10 Chapter', '')
        row['ICD-10 Chapter (short)'] = info.get('ICD-10 Chapter (short)', '')
        row['color']                  = info.get('color', '#aaaaaa')
    elif code:
        ch, col = icd10_chapter(code)
        row['ICD-10 Chapter']        = ch
        row['ICD-10 Chapter (short)'] = ch
        row['color']                  = col
    elif idx >= LAB_TOKEN_START:
        row['ICD-10 Chapter']        = 'Lab Values'
        row['ICD-10 Chapter (short)'] = 'Lab Values'
        row['color']                  = '#636efa'
    else:
        row['ICD-10 Chapter']        = 'Technical'
        row['ICD-10 Chapter (short)'] = 'Technical'
        row['color']                  = '#2a52be'

    rows.append(row)

labels_df = pd.DataFrame(rows)
labels_df.to_csv(os.path.join(OUT_DIR, 'mimic_labels.csv'), index=False)
print(f"  mimic_labels.csv: {len(labels_df)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# Step 13 — meta.pkl
# ══════════════════════════════════════════════════════════════════════════════
# ignore_tokens: all non-clinical tokens the loss should skip
# Shifted IDs (after get_batch +1): 0=mask, 2=NoEvent, 3=F, 4=M, 5-7=BMI
# Lab tokens are CLINICAL — do NOT ignore them.
# ICU(8) and ED(9) are also real events — do NOT ignore.
ignore_tokens_shifted = [0, 2, 3, 4, 5, 6, 7]

lab_vocab = {}
for lab_idx, (itemids, name, (lo, hi), unit) in enumerate(LAB_ITEMS):
    for lvl, lvl_name in enumerate(['LOW','MID','HIGH']):
        tok = lab_token_id(lab_idx, lvl)
        lab_vocab[f'{name}_{lvl_name}'] = {
            'token_id': tok,
            'itemids':  itemids,
            'range':    (lo, hi),
            'unit':     unit,
            'level':    lvl,
        }

meta = {
    'vocab_size':     vocab_size,
    'code2token':     code2token,
    'DEATH_TOKEN':    DEATH_TOKEN,
    'RESERVED':       RESERVED,
    'LAB_TOKEN_START': LAB_TOKEN_START,
    'N_LAB_TOKENS':   N_LAB_TOKENS,
    'lab_vocab':      lab_vocab,
    'ignore_tokens':  ignore_tokens_shifted,
    'bmi_thresholds': (float(q33), float(q67)),
    'min_code_count': MIN_CODE_COUNT,
    'test_cutoff_days': TEST_CUTOFF_DAYS,
    'random_seed':    RANDOM_SEED,
    'split': {
        'train': len(train_pids), 'val': len(val_pids), 'test': len(test_pids)
    },
    'events': {
        'train': len(train_arr), 'val': len(val_arr),
        'test_input': len(test_in_arr), 'test_future': len(test_fut_arr),
    },
    'improvements': ['first_occurrence_dedup', 'freq_threshold_25', 'lab_tokens'],
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

pd.DataFrame({
    'token_id':    [code2token[c] for c in codes_kept],
    'code3':       codes_kept,
    'count':       [int(code_counts.get(c, 0)) for c in codes_kept],
    'description': [icd10_desc.get(c, '') for c in codes_kept],
}).to_csv(os.path.join(OUT_DIR, 'vocab_stats.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("Phase 2 v3 COMPLETE — Summary")
print("="*62)
print(f"  Vocab size:           {vocab_size}")
print(f"    ICD codes:          {len(codes_kept)}"
      f"  (freq≥{MIN_CODE_COUNT}, first-occurrence dedup)")
print(f"    Death token:        1  (id={DEATH_TOKEN})")
print(f"    Lab tokens:         {N_LAB_TOKENS}"
      f"  (ids {LAB_TOKEN_START}–{vocab_size-1})")
print(f"  Train: {len(train_pids):>7,} patients / {len(train_arr):>9,} events")
print(f"  Val:   {len(val_pids):>7,} patients / {len(val_arr):>9,} events")
print(f"  Test:  {len(test_pids):>7,} patients "
      f"/ input:{len(test_in_arr):>9,} future:{len(test_fut_arr):>8,}")
print(f"  Output: {OUT_DIR}")
print()
print(f"  Next → config/train_delphi_mimic_v3.py")
print(f"    vocab_size   = {vocab_size}   # raw; model needs +1 = {vocab_size+1}")
print(f"    ignore_tokens= {ignore_tokens_shifted}")
print(f"    block_size   = 64  (upgrade to 128 in Phase C)")
