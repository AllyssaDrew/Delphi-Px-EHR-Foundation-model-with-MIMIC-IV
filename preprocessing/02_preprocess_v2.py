"""
Phase 2 (v2) - MIMIC-IV → Delphi binary format
Implements Option B+D:
  B: patient filtering (≥2 admissions with >30-day span OR 1 admission >7-day LOS)
  D: event-level granularity (intra-admission time points from transfers + procedures)

Split: 70 / 20 / 10  (train / val / test)
Test cutoff: each test patient's events are split at (last_admission - 180 days).
  test.bin       = pre-cutoff events (model input)
  test_future.bin = post-cutoff events (evaluation labels)
  test_cutoffs.csv = (patient_idx, cutoff_age_days, last_adm_age_days)

Token layout:
  0  Padding
  1  No event
  2  Female
  3  Male
  4  BMI_low   (BMI < q33)
  5  BMI_mid   (BMI q33–q67)
  6  BMI_high  (BMI > q67)
  7  ICU_admission
  8  ED_admission
  9+ ICD-10 3-char disease/procedure codes  (sorted, ≥MIN_CODE_COUNT)
  last  Death
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
OUT_DIR   = os.path.join(BASE, 'data/mimic_data_v2')
os.makedirs(OUT_DIR, exist_ok=True)

def data_path(name):
    return os.path.join(DATA, name, name)

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_CODE_COUNT      = 10     # min occurrences to keep a 3-char code
MIN_DISEASE_EVENTS  = 3      # min disease events per patient after filtering
TEST_CUTOFF_DAYS    = 180    # days before last admission withheld as test labels
TRAIN_FRAC          = 0.70
VAL_FRAC            = 0.20
# TEST_FRAC = 0.10 (remainder)
RANDOM_SEED         = 42

ICU_CAREUNIT_RE = r'Intensive Care|MICU|SICU|TSICU|CVICU|CCU|Neuro SICU'
ED_CAREUNIT     = 'Emergency Department'

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — GEMs ICD-9 → ICD-10 mapping
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Building GEMs ICD-9 → ICD-10 mapping ...")

gems_raw = pd.read_csv(GEMS_FILE, sep=r'\s+', header=None,
                       names=['icd9','icd10','flags'], dtype=str)
gems_raw['no_map']  = gems_raw['flags'].str[1].astype(int)
gems_raw['approx']  = gems_raw['flags'].str[0].astype(int)
gems_raw['combo']   = gems_raw['flags'].str[2].astype(int)
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

# Per-patient admission stats
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

# Keep only valid patients
patients   = patients[patients['subject_id'].isin(valid_pats_b)].copy()
admissions = admissions[admissions['subject_id'].isin(valid_pats_b)].copy()
diagnoses  = diagnoses[diagnoses['subject_id'].isin(valid_pats_b)].copy()
procedures = procedures[procedures['subject_id'].isin(valid_pats_b)].copy()
transfers  = transfers[transfers['subject_id'].isin(valid_pats_b)].copy()
omr        = omr[omr['subject_id'].isin(valid_pats_b)].copy()

# ── Helper: age_days from (year, doy, birth_year) ──────────────────────────
def age_days_from_year_doy(year_col, doy_col, birth_year_col):
    return ((year_col - birth_year_col) * 365.25 + doy_col).clip(lower=0).round().astype(np.int64)

# Attach birth_year to admissions
adm = admissions.merge(patients[['subject_id','birth_year']], on='subject_id')
adm['admit_year'] = adm['admittime'].dt.year
adm['admit_doy']  = adm['admittime'].dt.day_of_year
adm['age_days']   = age_days_from_year_doy(adm['admit_year'], adm['admit_doy'], adm['birth_year'])
adm_age = adm[['subject_id','hadm_id','age_days','admittime']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — ICD code events (diagnoses + procedures) with GEMs mapping
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Building ICD code events (Option D) ...")

def clean_icd9(c):
    return str(c).replace('.','').strip().upper()

# ── 4a: Diagnoses (ICD-10 native)
d10 = diagnoses[diagnoses['icd_version']==10].copy()
d10['code3'] = d10['icd_code'].str[:3].str.upper()
diag10 = d10.merge(adm_age[['subject_id','hadm_id','age_days']], on=['subject_id','hadm_id'], how='inner')
diag10 = diag10[['subject_id','age_days','code3']]

# ── 4b: Diagnoses (ICD-9 via GEMs) — vectorized via explode
d9 = diagnoses[diagnoses['icd_version']==9].copy()
d9['icd9_clean'] = d9['icd_code'].apply(clean_icd9)
d9['icd10_list'] = d9['icd9_clean'].map(icd9_to_icd10)
d9 = d9.dropna(subset=['icd10_list'])
d9 = d9.explode('icd10_list').rename(columns={'icd10_list':'code3'})
d9 = d9.merge(adm_age[['subject_id','hadm_id','age_days']], on=['subject_id','hadm_id'], how='inner')
d9 = d9[['subject_id','age_days','code3']]

# ── 4c: Procedures (ICD-10 with chartdate → precise intra-admission timing)
p10 = procedures[procedures['icd_version']==10].copy()
p10['code3'] = p10['icd_code'].str[:3].str.upper()
p10['chartdate'] = pd.to_datetime(p10['chartdate'])
p10['chart_year'] = p10['chartdate'].dt.year
p10['chart_doy']  = p10['chartdate'].dt.day_of_year
p10 = p10.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
p10['age_days'] = age_days_from_year_doy(p10['chart_year'], p10['chart_doy'], p10['birth_year'])
p10 = p10[['subject_id','age_days','code3']]

# ── 4d: Procedures (ICD-9 via GEMs)
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

# Combine disease events; deduplicate same (patient, day, code3)
all_disease = pd.concat([diag10, d9, p10, p9], ignore_index=True)
all_disease = all_disease.drop_duplicates(subset=['subject_id','age_days','code3'])
print(f"  Total (patient, day, code3) disease events: {len(all_disease):,}")

# ── Build vocabulary ──────────────────────────────────────────────────────────
code_counts = all_disease['code3'].value_counts()
codes_kept  = sorted(code_counts[code_counts >= MIN_CODE_COUNT].index.tolist())
print(f"  Codes with ≥{MIN_CODE_COUNT} occurrences: {len(codes_kept):,}")

# Token layout
# 0=Pad 1=NoEvent 2=Female 3=Male 4=BMI_low 5=BMI_mid 6=BMI_high
# 7=ICU_admission  8=ED_admission
RESERVED    = 9
DEATH_TOKEN = RESERVED + len(codes_kept)
vocab_size  = DEATH_TOKEN + 1
code2token  = {c: (i + RESERVED) for i, c in enumerate(codes_kept)}

# Filter events to kept codes and assign token_id
all_disease = all_disease[all_disease['code3'].isin(code2token)].copy()
all_disease['token_id'] = all_disease['code3'].map(code2token).astype(np.uint32)
all_disease = all_disease[['subject_id','age_days','token_id']]

print(f"  Vocab size: {vocab_size}  (reserved: {RESERVED}, disease: {len(codes_kept)}, death: 1)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — ICU and ED tokens from transfers (Option D intra-admission events)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Building ICU / ED clinical events from transfers ...")

transfers['intime'] = pd.to_datetime(transfers['intime'])
transfers = transfers.merge(patients[['subject_id','birth_year']], on='subject_id', how='inner')
transfers['in_year'] = transfers['intime'].dt.year
transfers['in_doy']  = transfers['intime'].dt.day_of_year
transfers['age_days'] = age_days_from_year_doy(
    transfers['in_year'], transfers['in_doy'], transfers['birth_year'])

# ICU admission events
icu_mask = transfers['careunit'].str.contains(ICU_CAREUNIT_RE, case=False, na=False)
icu_events = transfers[icu_mask][['subject_id','age_days']].copy()
icu_events['token_id'] = np.uint32(7)   # ICU_admission
icu_events = icu_events.drop_duplicates(subset=['subject_id','age_days'])
print(f"  ICU events: {len(icu_events):,}")

# ED admission events (first ED visit per hadm only to avoid double-counting)
ed_mask = transfers['careunit'].str.contains(ED_CAREUNIT, case=False, na=False)
ed_events = transfers[ed_mask][['subject_id','age_days']].copy()
ed_events['token_id'] = np.uint32(8)    # ED_admission
ed_events = ed_events.drop_duplicates(subset=['subject_id','age_days'])
print(f"  ED  events: {len(ed_events):,}")

clinical_events = pd.concat([icu_events, ed_events], ignore_index=True)
clinical_events = clinical_events[['subject_id','age_days','token_id']]

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — BMI, Sex, Death tokens
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: Building lifestyle / demographic / death tokens ...")

# BMI from OMR
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

# Sex tokens (once at age=0)
sex_events = patients[['subject_id','gender']].copy()
sex_events['token_id'] = sex_events['gender'].map({'F': 2, 'M': 3}).astype('Int64')
sex_events = sex_events.dropna(subset=['token_id'])
sex_events['token_id'] = sex_events['token_id'].astype(np.uint32)
sex_events['age_days'] = np.int64(0)
sex_events = sex_events[['subject_id','age_days','token_id']]
print(f"  Sex events: {len(sex_events):,}")

# Death tokens
dead = patients[patients['dod'].notna()].copy()
dead['dod'] = pd.to_datetime(dead['dod'])
dead['dod_year'] = dead['dod'].dt.year
dead['dod_doy']  = dead['dod'].dt.day_of_year
dead['age_days'] = age_days_from_year_doy(
    dead['dod_year'], dead['dod_doy'], dead['birth_year'])
death_events = dead[['subject_id','age_days']].copy()
death_events['token_id'] = np.uint32(DEATH_TOKEN)
print(f"  Death events: {len(death_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Combine, filter patients with too few disease events
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Combining all events and filtering patients ...")

combined = pd.concat([
    sex_events, bmi_events, clinical_events, all_disease, death_events,
], ignore_index=True)
combined['age_days']   = combined['age_days'].astype(np.int64)
combined['token_id']   = combined['token_id'].astype(np.uint32)
combined['subject_id'] = combined['subject_id'].astype(np.int64)

# Keep patients with ≥ MIN_DISEASE_EVENTS distinct disease events
disease_ct = all_disease.groupby('subject_id').size()
valid_pids  = disease_ct[disease_ct >= MIN_DISEASE_EVENTS].index
combined    = combined[combined['subject_id'].isin(valid_pids)]
print(f"  Patients with ≥{MIN_DISEASE_EVENTS} disease events: {len(valid_pids):,}")
print(f"  Total events after filter: {len(combined):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — 70 / 20 / 10 patient split
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 8: 70/20/10 patient split ...")

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
# Step 9 — Test set temporal cutoff (last 180 days withheld as labels)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: Applying 180-day temporal cutoff to test set ...")

# Compute last admission age for each test patient
adm_test = adm_age[adm_age['subject_id'].isin(test_pids)].copy()
last_adm  = adm_test.groupby('subject_id')['age_days'].max().rename('last_adm_age')
cutoff_map = (last_adm - TEST_CUTOFF_DAYS).rename('cutoff_age')
cutoff_df  = pd.concat([last_adm, cutoff_map], axis=1).reset_index()

# Split test events around cutoff
test_with_cut = test_df.merge(
    cutoff_df[['subject_id','cutoff_age']], on='subject_id', how='left')
test_with_cut['cutoff_age'] = test_with_cut['cutoff_age'].fillna(0)

test_input  = test_with_cut[test_with_cut['age_days'] <  test_with_cut['cutoff_age']].copy()
test_future = test_with_cut[test_with_cut['age_days'] >= test_with_cut['cutoff_age']].copy()

print(f"  Test patients: {len(test_pids):,}")
print(f"  Test input events (pre-cutoff):   {len(test_input):,}")
print(f"  Test future events (post-cutoff): {len(test_future):,}")

# Patients with at least some pre-cutoff history
pats_with_history = test_input['subject_id'].nunique()
print(f"  Test patients with pre-cutoff history: {pats_with_history:,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Build consecutive-ID uint32 arrays and save
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 10: Building binary arrays ...")

def build_bin_array(df, sort=True):
    df = df[['subject_id','age_days','token_id']].copy()
    if sort:
        df = df.sort_values(['subject_id','age_days']).reset_index(drop=True)
    uid, inv = np.unique(df['subject_id'].values, return_inverse=True)
    arr = np.stack([
        inv.astype(np.uint32),
        df['age_days'].clip(lower=0).values.astype(np.uint32),
        df['token_id'].values.astype(np.uint32),
    ], axis=1).astype(np.uint32)
    return arr, uid

train_arr, train_uids = build_bin_array(train_df)
val_arr,   val_uids   = build_bin_array(val_df)
test_in_arr,  test_in_uids  = build_bin_array(test_input)
test_fut_arr, test_fut_uids = build_bin_array(test_future)

for fname, arr in [('train.bin', train_arr), ('val.bin', val_arr),
                   ('test_input.bin', test_in_arr), ('test_future.bin', test_fut_arr)]:
    arr.tofile(os.path.join(OUT_DIR, fname))
    print(f"  {fname:<22}: {arr.shape[0]:>8,} events  |  "
          f"{arr[:,0].max()+1:>7,} patients  |  {arr.nbytes/1e6:5.1f} MB")

# Save cutoff metadata for test patients
# Re-attach the original (non-remapped) subject_id → sequential patient_idx mapping
test_in_pid_map = {pid: idx for idx, pid in enumerate(test_in_uids)}
cutoff_out = cutoff_df[cutoff_df['subject_id'].isin(test_pids)].copy()
cutoff_out['patient_idx'] = cutoff_out['subject_id'].map(test_in_pid_map)
cutoff_out.to_csv(os.path.join(OUT_DIR, 'test_cutoffs.csv'), index=False)
print(f"  test_cutoffs.csv saved ({len(cutoff_out)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 11 — labels.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 11: Writing labels.csv ...")

diag_dict = pd.read_csv(data_path('d_icd_diagnoses.csv'),
    usecols=['icd_code','icd_version','long_title'])
icd10_desc = (diag_dict[diag_dict['icd_version']==10]
              .assign(code3=lambda x: x['icd_code'].str[:3].str.upper())
              .drop_duplicates('code3').set_index('code3')['long_title'].to_dict())

label_rows = [
    'Padding', 'No event', 'Female', 'Male',
    f'BMI_low (BMI<{q33:.1f})', f'BMI_mid ({q33:.1f}-{q67:.1f})', f'BMI_high (BMI>{q67:.1f})',
    'ICU_admission', 'ED_admission',
]
for code in codes_kept:
    desc = icd10_desc.get(code, '')
    label_rows.append(f'{code} {desc}'.strip())
label_rows.append('Death')

assert len(label_rows) == vocab_size, f"labels {len(label_rows)} != vocab {vocab_size}"
pd.DataFrame({'event_name': label_rows}).to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)
print(f"  Saved: {vocab_size} labels")

# ══════════════════════════════════════════════════════════════════════════════
# Step 12 — meta.pkl
# ══════════════════════════════════════════════════════════════════════════════
meta = {
    'vocab_size':    vocab_size,
    'code2token':    code2token,
    'DEATH_TOKEN':   DEATH_TOKEN,
    'RESERVED':      RESERVED,
    'ignore_tokens': [0, 2, 3, 4, 5, 6, 7, 8],  # Padding + sex + BMI + ICU + ED
    'bmi_thresholds': (float(q33), float(q67)),
    'min_code_count': MIN_CODE_COUNT,
    'test_cutoff_days': TEST_CUTOFF_DAYS,
    'random_seed':   RANDOM_SEED,
    'split': {
        'train': len(train_pids), 'val': len(val_pids), 'test': len(test_pids)
    },
    'events': {
        'train': len(train_arr), 'val': len(val_arr),
        'test_input': len(test_in_arr), 'test_future': len(test_fut_arr),
    },
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# vocab stats
pd.DataFrame({
    'token_id':    [code2token[c] for c in codes_kept],
    'code3':       codes_kept,
    'count':       [int(code_counts.get(c,0)) for c in codes_kept],
    'description': [icd10_desc.get(c,'') for c in codes_kept],
}).to_csv(os.path.join(OUT_DIR, 'vocab_stats.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Phase 2 v2 COMPLETE — Summary")
print("="*60)
print(f"  Vocabulary size:   {vocab_size}")
print(f"  Disease tokens:    {len(codes_kept)}")
print(f"  Special tokens:    ICU(7), ED(8), BMI(4-6), Sex(2-3), Death({DEATH_TOKEN})")
print(f"  Train:  {len(train_pids):>7,} patients / {len(train_arr):>9,} events")
print(f"  Val:    {len(val_pids):>7,} patients / {len(val_arr):>9,} events")
print(f"  Test:   {len(test_pids):>7,} patients / input:{len(test_in_arr):>8,} future:{len(test_fut_arr):>7,}")
print(f"  Output: {OUT_DIR}")
print(f"\nNext → config/train_delphi_mimic_full.py  with  vocab_size={vocab_size}")
print(f"       ignore_tokens = {meta['ignore_tokens']}")
