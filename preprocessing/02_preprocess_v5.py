"""
Phase 2 (v5) - MIMIC-IV → Delphi binary format
Changes from v4:
  5.1  eGFR (4-level, CKD-EPI 2021) replaces Creatinine (3-level)
       Creatinine token IDs 1472–1474 reused as eGFR_HIGH/MID/LOW_MOD;
       one new token appended as eGFR_VERY_LOW (stored ID = DEATH+25).
  5.2  Troponin I (itemid 51002) replaces Troponin T (itemid 51003).
  5.3  Lab tokens added to ignore_tokens (not predicted by CE loss).
  5.4  Pre-2015 ICD-9 one-to-one GEM mappings retained globally.
       Only ICD-9 codes with a unique, non-combination ICD-10 mapping
       (from 2018_I9gem.txt) are kept; all others discarded.

Token layout (stored space → model space after +1 shift):
  Same as v4, except:
  - eGFR replaces Creatinine at same 3 token slots + 1 new slot at end
  - N_LAB_TOKENS = 25 (was 24)
  - vocab_size stored = v4_vocab + 1 (one extra eGFR level)
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, '../Data')
OUT_DIR = os.path.join(BASE, 'data/mimic_data_v5')
GEM_FILE= os.path.join(BASE, 'reference/2018_I9gem.txt')
os.makedirs(OUT_DIR, exist_ok=True)

def data_path(name):
    return os.path.join(DATA, name, name)

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_CODE_COUNT     = 25
MIN_DISEASE_EVENTS = 3
TEST_CUTOFF_DAYS   = 180
TRAIN_FRAC         = 0.70
VAL_FRAC           = 0.20
RANDOM_SEED        = 42

# ICD10_CUTOFF = '2015-10-01' not used: MIMIC-IV dates are shifted to 2100s.
# ICD version is identified via icd_version column (9 = ICD-9, 10 = ICD-10).

ICU_CAREUNIT_RE = r'Intensive Care|MICU|SICU|TSICU|CVICU|CCU|Neuro SICU'
ED_CAREUNIT     = 'Emergency Department'

# ── Lab configuration (v5) ─────────────────────────────────────────────────────
# eGFR uses CKD-EPI 2021 formula; thresholds are eGFR mL/min/1.73m^2
# Troponin I replaces Troponin T
# eGFR has 4 levels: HIGH (≥90), MID (60-89), LOW_MOD (30-59), VERY_LOW (<30)
LAB_ITEMS = [
    ([50852],        'HbA1c',      (5.7,  6.5),        '%',        3),
    ([50912],        'eGFR',       (60.,  90.),         'mL/min',   4),  # 4 levels
    ([51002],        'TroponinI',  (0.04, 0.10),        'ng/mL',    3),  # was 51003 TroponinT
    ([50905, 50906], 'LDL',        (100., 160.),        'mg/dL',    3),
    ([50963],        'NTproBNP',   (125., 900.),        'pg/mL',    3),
    ([51222],        'Hemoglobin', (10.,  12.),         'g/dL',     3),
    ([51265],        'Platelet',   (100., 400.),        'x10^9/L',  3),
    ([50861],        'ALT',        (40.,  120.),        'U/L',      3),
]
N_LABS       = len(LAB_ITEMS)
N_LAB_TOKENS = sum(item[4] for item in LAB_ITEMS)   # = 25 (4 for eGFR, 3 for rest)

def lab_token_offset(lab_idx):
    """Cumulative token offset for lab at position lab_idx."""
    return sum(LAB_ITEMS[i][4] for i in range(lab_idx))

# ── CKD-EPI 2021 eGFR (race-free) ─────────────────────────────────────────────
def compute_egfr_ckdepi2021(scr, age, is_female):
    """
    CKD-EPI 2021 race-free equation (vectorised — inputs can be arrays).
    scr: serum creatinine mg/dL; age: years; is_female: bool array
    Returns eGFR mL/min/1.73m^2
    """
    is_f  = np.asarray(is_female, dtype=bool)
    kappa = np.where(is_f, 0.7, 0.9)
    alpha = np.where(is_f, -0.241, -0.302)
    ratio = np.asarray(scr, dtype=float) / kappa
    base  = (np.minimum(ratio, 1.0) ** alpha) * (np.maximum(ratio, 1.0) ** -1.200)
    egfr  = 142.0 * base * (0.9938 ** np.asarray(age, dtype=float)) * np.where(is_f, 1.012, 1.0)
    return np.clip(egfr, 0, 120)

def discretise_egfr(egfr_val):
    """Returns level 0=HIGH(≥90), 1=MID(60-89), 2=LOW_MOD(30-59), 3=VERY_LOW(<30)"""
    if egfr_val >= 90: return 0
    if egfr_val >= 60: return 1
    if egfr_val >= 30: return 2
    return 3

# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Load GEM one-to-one mapping
# ══════════════════════════════════════════════════════════════════════════════
print("Step 0: Building ICD-9 → ICD-10 one-to-one GEM mapping ...")

gem_rows = []
with open(GEM_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            gem_rows.append((parts[0], parts[1], parts[2]))

gem_df = pd.DataFrame(gem_rows, columns=['icd9', 'icd10', 'flag'])
gem_df = gem_df[gem_df['flag'].str[0] != '1']   # drop combination entries

icd10_per_icd9 = gem_df.groupby('icd9')['icd10'].nunique()
one2one_icd9   = set(icd10_per_icd9[icd10_per_icd9 == 1].index)
one2one_map    = (gem_df[gem_df['icd9'].isin(one2one_icd9)]
                  .drop_duplicates('icd9')
                  .set_index('icd9')['icd10'].to_dict())
one2one_3char  = {k: v for k, v in one2one_map.items() if len(k) == 3}

print(f"  One-to-one ICD-9 codes: {len(one2one_icd9):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load MIMIC tables
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 1: Loading MIMIC-IV tables ...")

patients = pd.read_csv(data_path('patients.csv'),
    usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'])
patients['birth_year'] = patients['anchor_year'] - patients['anchor_age']

admissions = pd.read_csv(data_path('admissions.csv'),
    usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_days']  = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400

diagnoses  = pd.read_csv(data_path('diagnoses_icd.csv'),
    usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
procedures = pd.read_csv(data_path('procedures_icd.csv'),
    usecols=['subject_id', 'hadm_id', 'chartdate', 'icd_code', 'icd_version'])
transfers  = pd.read_csv(data_path('transfers.csv'),
    usecols=['subject_id', 'hadm_id', 'eventtype', 'careunit', 'intime'])
omr = pd.read_csv(data_path('omr.csv'),
    usecols=['subject_id', 'chartdate', 'result_name', 'result_value'])

print(f"  patients:{len(patients):>9,}  admissions:{len(admissions):>8,}")
print(f"  diagnoses:{len(diagnoses):>8,}  procedures:{len(procedures):>8,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Patient filtering (Option B, same as v4)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Applying Option B patient filter ...")

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
print(f"  Patients passing filter B: {len(valid_pats_b):,}")

patients   = patients[patients['subject_id'].isin(valid_pats_b)].copy()
admissions = admissions[admissions['subject_id'].isin(valid_pats_b)].copy()
diagnoses  = diagnoses[diagnoses['subject_id'].isin(valid_pats_b)].copy()
procedures = procedures[procedures['subject_id'].isin(valid_pats_b)].copy()
transfers  = transfers[transfers['subject_id'].isin(valid_pats_b)].copy()
omr        = omr[omr['subject_id'].isin(valid_pats_b)].copy()

def age_days_from_year_doy(year_col, doy_col, birth_year_col):
    return ((year_col - birth_year_col) * 365.25 + doy_col).clip(lower=0).round().astype(np.int64)

adm = admissions.merge(patients[['subject_id', 'birth_year']], on='subject_id')
adm['admit_year'] = adm['admittime'].dt.year
adm['admit_doy']  = adm['admittime'].dt.day_of_year
adm['age_days']   = age_days_from_year_doy(adm['admit_year'], adm['admit_doy'], adm['birth_year'])
adm_age = adm[['subject_id', 'hadm_id', 'age_days', 'admittime']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — ICD code events
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Building ICD code events ...")

# NOTE: MIMIC-IV dates are shifted to 2100s, so ICD_CUTOFF (2015-10-01) is NOT used
# for date filtering. Use icd_version directly: 10 = ICD-10 era, 9 = ICD-9 era.
adm_age_lookup = adm_age[['subject_id', 'hadm_id', 'age_days']]

# A) ICD-10 diagnoses and procedures (icd_version == 10)
d10 = diagnoses[diagnoses['icd_version'] == 10].copy()
d10['code3'] = d10['icd_code'].str[:3].str.upper()
diag10 = d10.merge(adm_age_lookup, on=['subject_id', 'hadm_id'],
                   how='inner')[['subject_id', 'age_days', 'code3']]

p10 = procedures[procedures['icd_version'] == 10].copy()
p10['code3'] = p10['icd_code'].str[:3].str.upper()
p10['chartdate'] = pd.to_datetime(p10['chartdate'])
p10['chart_year'] = p10['chartdate'].dt.year
p10['chart_doy']  = p10['chartdate'].dt.day_of_year
p10 = p10.merge(patients[['subject_id', 'birth_year']], on='subject_id', how='inner')
p10['age_days'] = age_days_from_year_doy(p10['chart_year'], p10['chart_doy'], p10['birth_year'])
proc10 = p10[['subject_id', 'age_days', 'code3']]

disease_post15 = pd.concat([diag10, proc10], ignore_index=True)

# B) ICD-9 diagnoses → one-to-one GEM → ICD-10 (NEW in v5)
# Filters by icd_version=9; admittime filtering is impossible (all dates shifted to 2100s).
d9 = diagnoses[diagnoses['icd_version'] == 9].copy()

# Map to ICD-10: try full code first, then 3-char prefix
d9['icd10'] = d9['icd_code'].map(one2one_map)
no_full = d9['icd10'].isna()
d9.loc[no_full, 'icd10'] = d9.loc[no_full, 'icd_code'].str[:3].map(one2one_3char)
d9 = d9[d9['icd10'].notna()].copy()
d9['code3'] = d9['icd10'].str[:3].str.upper()
diag9_mapped = d9.merge(adm_age_lookup, on=['subject_id', 'hadm_id'],
                        how='inner')[['subject_id', 'age_days', 'code3']]

n_post15 = len(disease_post15)
n_icd9   = len(diag9_mapped)
print(f"  ICD-10 events (icd_version=10): {n_post15:,}")
print(f"  ICD-9→10 GEM events (icd_version=9, one-to-one): {n_icd9:,}")

all_disease = pd.concat([disease_post15, diag9_mapped], ignore_index=True)
all_disease = all_disease.drop_duplicates(subset=['subject_id', 'age_days', 'code3'])
all_disease = all_disease.sort_values(['subject_id', 'age_days'])
all_disease = all_disease.drop_duplicates(subset=['subject_id', 'code3'], keep='first')
print(f"  After first-occurrence dedup: {len(all_disease):,}")

code_counts = all_disease['code3'].value_counts()
codes_passed = set(code_counts[code_counts >= MIN_CODE_COUNT].index)
print(f"  Codes with ≥{MIN_CODE_COUNT} occurrences: {len(codes_passed):,}")

# ── Anchor token IDs to v4 for finetuning compatibility ──────────────────────
# v4 codes keep their original IDs; new codes (from pre-2015) are appended.
V4_META = os.path.join(BASE, 'data/mimic_data_v4/meta.pkl')
with open(V4_META, 'rb') as _f:
    _v4meta = pickle.load(_f)
v4_code2token = _v4meta['code2token']
v4_DEATH_TOKEN = _v4meta['DEATH_TOKEN']   # = RESERVED + len(v4 ICD codes)

code2token = {c: t for c, t in v4_code2token.items() if c in codes_passed}
new_codes  = sorted(c for c in codes_passed if c not in v4_code2token)
next_id    = v4_DEATH_TOKEN                # first ID after v4 ICD codes
for i, c in enumerate(new_codes):
    code2token[c] = next_id + i

RESERVED    = 9
DEATH_TOKEN = v4_DEATH_TOKEN + len(new_codes)
print(f"  v4 ICD codes retained: {len(v4_code2token):,}  "
      f"New codes from pre-2015: {len(new_codes):,}  "
      f"DEATH_TOKEN: {DEATH_TOKEN}")
if new_codes:
    print(f"  New codes appended: {new_codes[:10]}{'...' if len(new_codes) > 10 else ''}")

codes_kept = sorted(code2token.keys(), key=lambda c: code2token[c])  # ordered by token ID

all_disease = all_disease[all_disease['code3'].isin(code2token)].copy()
all_disease['token_id'] = all_disease['code3'].map(code2token).astype(np.uint32)
all_disease = all_disease[['subject_id', 'age_days', 'token_id']]

LAB_TOKEN_START = DEATH_TOKEN + 1
vocab_size = LAB_TOKEN_START + N_LAB_TOKENS
print(f"  Vocab: {vocab_size} (RESERVED:{RESERVED}, ICD:{len(codes_kept)}, DEATH:1, LAB:{N_LAB_TOKENS})")

def lab_token_id(lab_idx, level):
    return LAB_TOKEN_START + lab_token_offset(lab_idx) + level

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — ICU and ED tokens
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Building ICU / ED events ...")

transfers['intime'] = pd.to_datetime(transfers['intime'])
transfers = transfers.merge(patients[['subject_id', 'birth_year']], on='subject_id', how='inner')
transfers['in_year'] = transfers['intime'].dt.year
transfers['in_doy']  = transfers['intime'].dt.day_of_year
transfers['age_days'] = age_days_from_year_doy(
    transfers['in_year'], transfers['in_doy'], transfers['birth_year'])

icu_mask = transfers['careunit'].str.contains(ICU_CAREUNIT_RE, case=False, na=False)
icu_events = transfers[icu_mask][['subject_id', 'age_days']].copy()
icu_events['token_id'] = np.uint32(7)
icu_events = icu_events.drop_duplicates(subset=['subject_id', 'age_days'])

ed_mask = transfers['careunit'].str.contains(ED_CAREUNIT, case=False, na=False)
ed_events = transfers[ed_mask][['subject_id', 'age_days']].copy()
ed_events['token_id'] = np.uint32(8)
ed_events = ed_events.drop_duplicates(subset=['subject_id', 'age_days'])
print(f"  ICU: {len(icu_events):,}  ED: {len(ed_events):,}")
clinical_events = pd.concat([icu_events, ed_events], ignore_index=True)[['subject_id', 'age_days', 'token_id']]

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — BMI, Sex, Death tokens (same as v4)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Building demographic / death tokens ...")

bmi_raw = omr[omr['result_name'] == 'BMI (kg/m2)'].copy()
bmi_raw['bmi_val'] = pd.to_numeric(bmi_raw['result_value'], errors='coerce')
bmi_raw = bmi_raw.dropna(subset=['bmi_val'])
bmi_raw = bmi_raw[(bmi_raw['bmi_val'] >= 10) & (bmi_raw['bmi_val'] <= 80)]
q33, q67 = bmi_raw['bmi_val'].quantile([0.333, 0.667])

bmi_raw['token_id'] = bmi_raw['bmi_val'].apply(
    lambda v: 4 if v < q33 else (5 if v < q67 else 6)).astype(np.uint32)
bmi_raw['chartdate'] = pd.to_datetime(bmi_raw['chartdate'])
bmi_raw['chart_year'] = bmi_raw['chartdate'].dt.year
bmi_raw['chart_doy']  = bmi_raw['chartdate'].dt.day_of_year
bmi_raw = bmi_raw.merge(patients[['subject_id', 'birth_year']], on='subject_id', how='inner')
bmi_raw['age_days'] = age_days_from_year_doy(
    bmi_raw['chart_year'], bmi_raw['chart_doy'], bmi_raw['birth_year'])
bmi_raw['year_'] = bmi_raw['chartdate'].dt.year
bmi_events = bmi_raw.drop_duplicates(subset=['subject_id', 'year_', 'token_id'])
bmi_events = bmi_events[['subject_id', 'age_days', 'token_id']]
print(f"  BMI events: {len(bmi_events):,}  (q33={q33:.1f} q67={q67:.1f})")

sex_events = patients[['subject_id', 'gender']].copy()
sex_events['token_id'] = sex_events['gender'].map({'F': 2, 'M': 3}).astype('Int64')
sex_events = sex_events.dropna(subset=['token_id'])
sex_events['token_id'] = sex_events['token_id'].astype(np.uint32)
sex_events['age_days'] = np.int64(0)
sex_events = sex_events[['subject_id', 'age_days', 'token_id']]

dead = patients[patients['dod'].notna()].copy()
dead['dod'] = pd.to_datetime(dead['dod'])
dead['dod_year'] = dead['dod'].dt.year
dead['dod_doy']  = dead['dod'].dt.day_of_year
dead['age_days'] = age_days_from_year_doy(
    dead['dod_year'], dead['dod_doy'], dead['birth_year'])
death_events = dead[['subject_id', 'age_days']].copy()
death_events['token_id'] = np.uint32(DEATH_TOKEN)
print(f"  Sex: {len(sex_events):,}  Death: {len(death_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Lab event tokens (v5: eGFR via CKD-EPI 2021, TroponinI)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: Building lab event tokens (v5: eGFR + TroponinI) ...")

valid_sids = set(valid_pats_b.tolist())
all_itemids = set()
for items, _, _, _, _ in LAB_ITEMS:
    all_itemids.update(items)

# Build gender and birthyear maps for eGFR computation
gender_map     = patients.set_index('subject_id')['gender'].to_dict()
birthyear_map  = patients.set_index('subject_id')['birth_year'].to_dict()

print("  Reading labevents.csv in chunks ...")
lab_chunks = []
for chunk in pd.read_csv(
        data_path('labevents.csv'),
        usecols=['subject_id', 'itemid', 'charttime', 'valuenum'],
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
labs_raw = labs_raw.merge(patients[['subject_id', 'birth_year', 'gender']], on='subject_id', how='inner')
labs_raw['chart_year'] = labs_raw['charttime'].dt.year
labs_raw['chart_doy']  = labs_raw['charttime'].dt.day_of_year
labs_raw['age_days']   = age_days_from_year_doy(
    labs_raw['chart_year'], labs_raw['chart_doy'], labs_raw['birth_year'])
# Approximate age in years for CKD-EPI
labs_raw['age_years'] = (labs_raw['chart_year'] - labs_raw['birth_year']).clip(18, 120)
labs_raw['is_female'] = labs_raw['gender'] == 'F'

lab_event_list = []
for lab_idx, (itemids, name, (thresh_lo, thresh_hi), unit, n_levels) in enumerate(LAB_ITEMS):
    sub = labs_raw[labs_raw['itemid'].isin(itemids)].copy()
    sub = sub.sort_values('age_days')
    sub = sub.drop_duplicates(subset=['subject_id'], keep='first')

    if name == 'eGFR':
        # Validate Creatinine range: skip clearly erroneous values
        sub = sub[(sub['valuenum'] > 0) & (sub['valuenum'] <= 20)].copy()
        sub['egfr'] = compute_egfr_ckdepi2021(
            sub['valuenum'].values,
            sub['age_years'].values,
            sub['is_female'].values,
        )
        sub['level'] = sub['egfr'].apply(discretise_egfr)
    else:
        def discretise(v, lo=thresh_lo, hi=thresh_hi):
            if v < lo: return 0
            if v < hi: return 1
            return 2
        sub['level'] = sub['valuenum'].apply(discretise)

    base_tok = lab_token_id(lab_idx, 0)
    sub['token_id'] = (base_tok + sub['level'].astype(np.int64)).astype(np.uint32)

    events = sub[['subject_id', 'age_days', 'token_id']]
    lab_event_list.append(events)
    tid_start = lab_token_id(lab_idx, 0)
    tid_end   = lab_token_id(lab_idx, n_levels - 1)
    print(f"  {name:<12}  token {tid_start}–{tid_end}  {len(sub):>7,} patients")

lab_events = pd.concat(lab_event_list, ignore_index=True)[['subject_id', 'age_days', 'token_id']]
print(f"  Total lab events: {len(lab_events):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Combine and filter
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Combining all events ...")

combined = pd.concat([
    sex_events, bmi_events, clinical_events, all_disease, death_events, lab_events,
], ignore_index=True)
combined['age_days']   = combined['age_days'].astype(np.int64)
combined['token_id']   = combined['token_id'].astype(np.uint32)
combined['subject_id'] = combined['subject_id'].astype(np.int64)

disease_ct = all_disease.groupby('subject_id').size()
valid_pids  = disease_ct[disease_ct >= MIN_DISEASE_EVENTS].index
combined    = combined[combined['subject_id'].isin(valid_pids)]
print(f"  Patients with ≥{MIN_DISEASE_EVENTS} disease codes: {len(valid_pids):,}")
print(f"  Total events: {len(combined):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — 70 / 20 / 10 split
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

# Save subject_id → split mapping for diagnostic 1.2
split_df = pd.DataFrame({
    'subject_id': list(train_pids) + list(val_pids) + list(test_pids),
    'split':      ['train'] * len(train_pids) + ['val'] * len(val_pids) + ['test'] * len(test_pids),
})
split_df.to_csv(os.path.join(OUT_DIR, 'patient_splits.csv'), index=False)
print(f"  Saved patient_splits.csv for cross-version diagnostics")

train_df = combined[combined['subject_id'].isin(train_pids)].copy()
val_df   = combined[combined['subject_id'].isin(val_pids)].copy()
test_df  = combined[combined['subject_id'].isin(test_pids)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Test set temporal cutoff
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: 180-day test cutoff ...")

adm_test  = adm_age[adm_age['subject_id'].isin(test_pids)].copy()
last_adm  = adm_test.groupby('subject_id')['age_days'].max().rename('last_adm_age')
cutoff_map = (last_adm - TEST_CUTOFF_DAYS).rename('cutoff_age')
cutoff_df  = pd.concat([last_adm, cutoff_map], axis=1).reset_index()

test_with_cut = test_df.merge(cutoff_df[['subject_id', 'cutoff_age']], on='subject_id', how='left')
test_with_cut['cutoff_age'] = test_with_cut['cutoff_age'].fillna(0)
test_input  = test_with_cut[test_with_cut['age_days'] <  test_with_cut['cutoff_age']].copy()
test_future = test_with_cut[test_with_cut['age_days'] >= test_with_cut['cutoff_age']].copy()
print(f"  Test input: {len(test_input):,}  Test future: {len(test_future):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Build binary arrays and save
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 10: Building binary arrays ...")

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

train_arr,    train_uids    = build_bin_array(train_df)
val_arr,      val_uids      = build_bin_array(val_df)
test_in_arr,  test_in_uids  = build_bin_array(test_input)
test_fut_arr, test_fut_uids = build_bin_array(test_future)

for fname, arr in [('train.bin', train_arr), ('val.bin', val_arr),
                   ('test_input.bin', test_in_arr), ('test_future.bin', test_fut_arr)]:
    arr.tofile(os.path.join(OUT_DIR, fname))
    print(f"  {fname:<22}: {arr.shape[0]:>9,} events  {arr[:,0].max()+1:>8,} patients  {arr.nbytes/1e6:.1f} MB")

test_in_pid_map = {pid: idx for idx, pid in enumerate(test_in_uids)}
cutoff_out = cutoff_df[cutoff_df['subject_id'].isin(test_pids)].copy()
cutoff_out['patient_idx'] = cutoff_out['subject_id'].map(test_in_pid_map)
cutoff_out.to_csv(os.path.join(OUT_DIR, 'test_cutoffs.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 11 — Labels CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 11: Writing labels ...")

diag_dict   = pd.read_csv(data_path('d_icd_diagnoses.csv'),
    usecols=['icd_code', 'icd_version', 'long_title'])
icd10_desc  = (diag_dict[diag_dict['icd_version'] == 10]
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

# eGFR: 4 levels (same token IDs as v4 Creatinine for first 3 levels)
egfr_labels = [
    f'eGFR_HIGH  (≥90 mL/min)',
    f'eGFR_MID   (60–89 mL/min)',
    f'eGFR_LOW_MOD (30–59 mL/min)',
    f'eGFR_VERY_LOW (<30 mL/min)',
]
for lab_idx, (itemids, name, (lo, hi), unit, n_levels) in enumerate(LAB_ITEMS):
    if name == 'eGFR':
        for lbl in egfr_labels:
            label_names.append(lbl)
    else:
        label_names.append(f'{name}_LOW  (<{lo} {unit})')
        label_names.append(f'{name}_MID  ({lo}–{hi} {unit})')
        label_names.append(f'{name}_HIGH (≥{hi} {unit})')

assert len(label_names) == vocab_size, f"Label count {len(label_names)} != vocab_size {vocab_size}"

label_counts = [None] * vocab_size
for code, tok in code2token.items():
    label_counts[tok] = int(code_counts.get(code, 0))

def icd10_chapter(code3):
    c = str(code3).upper()
    if c[0] in ('A','B'): return ('I. Infectious Diseases',        '#1f77b4')
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
    if c[0] in ('S','T'): return ('XIX. Injury & Poisoning',       '#9edae5')
    if c[0] in ('V','W','X','Y'): return ('XX. External Causes',   '#ad494a')
    if c[0] == 'Z': return ('XXI. Health Factors',                 '#8c6d31')
    return ('Procedures (ICD-10-PCS)', '#aaaaaa')

rows = []
for idx, name in enumerate(label_names):
    row = {'index': idx, 'name': name, 'count': label_counts[idx]}
    code = None
    if RESERVED <= idx < DEATH_TOKEN:
        code = codes_kept[idx - RESERVED]
    if code:
        ch, col = icd10_chapter(code)
        row['ICD-10 Chapter']         = ch
        row['ICD-10 Chapter (short)'] = ch
        row['color']                  = col
    elif idx >= LAB_TOKEN_START:
        row['ICD-10 Chapter']         = 'Lab Values'
        row['ICD-10 Chapter (short)'] = 'Lab Values'
        row['color']                  = '#636efa'
    else:
        row['ICD-10 Chapter']         = 'Technical'
        row['ICD-10 Chapter (short)'] = 'Technical'
        row['color']                  = '#2a52be'
    rows.append(row)

labels_df = pd.DataFrame(rows)
labels_df.to_csv(os.path.join(OUT_DIR, 'mimic_labels.csv'), index=False)
print(f"  mimic_labels.csv: {len(labels_df)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# Step 12 — meta.pkl
# ══════════════════════════════════════════════════════════════════════════════
# Lab tokens in ignore list (model space, after +1 shift)
# Lab tokens stored: LAB_TOKEN_START to vocab_size-1
# Lab tokens model:  LAB_TOKEN_START+1 to vocab_size
lab_ignore = list(range(LAB_TOKEN_START + 1, vocab_size + 1))
ignore_tokens_shifted = [0, 2, 3, 4, 5, 6, 7] + lab_ignore

lab_vocab = {}
for lab_idx, (itemids, name, (lo, hi), unit, n_levels) in enumerate(LAB_ITEMS):
    if name == 'eGFR':
        level_names = ['HIGH', 'MID', 'LOW_MOD', 'VERY_LOW']
        for lvl in range(n_levels):
            tok = lab_token_id(lab_idx, lvl)
            lab_vocab[f'eGFR_{level_names[lvl]}'] = {
                'token_id': tok, 'itemids': itemids,
                'range': (60., 90.), 'unit': 'mL/min', 'level': lvl,
            }
    else:
        for lvl, lvl_name in enumerate(['LOW', 'MID', 'HIGH']):
            tok = lab_token_id(lab_idx, lvl)
            lab_vocab[f'{name}_{lvl_name}'] = {
                'token_id': tok, 'itemids': itemids,
                'range': (lo, hi), 'unit': unit, 'level': lvl,
            }

meta = {
    'vocab_size':       vocab_size,
    'code2token':       code2token,
    'DEATH_TOKEN':      DEATH_TOKEN,
    'RESERVED':         RESERVED,
    'LAB_TOKEN_START':  LAB_TOKEN_START,
    'N_LAB_TOKENS':     N_LAB_TOKENS,
    'lab_vocab':        lab_vocab,
    'ignore_tokens':    ignore_tokens_shifted,
    'bmi_thresholds':   (float(q33), float(q67)),
    'min_code_count':   MIN_CODE_COUNT,
    'test_cutoff_days': TEST_CUTOFF_DAYS,
    'random_seed':      RANDOM_SEED,
    'icd10_cutoff':     '2015-10-01 (historical; MIMIC dates shifted to 2100s)',
    'split': {'train': len(train_pids), 'val': len(val_pids), 'test': len(test_pids)},
    'events': {
        'train': len(train_arr), 'val': len(val_arr),
        'test_input': len(test_in_arr), 'test_future': len(test_fut_arr),
    },
    'improvements': [
        'post2015_icd10_only', 'pre2015_icd9_one2one_gem',
        'no_gems_mapping_post2015', 'first_occurrence_dedup',
        'freq_threshold_25', 'egfr_4level_ckdepi2021',
        'troponin_i_not_t', 'lab_tokens_ignored_in_ce',
    ],
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

pd.DataFrame({
    'token_id':    [code2token[c] for c in codes_kept],
    'code3':       codes_kept,
    'count':       [int(code_counts.get(c, 0)) for c in codes_kept],
    'description': [icd10_desc.get(c, '') for c in codes_kept],
}).to_csv(os.path.join(OUT_DIR, 'vocab_stats.csv'), index=False)

print("\n" + "=" * 62)
print("Phase 2 v5 COMPLETE — Summary")
print("=" * 62)
print(f"  Vocab size:           {vocab_size}")
print(f"    ICD codes:          {len(codes_kept)}  (post-2015 ICD-10 + pre-2015 one-to-one)")
print(f"    Death token:        1  (id={DEATH_TOKEN})")
print(f"    Lab tokens:         {N_LAB_TOKENS}  (ids {LAB_TOKEN_START}–{vocab_size-1}, eGFR 4-level)")
print(f"    Lab tokens in CE:   False  (added to ignore_tokens)")
print(f"  Train: {len(train_pids):>7,} patients / {len(train_arr):>9,} events")
print(f"  Val:   {len(val_pids):>7,} patients / {len(val_arr):>9,} events")
print(f"  Output: {OUT_DIR}")
_v4_vs = _v4meta['vocab_size']   # stored vocab_size from v4 (1493)
print(f"\n  Finetune from v4 ckpt:")
print(f"    v4 vocab_size = {_v4_vs + 1}  →  v5 vocab_size = {vocab_size + 1}  (model space: +1 shift)")
print(f"    New ICD codes: {len(new_codes)}  (appended after v4 ICD range)")
print(f"    New token rows total: {vocab_size - _v4_vs}  (new ICD + eGFR_VERY_LOW)")
print(f"    ignore_tokens: [0,2,3,4,5,6,7] + lab range ({len(ignore_tokens_shifted)} total)")
