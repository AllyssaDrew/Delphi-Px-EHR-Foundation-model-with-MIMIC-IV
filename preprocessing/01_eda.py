"""
Phase 1 - Exploratory Data Analysis of MIMIC-IV
Outputs:
  figures/eda_*.png  - EDA plots
  figures/eda_stats.txt - summary statistics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '../Data')
FIG  = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

def data_path(name):
    return os.path.join(DATA, name, name)

# ── 1. Load core tables ────────────────────────────────────────────────────────
print("Loading MIMIC-IV tables ...")

patients   = pd.read_csv(data_path('patients.csv'))
admissions = pd.read_csv(data_path('admissions.csv'),
                         usecols=['subject_id','hadm_id','admittime','deathtime'])
diagnoses  = pd.read_csv(data_path('diagnoses_icd.csv'),
                         usecols=['subject_id','hadm_id','icd_code','icd_version'])
procedures = pd.read_csv(data_path('procedures_icd.csv'),
                         usecols=['subject_id','hadm_id','chartdate','icd_code','icd_version'])

print(f"  patients:   {len(patients):>8,}")
print(f"  admissions: {len(admissions):>8,}")
print(f"  diagnoses:  {len(diagnoses):>8,}")
print(f"  procedures: {len(procedures):>8,}")

# ── 2. ICD version split ───────────────────────────────────────────────────────
print("\n── ICD version distribution ──")
diag_ver = diagnoses['icd_version'].value_counts().sort_index()
proc_ver  = procedures['icd_version'].value_counts().sort_index()
print("Diagnoses:\n", diag_ver.to_string())
print("Procedures:\n", proc_ver.to_string())

# Patients who have AT LEAST ONE ICD-10 diagnosis
pat_with_icd10 = diagnoses[diagnoses['icd_version']==10]['subject_id'].nunique()
pat_total       = patients['subject_id'].nunique()
print(f"\nPatients with ≥1 ICD-10 diagnosis: {pat_with_icd10:,} / {pat_total:,} ({100*pat_with_icd10/pat_total:.1f}%)")

# ── 3. Patient demographics ────────────────────────────────────────────────────
print("\n── Patient demographics ──")
print(patients['gender'].value_counts().to_string())
print(f"anchor_age - mean: {patients['anchor_age'].mean():.1f}, "
      f"median: {patients['anchor_age'].median():.0f}, "
      f"min: {patients['anchor_age'].min()}, max: {patients['anchor_age'].max()}")
pct_death = patients['dod'].notna().mean()
print(f"Patients with recorded death: {pct_death*100:.1f}%")

# ── 4. Sequence length per patient ────────────────────────────────────────────
print("\n── Events per patient ──")
diag_per_pat  = diagnoses.groupby('subject_id')['icd_code'].count()
icd10_per_pat = diagnoses[diagnoses['icd_version']==10].groupby('subject_id')['icd_code'].count()
adm_per_pat   = admissions.groupby('subject_id')['hadm_id'].count()

for name, s in [('All diag',diag_per_pat), ('ICD-10 diag only',icd10_per_pat), ('Admissions',adm_per_pat)]:
    print(f"  {name:<18}: mean={s.mean():.1f}  median={s.median():.0f}  "
          f"p95={s.quantile(.95):.0f}  max={s.max()}")

# ── 5. Top ICD-10 3-char codes ────────────────────────────────────────────────
print("\n── Top 30 ICD-10 3-char codes (diagnoses) ──")
d10 = diagnoses[diagnoses['icd_version']==10].copy()
d10['code3'] = d10['icd_code'].str[:3].str.upper()
top30 = d10['code3'].value_counts().head(30)
print(top30.to_string())

# ── 6. Temporal span per patient ──────────────────────────────────────────────
print("\n── Temporal span of clinical records ──")
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
span = admissions.groupby('subject_id')['admittime'].agg(first='min', last='max')
span['span_years'] = (span['last'] - span['first']).dt.days / 365.25
print(f"  Median record span: {span['span_years'].median():.1f} years")
print(f"  Mean   record span: {span['span_years'].mean():.1f} years")
print(f"  % patients with span > 1 year:  {(span['span_years']>1).mean()*100:.1f}%")
print(f"  % patients with span > 5 years: {(span['span_years']>5).mean()*100:.1f}%")

# ── 7. Events-per-year density ────────────────────────────────────────────────
# Only for patients with span > 0
span_pos = span[span['span_years'] > 0]
diag_ct   = diag_per_pat.reindex(span_pos.index).fillna(0)
events_py = diag_ct / span_pos['span_years']
print(f"\n── Events per patient-year ──")
print(f"  Median: {events_py.median():.1f}  Mean: {events_py.mean():.1f}  "
      f"p95: {events_py.quantile(.95):.0f}")

# ── 8. Vocabulary size after filtering ────────────────────────────────────────
print("\n── Vocabulary analysis ──")
icd9_3char_count  = diagnoses[diagnoses['icd_version']==9]['icd_code'].str[:3].nunique()
icd10_3char_count = diagnoses[diagnoses['icd_version']==10]['icd_code'].str[:3].nunique()
proc_icd10_3char  = procedures[procedures['icd_version']==10]['icd_code'].str[:3].nunique()
print(f"  Unique ICD-9  3-char codes (diagnoses):  {icd9_3char_count}")
print(f"  Unique ICD-10 3-char codes (diagnoses):  {icd10_3char_count}")
print(f"  Unique ICD-10 3-char codes (procedures): {proc_icd10_3char}")

for min_ct in [5, 10, 50, 100]:
    kept = (d10['code3'].value_counts() >= min_ct).sum()
    print(f"  ICD-10 diag codes with ≥{min_ct:3d} occurrences: {kept}")

# ── 9. Coverage of GEMs-mapped ICD-9 codes ────────────────────────────────────
print("\n── ICD-9 code coverage in GEMs ──")
GEMS = os.path.join(BASE, 'reference/2018_I9gem.txt')
gems_df = pd.read_csv(GEMS, sep=r'\s+', header=None,
                      names=['icd9','icd10','flags'], dtype=str)
# Flag bit 2 (no_map) = flags[1] == '1'
gems_df = gems_df[gems_df['flags'].str[1] != '1']  # exclude no-map entries
gems_icd9 = set(gems_df['icd9'].str.strip())

d9 = diagnoses[diagnoses['icd_version']==9].copy()
d9['icd9_clean'] = d9['icd_code'].str.replace('.','',regex=False).str.strip().str.upper()
in_gems = d9['icd9_clean'].isin(gems_icd9)
print(f"  ICD-9 diagnosis records mappable via GEMs: {in_gems.sum():,} / {len(d9):,} "
      f"({in_gems.mean()*100:.1f}%)")

# ── 10. FIGURES ────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('MIMIC-IV EDA — Delphi Pipeline', fontsize=14, fontweight='bold')

# (a) Age at anchor
ax = axes[0,0]
patients['anchor_age'].hist(bins=40, ax=ax, color='steelblue', edgecolor='white', linewidth=0.4)
ax.set_title('(a) Age Distribution at Anchor Year')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Count')

# (b) Gender split
ax = axes[0,1]
patients['gender'].value_counts().plot.bar(ax=ax, color=['#ff9f9f','#9fb8ff'], edgecolor='white')
ax.set_title('(b) Sex Distribution')
ax.set_xlabel(''); ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)

# (c) Diagnoses per patient (log scale)
ax = axes[0,2]
diag_per_pat.clip(upper=200).hist(bins=60, ax=ax, color='darkorange', edgecolor='white', linewidth=0.4)
ax.set_title('(c) Diagnoses per Patient (clipped @200)')
ax.set_xlabel('# diagnoses'); ax.set_ylabel('# patients')
ax.set_yscale('log')

# (d) ICD-10 diagnoses per patient
ax = axes[1,0]
icd10_per_pat.reindex(patients['subject_id'], fill_value=0).clip(upper=100).hist(
    bins=50, ax=ax, color='seagreen', edgecolor='white', linewidth=0.4)
ax.set_title('(d) ICD-10 Diagnoses per Patient')
ax.set_xlabel('# ICD-10 diagnoses'); ax.set_ylabel('# patients')
ax.set_yscale('log')

# (e) ICD version split (pie)
ax = axes[1,1]
ver_counts = diagnoses['icd_version'].value_counts().sort_index()
ax.pie(ver_counts, labels=[f'ICD-{v}\n({c/1e6:.1f}M)' for v,c in zip(ver_counts.index, ver_counts)],
       autopct='%1.1f%%', colors=['#aec7e8','#ffbb78'], startangle=90)
ax.set_title('(e) ICD Version Split (Diagnoses)')

# (f) Temporal record span
ax = axes[1,2]
span['span_years'].clip(upper=20).hist(bins=50, ax=ax, color='mediumpurple', edgecolor='white', linewidth=0.4)
ax.set_title('(f) Record Span per Patient (clipped @20y)')
ax.set_xlabel('Years'); ax.set_ylabel('# patients')

# (g) Top 20 ICD-10 3-char codes
ax = axes[2,0]
top20 = d10['code3'].value_counts().head(20)
top20.sort_values().plot.barh(ax=ax, color='steelblue')
ax.set_title('(g) Top 20 ICD-10 3-char Codes')
ax.set_xlabel('Count')

# (h) Events per patient-year
ax = axes[2,1]
events_py.clip(upper=50).hist(bins=50, ax=ax, color='salmon', edgecolor='white', linewidth=0.4)
ax.set_title('(h) Events per Patient-Year')
ax.set_xlabel('Events/year'); ax.set_ylabel('# patients')
ax.set_yscale('log')

# (i) Admissions per patient
ax = axes[2,2]
adm_per_pat.clip(upper=30).hist(bins=30, ax=ax, color='teal', edgecolor='white', linewidth=0.4)
ax.set_title('(i) Admissions per Patient')
ax.set_xlabel('# admissions'); ax.set_ylabel('# patients')
ax.set_yscale('log')

plt.tight_layout()
out_fig = os.path.join(FIG, 'eda_overview.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_fig}")

# ── Write stats summary ────────────────────────────────────────────────────────
stats_path = os.path.join(FIG, 'eda_stats.txt')
with open(stats_path, 'w') as f:
    f.write("=== MIMIC-IV EDA Summary ===\n\n")
    f.write(f"Patients total: {pat_total:,}\n")
    f.write(f"Patients with ≥1 ICD-10 diagnosis: {pat_with_icd10:,} ({100*pat_with_icd10/pat_total:.1f}%)\n")
    f.write(f"Patients with recorded death: {int(pct_death*pat_total):,} ({pct_death*100:.1f}%)\n\n")

    f.write("── ICD version split (diagnoses) ──\n")
    f.write(diag_ver.to_string() + "\n\n")

    f.write("── Events per patient ──\n")
    for name, s in [('All diagnoses', diag_per_pat),
                    ('ICD-10 diagnoses', icd10_per_pat),
                    ('Admissions', adm_per_pat)]:
        f.write(f"  {name}: mean={s.mean():.1f} median={s.median():.0f} "
                f"p95={s.quantile(.95):.0f} max={s.max()}\n")

    f.write(f"\n── Record span ──\n")
    f.write(f"  Median: {span['span_years'].median():.1f} years\n")
    f.write(f"  Mean:   {span['span_years'].mean():.1f} years\n")

    f.write(f"\n── ICD-10 vocabulary (diagnoses) ──\n")
    for min_ct in [5, 10, 50, 100]:
        kept = (d10['code3'].value_counts() >= min_ct).sum()
        f.write(f"  ≥{min_ct:3d} occurrences: {kept} codes\n")

    f.write(f"\n── GEMs coverage ──\n")
    f.write(f"  ICD-9 records mappable: {in_gems.sum():,} / {len(d9):,} ({in_gems.mean()*100:.1f}%)\n")

    f.write(f"\n── Top 30 ICD-10 3-char codes ──\n")
    f.write(top30.to_string() + "\n")

print(f"  Saved: {stats_path}")
print("\nPhase 1 EDA complete.")
