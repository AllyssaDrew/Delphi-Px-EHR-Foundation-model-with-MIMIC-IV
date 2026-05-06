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
Phase A: discharge.csv statistics + hadm_id linkage map for Phase C.

Outputs:
  Phase_A/hadm_note_counts.csv       — records-per-hadm_id distribution
  Phase_A/coverage_report.txt        — hadm/patient coverage by split
  Phase_A/hadm_linkage_map.csv       — subject_id, hadm_id, age_days_at_disch, split
                                        (only rows where discharge note exists)
  Phase_A/merged_texts.csv           — subject_id, hadm_id, text  (already 1-to-1)
"""
import os, sys, pickle
import numpy as np
import pandas as pd

PIPE      = PIPELINE_DIR
# Path to raw MIMIC-IV-Note data directory (must contain note/discharge.csv)
DATA      = Path(os.environ.get('MIMIC_NOTE_DIR',
                                 _ROOT / 'Data'))
OUT       = PIPE / 'multimodal_notes/Phase_A'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading discharge.csv...", flush=True)
disc = pd.read_csv(DATA / 'note/discharge.csv',
                   usecols=['subject_id', 'hadm_id', 'note_seq', 'charttime', 'text'])
print(f"  {len(disc):,} rows, {disc['subject_id'].nunique():,} subjects, "
      f"{disc['hadm_id'].nunique():,} unique hadm_ids", flush=True)

print("Loading admissions.csv...", flush=True)
adm = pd.read_csv(DATA / 'admissions.csv/admissions.csv',
                  usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])

print("Loading patients.csv...", flush=True)
pts = pd.read_csv(DATA / 'patients.csv/patients.csv',
                  usecols=['subject_id', 'anchor_age', 'anchor_year'])
pts['birth_year'] = pts['anchor_year'] - pts['anchor_age']

print("Loading patient_splits.csv...", flush=True)
splits = pd.read_csv(PIPE / 'data/mimic_data_v5/patient_splits.csv')

# ── 1. Records-per-hadm distribution ─────────────────────────────────────────
per_hadm = disc.groupby('hadm_id').size()
counts = per_hadm.value_counts().sort_index()
print("\n=== Records per hadm_id ===")
for n, c in counts.items():
    print(f"  {n} record(s): {c:,} hadm_ids ({100*c/len(per_hadm):.1f}%)")
counts.to_csv(OUT / 'hadm_note_counts.csv', header=['count'])

# ── 2. Merge admissions + patients to get age_days at dischtime ───────────────
print("\nComputing age_days at dischtime...", flush=True)

adm = adm.merge(pts[['subject_id', 'birth_year']], on='subject_id', how='inner')
adm['dischtime'] = pd.to_datetime(adm['dischtime'])
adm['disch_year'] = adm['dischtime'].dt.year
adm['disch_doy']  = adm['dischtime'].dt.day_of_year
adm['age_days_at_disch'] = (
    (adm['disch_year'] - adm['birth_year']) * 365.25 + adm['disch_doy']
).clip(lower=0).round().astype(np.int64)

# ── 3. Build hadm linkage map (v5 patients only, with discharge notes) ────────
print("Building hadm linkage map...", flush=True)

v5_adm = adm[adm['subject_id'].isin(splits['subject_id'])].copy()
v5_adm_with_note = v5_adm[v5_adm['hadm_id'].isin(disc['hadm_id'])].copy()
v5_adm_with_note = v5_adm_with_note.merge(
    splits[['subject_id', 'split']], on='subject_id', how='left')

linkage = v5_adm_with_note[['subject_id', 'hadm_id', 'age_days_at_disch', 'split']].copy()
linkage = linkage.sort_values(['subject_id', 'age_days_at_disch']).reset_index(drop=True)
linkage.to_csv(OUT / 'hadm_linkage_map.csv', index=False)
print(f"  Linkage map: {len(linkage):,} rows", flush=True)

# ── 4. Coverage report ────────────────────────────────────────────────────────
lines = []
lines.append("=== Phase A Coverage Report ===\n")
lines.append(f"discharge.csv total: {len(disc):,} rows, {disc['hadm_id'].nunique():,} unique hadm_ids\n")
lines.append(f"note_seq range: {disc['note_seq'].min()} – {disc['note_seq'].max()} "
             f"(note_seq=1 absent: MIMIC-IV convention)\n\n")

all_v5_adm = adm[adm['subject_id'].isin(splits['subject_id'])]
total_v5_hadm = len(all_v5_adm)
covered_hadm  = v5_adm_with_note['hadm_id'].nunique()
lines.append(f"v5 admissions total:              {total_v5_hadm:,}\n")
lines.append(f"v5 admissions with discharge note: {covered_hadm:,} ({100*covered_hadm/total_v5_hadm:.1f}%)\n")
lines.append(f"v5 admissions WITHOUT note:        {total_v5_hadm - covered_hadm:,} "
             f"({100*(total_v5_hadm-covered_hadm)/total_v5_hadm:.1f}%) → no NOTE token inserted\n\n")

for split in ['train', 'val', 'test']:
    s_pids  = set(splits[splits['split'] == split]['subject_id'])
    s_hadms = all_v5_adm[all_v5_adm['subject_id'].isin(s_pids)]
    s_cover = v5_adm_with_note[v5_adm_with_note['split'] == split]
    lines.append(f"{split:5s}: {len(s_pids):,} patients | "
                 f"{len(s_hadms):,} admissions | "
                 f"{s_cover['hadm_id'].nunique():,} with note "
                 f"({100*s_cover['hadm_id'].nunique()/len(s_hadms):.1f}%)\n")

lines.append("\n=== age_days_at_disch sanity check ===\n")
lines.append(str(linkage['age_days_at_disch'].describe()) + "\n")

report = "".join(lines)
print("\n" + report)
with open(OUT / 'coverage_report.txt', 'w') as f:
    f.write(report)

# ── 5. Save merged_texts (already 1-to-1, just clean columns) ────────────────
print("Saving merged_texts.csv...", flush=True)
merged = disc[['subject_id', 'hadm_id', 'text']].copy()
merged.to_csv(OUT / 'merged_texts.csv', index=False)
print(f"  merged_texts.csv: {len(merged):,} rows", flush=True)

print("\nPhase A complete.", flush=True)
