"""
Phase 3 — Sex + Age Logistic Regression Baseline AUC
Computes per-disease AUC using only sex and age as predictors.
This is the baseline Delphi must beat to confirm the model learns real signal.

Reads val.bin (Option A data) and outputs:
  eval_output/baseline_auc.csv  — per-disease AUC from sex+age LR
  eval_output/baseline_summary.txt — mean / median AUC summary

Usage:
  python3 03_baseline_auc.py [--data_dir data/mimic_data] [--min_cases 20]
"""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',  default='data/mimic_data')
parser.add_argument('--min_cases', type=int, default=20,
                    help='Minimum cases per disease to compute AUC')
parser.add_argument('--age_groups_start', type=int, default=20)
parser.add_argument('--age_groups_end',   type=int, default=80)
parser.add_argument('--age_step',         type=int, default=5)
args = parser.parse_args()

DATA_DIR  = os.path.join(BASE, args.data_dir)
OUT_DIR   = os.path.join(BASE, 'eval_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_DIR} ...")

with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size  = meta['vocab_size']
DEATH_TOKEN = meta['DEATH_TOKEN']
RESERVED    = meta['RESERVED']

val = np.fromfile(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint32).reshape(-1, 3)
print(f"  val.bin: {val.shape[0]:,} events, {val[:,0].max()+1:,} patients")

labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))

# ── Build patient-level feature matrix ────────────────────────────────────────
print("Building patient-level features ...")

n_patients = int(val[:,0].max()) + 1
SEX_F, SEX_M = 2, 3

# For each patient: sex (0=F,1=M) and median age (in years)
patient_sex  = np.full(n_patients, -1, dtype=np.int8)
patient_age  = np.zeros(n_patients, dtype=np.float32)
age_counts   = np.zeros(n_patients, dtype=np.int32)

for row in val:
    pid, age_d, tok = int(row[0]), float(row[1]), int(row[2])
    if tok == SEX_F:
        patient_sex[pid] = 0
    elif tok == SEX_M:
        patient_sex[pid] = 1
    if tok >= RESERVED:
        patient_age[pid] += age_d / 365.25
        age_counts[pid]  += 1

# Patients with no age info: use 0 (will be filtered out by min_cases anyway)
with np.errstate(invalid='ignore'):
    patient_age = np.where(age_counts > 0, patient_age / age_counts, 0)

# ── Build disease occurrence matrix (sparse loop) ─────────────────────────────
print("Building disease occurrence matrix ...")

# For each patient × disease: did patient ever have this disease?
disease_tokens = list(range(RESERVED, DEATH_TOKEN))
n_diseases     = len(disease_tokens)

# Use a set-based approach for memory efficiency
patient_diseases = [set() for _ in range(n_patients)]
for row in val:
    pid, tok = int(row[0]), int(row[2])
    if RESERVED <= tok < DEATH_TOKEN:
        patient_diseases[pid].add(tok)

# ── Per-disease AUC: sex + age logistic regression ────────────────────────────
print(f"Computing baseline AUC for {n_diseases} diseases ...")

valid_mask = (patient_sex >= 0) & (age_counts > 0)
X_full = np.stack([patient_sex.astype(np.float32), patient_age], axis=1)

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)

results = []
age_groups = range(args.age_groups_start, args.age_groups_end, args.age_step)

for tok in disease_tokens:
    y = np.array([1 if tok in patient_diseases[i] else 0 for i in range(n_patients)])
    y_valid = y[valid_mask]
    X_valid = X_scaled_full[valid_mask]

    n_cases = y_valid.sum()
    n_ctrl  = (y_valid == 0).sum()

    if n_cases < args.min_cases or n_ctrl < args.min_cases:
        continue

    try:
        lr = LogisticRegression(max_iter=200, C=1.0)
        lr.fit(X_valid, y_valid)
        probs = lr.predict_proba(X_valid)[:, 1]
        auc   = roc_auc_score(y_valid, probs)
    except Exception:
        continue

    label = labels_df.iloc[tok]['event_name'] if tok < len(labels_df) else str(tok)
    results.append({
        'token_id':   tok,
        'event_name': label,
        'n_cases':    int(n_cases),
        'n_controls': int(n_ctrl),
        'auc_baseline': float(auc),
    })

df_base = pd.DataFrame(results)
df_base = df_base.sort_values('auc_baseline', ascending=False)

out_csv = os.path.join(OUT_DIR, 'baseline_auc.csv')
df_base.to_csv(out_csv, index=False)

print(f"\n── Baseline AUC Summary (sex + age logistic regression) ──")
print(f"  Diseases evaluated:    {len(df_base):,}")
print(f"  Mean  AUC:             {df_base['auc_baseline'].mean():.4f}")
print(f"  Median AUC:            {df_base['auc_baseline'].median():.4f}")
print(f"  AUC > 0.60:            {(df_base['auc_baseline'] > 0.60).sum()}")
print(f"  AUC > 0.70:            {(df_base['auc_baseline'] > 0.70).sum()}")
print(f"\n  Top 10 (easiest for baseline):")
print(df_base.head(10)[['event_name','auc_baseline','n_cases']].to_string(index=False))
print(f"\n  Bottom 10 (hardest for baseline):")
print(df_base.tail(10)[['event_name','auc_baseline','n_cases']].to_string(index=False))

# Save summary
with open(os.path.join(OUT_DIR, 'baseline_summary.txt'), 'w') as f:
    f.write("=== Sex+Age Baseline AUC Summary ===\n\n")
    f.write(f"Diseases evaluated: {len(df_base)}\n")
    f.write(f"Mean  AUC: {df_base['auc_baseline'].mean():.4f}\n")
    f.write(f"Median AUC: {df_base['auc_baseline'].median():.4f}\n")
    f.write(f"AUC > 0.60: {(df_base['auc_baseline'] > 0.60).sum()}\n")
    f.write(f"AUC > 0.70: {(df_base['auc_baseline'] > 0.70).sum()}\n\n")
    f.write("Top 10:\n")
    f.write(df_base.head(10)[['event_name','auc_baseline','n_cases']].to_string(index=False))

print(f"\nSaved: {out_csv}")
print("Phase 3 baseline complete.")
