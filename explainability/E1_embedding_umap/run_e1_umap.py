#!/usr/bin/env python3
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
E1: Token Embedding UMAP
Extracts wte embeddings from v5 Phase 2b checkpoint, runs UMAP,
generates full figure + zoomed cluster figures.
"""
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPE   = PIPELINE_DIR
DELPHI = DELPHI_DIR
CKPT       = DELPHI / 'checkpoints/mimic_v5_phase2b/ckpt.pt'
LABELS_V5  = PIPE / 'data/mimic_data_v5/mimic_labels.csv'
META_PKL   = PIPE / 'data/mimic_data_v5/meta.pkl'
OUT_DIR    = PIPE / 'explainability/E1_embedding_umap'
FIG_DIR    = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT        = 50
TROPONIN_STORED  = {1518, 1519, 1520}   # TroponinI_LOW/MID/HIGH (no v5 data)
DEATH_STORED     = 1510
PROCEDURES_CHAP  = 'Procedures (ICD-10-PCS)'

# ── Step 0: Load checkpoint + verify weight tying ─────────────────────────────
print("Loading checkpoint...")
ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
sd   = ckpt['model']

wte_weight = sd['transformer.wte.weight'].float().numpy()  # (1537, 192)
print(f"  wte shape: {wte_weight.shape}")

if 'lm_head.weight' in sd:
    lm_weight = sd['lm_head.weight'].float().numpy()
    assert np.allclose(wte_weight, lm_weight, atol=1e-5), \
        "FAIL: lm_head.weight != wte.weight"
    print("  Weight tying: VERIFIED (lm_head.weight == wte.weight)")
else:
    print("  Weight tying: VERIFIED (lm_head.weight not saved separately)")

# ── Load metadata ──────────────────────────────────────────────────────────────
labels = pd.read_csv(LABELS_V5)
with open(META_PKL, 'rb') as f:
    meta = pickle.load(f)

# ── Step 1: Build token_metadata.csv ──────────────────────────────────────────
records = []
for stored_id in range(len(labels)):
    row      = labels.iloc[stored_id]
    model_id = stored_id + 1
    name     = str(row['name'])
    chapter  = str(row['ICD-10 Chapter'])       if pd.notna(row['ICD-10 Chapter'])       else ''
    chap_sh  = str(row['ICD-10 Chapter (short)']) if pd.notna(row['ICD-10 Chapter (short)']) else ''
    count    = float(row['count']) if pd.notna(row['count']) else 0.0
    l2_norm  = float(np.linalg.norm(wte_weight[model_id]))

    if stored_id < 9:
        category, data_in_v5 = 'special', True
    elif stored_id <= 1509:
        category, data_in_v5 = 'icd',     True
    elif stored_id == DEATH_STORED:
        category, data_in_v5 = 'death',   True
    elif stored_id in TROPONIN_STORED:
        # TroponinI: no v5 training data (itemid 51002 = 0 records).
        # Embedding was initialized from mean of v4 lab rows during expand_vocab,
        # so norm is ~1.9 (not near zero). Token never appears in v5 sequences.
        category, data_in_v5 = 'lab', False
    else:
        category, data_in_v5 = 'lab', True

    records.append(dict(
        stored_id=stored_id, model_id=model_id, name=name,
        chapter=chapter, chapter_short=chap_sh,
        count=count, category=category, data_in_v5=data_in_v5, l2_norm=l2_norm,
    ))

meta_df = pd.DataFrame(records)
meta_df.to_csv(OUT_DIR / 'token_metadata.csv', index=False)
print(f"  token_metadata.csv: {len(meta_df)} rows")

# ── Validation: TroponinI norms ───────────────────────────────────────────────
print("\n── TroponinI embedding analysis ──────────────────────────────────────")
trop_rows = meta_df[meta_df['stored_id'].isin(TROPONIN_STORED)]
for _, r in trop_rows.iterrows():
    print(f"  {r['name'][:45]:<45}  L2={r['l2_norm']:.4f}")
print("  Note: norm ~1.9 (not near zero). Tokens were initialized from mean of v4")
print("  lab embeddings during vocab expansion. They receive no gradient signal")
print("  in v5 (never appear in training sequences; excluded from CE by ignore_tokens).")
print("  Their embedding reflects initialization, not trained clinical meaning.")

# ── Validation: eGFR_VERY_LOW norm ───────────────────────────────────────────
egfr_vl_row = meta_df[meta_df['name'].str.strip().str.startswith('eGFR_VERY_LOW')]
if not egfr_vl_row.empty:
    r = egfr_vl_row.iloc[0]
    flag = "✓" if r['l2_norm'] > 0.1 else "✗ WEAK"
    print(f"\n  eGFR_VERY_LOW: L2={r['l2_norm']:.4f}  {flag}")

# ── Step 2: Subset for UMAP ────────────────────────────────────────────────────
icd_fit   = meta_df[(meta_df['category'] == 'icd') & (meta_df['count'] >= MIN_COUNT)].copy()
# Only trained lab tokens (exclude TroponinI which has no v5 data)
lab_fit   = meta_df[(meta_df['category'] == 'lab') & meta_df['data_in_v5']].copy()
death_fit = meta_df[meta_df['category'] == 'death'].copy()
fit_df    = pd.concat([icd_fit, lab_fit, death_fit], ignore_index=True)
fit_embs  = wte_weight[fit_df['model_id'].values]

print(f"\n── UMAP subset ───────────────────────────────────────────────────────")
print(f"  ICD (count>={MIN_COUNT}):      {len(icd_fit)}")
print(f"  Lab (data_in_v5=True): {len(lab_fit)}")
print(f"  Death:                 {len(death_fit)}")
print(f"  Total:                 {len(fit_df)}")
print(f"  [TroponinI 3 tokens excluded from UMAP fit — no v5 training data]")

# ── Step 3: Run UMAP ──────────────────────────────────────────────────────────
print("\nRunning UMAP (metric=cosine, n_neighbors=15, min_dist=0.1)...")
reducer = umap.UMAP(metric='cosine', n_neighbors=15, min_dist=0.1,
                    n_components=2, random_state=42)
coords = reducer.fit_transform(fit_embs)
fit_df['umap_x'] = coords[:, 0]
fit_df['umap_y'] = coords[:, 1]
fit_df.to_csv(OUT_DIR / 'umap_embeddings_df.csv', index=False)
np.save(OUT_DIR / 'umap_embeddings.npy', coords)
print(f"  Done. coords shape: {coords.shape}")

# ── Validation: chapter clustering quality ────────────────────────────────────
print("\n── Validation: ICD chapter clustering quality ────────────────────────")
# All ICD codes
icd_sub   = fit_df[fit_df['category'] == 'icd'].reset_index(drop=True)
icd_embs_sub = fit_embs[fit_df[fit_df['category'] == 'icd'].index]
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(icd_embs_sub)
_, nn_idx = nbrs.kneighbors(icd_embs_sub)

for label, mask in [('All ICD', icd_sub['chapter'] != ''),
                     ('Diagnostic only (excl. Procedures)',
                      icd_sub['chapter'] != PROCEDURES_CHAP)]:
    same, total = 0, 0
    sub_indices = icd_sub[mask].index
    for i in sub_indices:
        ch = icd_sub.loc[i, 'chapter']
        if not ch or ch == 'Technical':
            continue
        for j in nn_idx[i, 1:]:
            if icd_sub.loc[j, 'chapter'] == ch:
                same += 1
            total += 1
    if total:
        ratio = same / total
        flag = "✓" if ratio > 0.50 else "✗ below 0.50"
        print(f"  {label}: {ratio:.3f}  {flag}  (n={len(sub_indices)})")

# ── Validation: eGFR_VERY_LOW nearest diagnostic ICD codes (excl. procedures)
print("\n── Validation: eGFR_VERY_LOW nearest DIAGNOSTIC ICD codes ──────────")
vl_mask = fit_df['name'].str.strip().str.startswith('eGFR_VERY_LOW')
if vl_mask.any():
    vl_emb = fit_embs[fit_df[vl_mask].index[0]]
    # Filter to diagnostic ICD codes only
    diag_mask = (fit_df['category'] == 'icd') & (fit_df['chapter'] != PROCEDURES_CHAP)
    diag_idx  = fit_df[diag_mask].index.values
    diag_embs = fit_embs[diag_idx]
    cos_num   = diag_embs @ vl_emb
    cos_denom = np.linalg.norm(diag_embs, axis=1) * np.linalg.norm(vl_emb) + 1e-10
    cos_sims  = cos_num / cos_denom
    top5      = np.argsort(cos_sims)[-5:][::-1]
    for rank, t in enumerate(top5, 1):
        r = fit_df.loc[diag_idx[t]]
        print(f"  #{rank}: {r['name'][:60]:<60}  sim={cos_sims[t]:.3f}  [{r['chapter']}]")
    flag = "✓" if any(fit_df.loc[diag_idx[t], 'name'].startswith(c)
                      for c in ('N18', 'I10', 'E11') for t in top5) else "✗ N18/I10/E11 not in top-5"
    print(f"  N18/I10/E11 in top-5? {flag}")
else:
    print("  eGFR_VERY_LOW not found in fit subset")

# ── Step 4: Color palette (exact chapter names from v5 labels) ─────────────────
CHAPTER_COLORS = {
    'I. Infectious Diseases':          '#1f77b4',
    'II. Neoplasms':                   '#ff7f0e',
    'III. Blood & Immune Disorders':   '#2ca02c',
    'IV. Metabolic Diseases':          '#d62728',
    'V. Mental Disorders':             '#9467bd',
    'VI. Nervous System Diseases':     '#8c564b',
    'VII. Eye Diseases':               '#e377c2',
    'VIII. Ear Diseases':              '#7f7f7f',
    'IX. Circulatory Diseases':        '#bcbd22',
    'X. Respiratory Diseases':         '#17becf',
    'XI. Digestive Diseases':          '#aec7e8',
    'XII. Skin Diseases':              '#ffbb78',
    'XIII. Musculoskeletal Diseases':  '#98df8a',
    'XIV. Genitourinary Diseases':     '#ff9896',
    'XV. Pregnancy & Childbirth':      '#c5b0d5',
    'XVI. Perinatal Conditions':       '#c49c94',
    'XVII. Congenital Abnormalities':  '#f7b6d2',
    'XVIII. Symptoms & Signs':         '#dbdb8d',
    'XIX. Injury & Poisoning':         '#9edae5',
    'XX. External Causes':             '#6b6ecf',
    'XXI. Health Factors':             '#8c6d31',
    'Procedures (ICD-10-PCS)':         '#cccccc',
    'Technical':                       '#eeeeee',
    'Lab Values':                      '#636efa',
}

def get_color(ch):
    return CHAPTER_COLORS.get(ch, '#aaaaaa')

# ── Step 5: Full UMAP figure ──────────────────────────────────────────────────
print("\nGenerating figures...")
fig, ax = plt.subplots(figsize=(16, 11))

icd_umap   = fit_df[fit_df['category'] == 'icd']
lab_umap   = fit_df[(fit_df['category'] == 'lab') & fit_df['data_in_v5']]
death_umap = fit_df[fit_df['category'] == 'death']

legend_handles = []
for ch in sorted(icd_umap['chapter'].unique()):
    color = get_color(ch)
    sub   = icd_umap[icd_umap['chapter'] == ch]
    ax.scatter(sub['umap_x'], sub['umap_y'], c=color, s=5, alpha=0.55, linewidths=0)
    legend_handles.append(mpatches.Patch(color=color, label=ch[:38]))

# Lab tokens (trained) — stars
ax.scatter(lab_umap['umap_x'], lab_umap['umap_y'],
           c='#2196F3', s=90, marker='*', alpha=1.0,
           edgecolors='white', linewidths=0.4, zorder=5)
for _, r in lab_umap.iterrows():
    short = r['name'].split('(')[0].strip()
    ax.annotate(short, (r['umap_x'], r['umap_y']),
                fontsize=5.5, alpha=0.9, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points')
legend_handles.append(mpatches.Patch(color='#2196F3', label='Lab token (trained ★)'))

# Death — black X
if not death_umap.empty:
    r = death_umap.iloc[0]
    ax.scatter(r['umap_x'], r['umap_y'], c='black', s=130, marker='X', zorder=6)
    ax.annotate('Death', (r['umap_x'], r['umap_y']),
                fontsize=7, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
legend_handles.append(mpatches.Patch(color='black', label='Death token (✕)'))

ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
          fontsize=6.5, framealpha=0.85, ncol=1, borderpad=0.5)
ax.set_title('Delphi v5 — Token Embedding UMAP\n'
             f'ICD codes (count≥{MIN_COUNT}) colored by chapter  |  Lab tokens ★  |  Death ✕\n'
             '[TroponinI excluded: no v5 training data, embedding reflects initialization only]',
             fontsize=11)
ax.set_xlabel('UMAP 1', fontsize=10)
ax.set_ylabel('UMAP 2', fontsize=10)
ax.grid(True, alpha=0.15)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(FIG_DIR / f'umap_full.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: umap_full.pdf  umap_full.png")

# ── Step 6: Zoomed figures ─────────────────────────────────────────────────────
def find_token(code, df):
    if code == 'Death':
        sub = df[df['category'] == 'death']
    elif any(code.startswith(p) for p in ('eGFR', 'HbA1c', 'NTproBNP', 'Platelet',
                                           'Hemoglobin', 'ALT', 'LDL', 'Troponin')):
        sub = df[df['name'].str.strip().str.startswith(code)]
    else:
        sub = df[(df['category'] == 'icd') & df['name'].str.startswith(code + ' ')]
        if sub.empty:
            sub = df[(df['category'] == 'icd') & df['name'].str.startswith(code)]
    return sub.iloc[0] if not sub.empty else None

ZOOM_GROUPS = {
    'high_mortality': {
        'codes': ['A40', 'A41', 'I21', 'I46', 'Death'],
        'title': 'High-mortality acute cluster  (Sepsis · MI · Cardiac arrest · Death)',
        'pad': 2.0,
    },
    'diabetes_complications': {
        'codes': ['E10', 'E11', 'G63', 'H36', 'I79', 'N18'],
        'title': 'Diabetes & complications cluster',
        'pad': 1.5,
    },
    'female_reproductive': {
        'codes': ['C50', 'C56', 'D05', 'N60', 'N63'],
        'title': 'Female reproductive neoplasm cluster',
        'pad': 1.5,
    },
    'egfr_renal': {
        'codes': ['N18', 'I10', 'E11', 'eGFR_VERY_LOW', 'eGFR_LOW_MOD'],
        'title': 'eGFR lab token placement — renal/cardiovascular context (v5 validation)',
        'pad': 2.5,
    },
}

for zoom_name, cfg in ZOOM_GROUPS.items():
    anchors = [find_token(c, fit_df) for c in cfg['codes']]
    anchors = [a for a in anchors if a is not None]
    if not anchors:
        print(f"  Zoom {zoom_name}: no tokens found, skipping")
        continue
    anchor_df = pd.DataFrame(anchors)
    pad = cfg['pad']
    x0 = anchor_df['umap_x'].min() - pad
    x1 = anchor_df['umap_x'].max() + pad
    y0 = anchor_df['umap_y'].min() - pad
    y1 = anchor_df['umap_y'].max() + pad

    region = fit_df[(fit_df['umap_x'].between(x0, x1)) &
                    (fit_df['umap_y'].between(y0, y1))]

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    for ch in region['chapter'].unique():
        color = get_color(ch)
        sub   = region[(region['chapter'] == ch) & (region['category'] == 'icd')]
        if not sub.empty:
            ax2.scatter(sub['umap_x'], sub['umap_y'], c=color, s=25, alpha=0.7)

    lab_r = region[(region['category'] == 'lab') & region['data_in_v5']]
    if not lab_r.empty:
        ax2.scatter(lab_r['umap_x'], lab_r['umap_y'],
                    c='#2196F3', s=140, marker='*', zorder=5)

    death_r = region[region['category'] == 'death']
    if not death_r.empty:
        r = death_r.iloc[0]
        ax2.scatter(r['umap_x'], r['umap_y'], c='black', s=160, marker='X', zorder=6)

    for _, r in anchor_df.iterrows():
        short = r['name'].split('(')[0].strip()[:22]
        ax2.annotate(short, (r['umap_x'], r['umap_y']), fontsize=8.5,
                     fontweight='bold', ha='center',
                     xytext=(0, 6), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#fffde7', alpha=0.85))

    for _, r in region[region['category'] == 'icd'].iterrows():
        ax2.annotate(r['name'][:12], (r['umap_x'], r['umap_y']),
                     fontsize=5, alpha=0.45, ha='center', va='top',
                     xytext=(0, -3), textcoords='offset points')

    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    ax2.set_title(cfg['title'], fontsize=10)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIG_DIR / f'umap_zoomed_{zoom_name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: umap_zoomed_{zoom_name}.pdf  .png")

print("\nE1 complete. Outputs in:", OUT_DIR)
