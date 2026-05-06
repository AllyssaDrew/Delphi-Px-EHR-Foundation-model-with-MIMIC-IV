"""
SHAP + AUC visualisations for Delphi v6.1.

Figures produced:
  fig1_auc_chapter.png      — v5 vs v6.1 chapter-level AUC comparison
  fig2_shap_top_tokens.png  — Top-30 tokens by |SHAP| for Death (type-coloured)
  fig3_shap_phenotype.png   — All 33 phenotype tokens, mean SHAP for Death
  fig4_shap_scatter.png     — Phenotype tokens: frequency vs mean SHAP
  fig5_shap_patient.png     — Waterfall plot for one representative patient
"""
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


import pickle, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = PIPELINE_DIR / 'clinical_phenotyping/Phase_E/results'
SHAP_DIR = PIPELINE_DIR / 'clinical_phenotyping/Phase_E/shap_results'
OUT_DIR  = PIPELINE_DIR / 'clinical_phenotyping/Phase_E/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR  = OUT_DIR.parent / 'figure_pdf'
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour scheme ─────────────────────────────────────────────────────────────
C_V5    = '#4878CF'   # blue
C_V61   = '#D65F5F'   # red
C_ICD   = '#7B9EC9'   # light blue
C_PHENO = '#E8735A'   # orange-red
C_DEATH = '#2C4770'   # dark blue (Death token)
C_PROC  = '#AAAAAA'   # grey (procedure codes)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
v5  = pd.read_parquet(RESULTS / 'v5_baseline/df_both.parquet')
v61 = pd.read_parquet(RESULTS / 'v61_phase_f/df_both.parquet')

shap_df = pd.read_csv(SHAP_DIR / 'shap_v61_token_summary.csv')

with open(SHAP_DIR / 'shap_v61_raw.pkl', 'rb') as f:
    raw_shap = pickle.load(f)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Chapter-level AUC: v5 vs v6.1
# ═══════════════════════════════════════════════════════════════════════════════
def chapter_auc(df):
    return df.groupby('ICD-10 Chapter (short)')['auc'].mean()

v5_ch  = chapter_auc(v5)
v61_ch = chapter_auc(v61)
chapters = v5_ch.index.tolist()

# Sort by v5 AUC
order = v5_ch.sort_values().index.tolist()
v5_ord  = v5_ch[order].values
v61_ord = v61_ch[order].values
delta   = v61_ord - v5_ord

fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(order))
h = 0.35

bars_v5  = ax.barh(y - h/2, v5_ord,  h, label='Delphi-Base (baseline)',       color=C_V5,  alpha=0.85)
bars_v61 = ax.barh(y + h/2, v61_ord, h, label='Delphi-Px (phenotype tokens)', color=C_V61, alpha=0.85)

# Annotate delta
for i, (yp, d) in enumerate(zip(y, delta)):
    xmax = max(v5_ord[i], v61_ord[i])
    color = C_V61 if d > 0 else '#888888'
    ax.text(xmax + 0.003, yp + h/2, f'{d:+.3f}', va='center', fontsize=7.5,
            color=color, fontweight='bold' if abs(d) > 0.02 else 'normal')

# Highlight Death
death_idx = order.index('Death')
ax.axhspan(death_idx - 0.5, death_idx + 0.5, color='#FFF3CD', alpha=0.6, zorder=0)
ax.text(0.62, death_idx, '← Death  Δ=+0.041', va='center', fontsize=8.5,
        color='#9B5000', fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels([o.replace('XVI. ', 'XVI. ').replace('XVII. ', 'XVII. ')
                    for o in order], fontsize=8.5)
ax.set_xlabel('Mean AUC (chapter level)', fontsize=10)
ax.set_title('Delphi-Base vs Delphi-Px: ICD Chapter AUC\n'
             '(Delphi-Px = backbone fine-tuned with 33 discrete clinical phenotype tokens)',
             fontsize=10.5, pad=10)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0.60, 1.02)
ax.axvline(v5['auc'].mean(),  color=C_V5,  linestyle='--', linewidth=0.8, alpha=0.6)
ax.axvline(v61['auc'].mean(), color=C_V61, linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(0.04, 0.97, f'Delphi-Base: {v5["auc"].mean():.4f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=C_V5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.92, ec=C_V5, lw=0.8))
ax.text(0.96, 0.97, f'Delphi-Px: {v61["auc"].mean():.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=C_V61,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.92, ec=C_V61, lw=0.8))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_auc_chapter.png', dpi=180, bbox_inches='tight')
plt.savefig(PDF_DIR / 'fig1_auc_chapter.pdf', bbox_inches='tight')
plt.close()
print("Saved fig1_auc_chapter.png/pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Top-30 tokens by |SHAP| for Death
# ═══════════════════════════════════════════════════════════════════════════════
# Exclude Death token itself and rare (n<5) single-occurrence procedure codes
# with alpha-only names (procedure codes have short uppercase names)
def is_proc_code(name):
    return bool(re.fullmatch(r'[0-9A-Z]{2,4}', str(name)))

plot_df = shap_df[
    (shap_df['name'] != 'Death') &
    ~shap_df['name'].apply(is_proc_code)
].copy()

top30 = plot_df.nlargest(30, 'mean_abs_shap').sort_values('mean_abs_shap')

def token_color(row):
    if row['is_phenotype']:   return C_PHENO
    if row['name'] == 'Death': return C_DEATH
    return C_ICD

colors = [token_color(r) for _, r in top30.iterrows()]

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(range(len(top30)), top30['mean_shap'], color=colors, alpha=0.88)

# Overlay error bars as |mean_abs_shap - |mean_shap|| (directional spread)
for i, (_, row) in enumerate(top30.iterrows()):
    ax.scatter(row['mean_abs_shap'], i, marker='|', s=60,
               color='#333333', linewidth=1.2, zorder=5)

ax.set_yticks(range(len(top30)))
ax.set_yticklabels(
    [f"{r['name']}  (n={r['n_occurrences']:,})"
     for _, r in top30.iterrows()],
    fontsize=8
)
ax.axvline(0, color='black', linewidth=0.7)
ax.set_xlabel('Mean SHAP value for Death logit', fontsize=10)
ax.set_title('Top 30 tokens by |SHAP| — Death prediction\n'
             '(Delphi-Px, 500 val patients with ≥1 phenotype token)',
             fontsize=10.5, pad=10)

legend_handles = [
    mpatches.Patch(color=C_PHENO, label='Phenotype token (new in Delphi-Px)'),
    mpatches.Patch(color=C_ICD,   label='ICD / demographic token'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
ax.set_xlim(-0.2, max(top30['mean_abs_shap']) * 1.15)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_shap_top_tokens.png', dpi=180, bbox_inches='tight')
plt.savefig(PDF_DIR / 'fig2_shap_top_tokens.pdf', bbox_inches='tight')
plt.close()
print("Saved fig2_shap_top_tokens.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — All 33 phenotype tokens, mean SHAP for Death
# ═══════════════════════════════════════════════════════════════════════════════
pheno_df = shap_df[shap_df['is_phenotype']].copy()
pheno_df = pheno_df.sort_values('mean_shap', ascending=True)

# Group labels by clinical domain
domain_colors = {
    'Cancer': '#E07B54',
    'Psychiatry': '#9B59B6',
    'Critical care': '#C0392B',
    'Substance use': '#2ECC71',
}
def domain(name):
    cancer = {'CANCER_STAGE_I','CANCER_STAGE_II','CANCER_STAGE_III','CANCER_STAGE_IV',
               'CANCER_METASTATIC','CANCER_RECURRENT','CHEMO_RECEIVED','CHEMO_PLANNED',
               'RADIOTHERAPY_RECEIVED','RADIOTHERAPY_PLANNED','IMMUNOTHERAPY_RECEIVED',
               'HORMONE_THERAPY_RECEIVED','CANCER_RESECTED','CANCER_STAGE_UNKNOWN'}
    psych  = {'SUICIDAL_IDEATION_PRESENT','SUICIDAL_IDEATION_DENIED','SUICIDE_ATTEMPT_CURRENT',
               'SUICIDE_ATTEMPT_HISTORY','HOMICIDAL_IDEATION_PRESENT','PSYCHOSIS_ACTIVE',
               'SELF_HARM_PRESENT','PSYCHIATRIC_HOLD'}
    crit   = {'SEPSIS_PRESENT','INTUBATED_DURING_STAY','COMFORT_MEASURES_ONLY','DNR_PRESENT',
               'AKI_PRESENT','DELIRIUM_PRESENT','ICU_ADMISSION'}
    subst  = {'ALCOHOL_WITHDRAWAL_ACTIVE','OPIOID_WITHDRAWAL_ACTIVE','SUBSTANCE_USE_ACTIVE',
               'NALOXONE_ADMINISTERED'}
    if name in cancer: return 'Cancer', domain_colors['Cancer']
    if name in psych:  return 'Psychiatry', domain_colors['Psychiatry']
    if name in crit:   return 'Critical care', domain_colors['Critical care']
    if name in subst:  return 'Substance use', domain_colors['Substance use']
    return 'Other', '#888888'

bar_colors = [domain(r['name'])[1] for _, r in pheno_df.iterrows()]

fig, ax = plt.subplots(figsize=(8, 8))
bars = ax.barh(range(len(pheno_df)), pheno_df['mean_shap'],
               color=bar_colors, alpha=0.88, edgecolor='white')

# Size marker proportional to n_occurrences
max_n = pheno_df['n_occurrences'].max()
for i, (_, row) in enumerate(pheno_df.iterrows()):
    size = 30 + 120 * (row['n_occurrences'] / max_n)
    ax.scatter(row['mean_shap'], i, s=size, color='white',
               edgecolors='#333333', linewidths=0.8, zorder=5)
    ax.text(row['mean_shap'] + 0.01, i, f"n={row['n_occurrences']}",
            va='center', fontsize=7, color='#444444')

ax.set_yticks(range(len(pheno_df)))
ax.set_yticklabels(pheno_df['name'].tolist(), fontsize=8.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean SHAP value for Death logit', fontsize=10)
ax.set_title('All 33 phenotype tokens — contribution to Death prediction\n'
             '(positive = increases Death risk estimate)',
             fontsize=10.5, pad=10)

legend_handles = [mpatches.Patch(color=c, label=d)
                  for d, c in domain_colors.items()]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
          title='Clinical domain', title_fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_shap_phenotype.png', dpi=180, bbox_inches='tight')
plt.savefig(PDF_DIR / 'fig3_shap_phenotype.pdf', bbox_inches='tight')
plt.close()
print("Saved fig3_shap_phenotype.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Phenotype tokens: frequency vs mean SHAP (bubble chart)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

for _, row in pheno_df.iterrows():
    d, c = domain(row['name'])
    ax.scatter(row['n_occurrences'], row['mean_shap'],
               s=80, color=c, alpha=0.85, edgecolors='white', linewidths=0.5, zorder=5)
    # Label high-SHAP or high-frequency tokens
    if row['mean_shap'] > 0.25 or row['n_occurrences'] > 120:
        ax.annotate(row['name'].replace('_', '\n'),
                    (row['n_occurrences'], row['mean_shap']),
                    textcoords='offset points', xytext=(6, 2),
                    fontsize=6.5, color='#333333')

ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
ax.set_xlabel('Frequency in val set (n occurrences)', fontsize=10)
ax.set_ylabel('Mean SHAP for Death logit', fontsize=10)
ax.set_title('Phenotype token frequency vs. Death prediction contribution\n'
             '(Delphi-Px)', fontsize=10.5, pad=10)

legend_handles = [mpatches.Patch(color=c, label=d)
                  for d, c in domain_colors.items()]
ax.legend(handles=legend_handles, fontsize=9, title='Clinical domain',
          title_fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_shap_scatter.png', dpi=180, bbox_inches='tight')
plt.savefig(PDF_DIR / 'fig4_shap_scatter.pdf', bbox_inches='tight')
plt.close()
print("Saved fig4_shap_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Waterfall for one representative patient (most phenotype tokens)
# ═══════════════════════════════════════════════════════════════════════════════
# Pick patient with most phenotype tokens and highest death base value
best = max(raw_shap,
           key=lambda r: (sum(1 for t in r['token_ids'] if t >= 1537),
                          float(r['base_values'][0])))

tok_ids_p = best['token_ids']
sv_p      = best['shap_values'][:, 0]   # Death target
base_p    = float(best['base_values'][0])

# Load token names
from pathlib import Path
data_dir  = PIPELINE_DIR / 'data/mimic_data_v61'
labels_df2 = pd.read_csv(data_dir / 'mimic_labels.csv')
id2name2   = {int(r['index']) + 1: r['name']
              for _, r in labels_df2.iterrows() if pd.notna(r['name']) and r['name']}
import pickle as pkl2
with open(data_dir / 'meta_v61.pkl', 'rb') as ff:
    meta2 = pkl2.load(ff)
for name, sid in meta2['PHENOTYPE_TOKENS'].items():
    id2name2[sid + 1] = name
id2name2[1511] = 'Death'

tok_names_p = [id2name2.get(t, f'tok_{t}') for t in tok_ids_p]

# Keep top-15 by |SHAP| for readability
top_idx = np.argsort(np.abs(sv_p))[-15:][::-1]
top_names = [tok_names_p[i] for i in top_idx]
top_sv    = sv_p[top_idx]
# Sort by SHAP value for waterfall
order_wf = np.argsort(top_sv)
names_wf  = [top_names[i] for i in order_wf]
sv_wf     = top_sv[order_wf]

running = base_p
xs = []
for v in sv_wf:
    xs.append(running)
    running += v
xs = np.array(xs)

fig, ax = plt.subplots(figsize=(9, 6))
colors_wf = [C_PHENO if n in meta2['PHENOTYPE_TOKENS'] else
             (C_DEATH if n == 'Death' else C_ICD)
             for n in names_wf]
bars = ax.barh(range(len(names_wf)), sv_wf, left=xs,
               color=colors_wf, alpha=0.85, edgecolor='white')

ax.axvline(base_p,    color='grey',  linestyle='--', linewidth=1, label=f'Base = {base_p:.2f}')
ax.axvline(running,   color='black', linestyle='-',  linewidth=1.2, label=f'Pred = {running:.2f}')

ax.set_yticks(range(len(names_wf)))
ax.set_yticklabels(names_wf, fontsize=8.5)
ax.set_xlabel('Death logit (cumulative SHAP)', fontsize=10)
ax.set_title('SHAP waterfall — representative patient\n'
             '(top 15 tokens by |SHAP| for Death prediction)',
             fontsize=10.5, pad=10)

legend_handles = [
    mpatches.Patch(color=C_PHENO, label='Phenotype token'),
    mpatches.Patch(color=C_ICD,   label='ICD / demographic'),
    mpatches.Patch(color=C_DEATH, label='Death token'),
    plt.Line2D([0],[0], color='grey', linestyle='--', label=f'Base={base_p:.2f}'),
    plt.Line2D([0],[0], color='black', linestyle='-', label=f'Final={running:.2f}'),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc='lower right')

n_pheno_p = sum(1 for t in tok_ids_p if t >= 1537)
ax.set_title(f'SHAP waterfall — representative patient '
             f'({n_pheno_p} phenotype tokens in history)\n'
             f'Top 15 contributors to Death logit',
             fontsize=10.5, pad=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_shap_patient.png', dpi=180, bbox_inches='tight')
plt.savefig(PDF_DIR / 'fig5_shap_patient.pdf', bbox_inches='tight')
plt.close()
print("Saved fig5_shap_patient.png")

print(f"\nAll figures saved to {OUT_DIR}")
