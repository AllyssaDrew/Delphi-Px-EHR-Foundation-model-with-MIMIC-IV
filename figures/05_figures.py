#!/usr/bin/env python3
"""
Phase 7 — Figure reproduction (MIMIC-IV Delphi v6.1)

Fig 2f : AUC by ICD-10 chapter — v5 vs v6.1 grouped comparison
Fig 3b : Disease + phenotype token embedding UMAP (v6.1 wte)
Fig 3c : Attention pattern heatmap (single patient, v6.1 model)
Fig 3d : Model-predicted risk curves — Diabetic vs Control (v6.1 model)
"""

import sys, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPE    = Path(__file__).parent
DELPHI  = PIPE.parent / 'Delphi' / 'Delphi-main'
sys.path.insert(0, str(DELPHI))

CKPT_PATH  = DELPHI / 'checkpoints' / 'mimic_v61_phase_f' / 'ckpt.pt'
DATA_DIR   = PIPE / 'data' / 'mimic_data_v61'
EVAL_V5    = PIPE / 'clinical_phenotyping' / 'Phase_E' / 'results' / 'v5_baseline'
EVAL_V61   = PIPE / 'clinical_phenotyping' / 'Phase_E' / 'results' / 'v61_phase_f'
FIG_DIR    = PIPE / 'figures'
FIG_DIR.mkdir(exist_ok=True)

from model_v4 import DelphiV4, DelphiConfigV4
from utils import get_batch_v4, get_p2i

# ── Shared data ────────────────────────────────────────────────────────────────
labels   = pd.read_csv(DATA_DIR / 'mimic_labels.csv')
meta     = pickle.load(open(DATA_DIR / 'meta_v61.pkl', 'rb'))
df_v5    = pd.read_parquet(EVAL_V5  / 'df_both.parquet')
df_v61   = pd.read_parquet(EVAL_V61 / 'df_both.parquet')

PHENO_STORED = meta['PHENOTYPE_TOKENS']   # name → stored_id (1536–1568)
FIRST_PHENO  = meta['FIRST_PHENO_TOKEN']  # 1536

# Domain colours for phenotype tokens
PHENO_DOMAIN_COLOR = {
    'Cancer':        '#E8735A',
    'Psychiatry':    '#9B59B6',
    'Critical Care': '#E74C3C',
    'Substance Use': '#F39C12',
}
CANCER_TOKENS = {
    'CANCER_STAGE_I','CANCER_STAGE_II','CANCER_STAGE_III','CANCER_STAGE_IV',
    'CANCER_METASTATIC','CANCER_RECURRENT','CHEMO_RECEIVED','CHEMO_PLANNED',
    'RADIOTHERAPY_RECEIVED','RADIOTHERAPY_PLANNED','IMMUNOTHERAPY_RECEIVED',
    'HORMONE_THERAPY_RECEIVED','CANCER_RESECTED','CANCER_STAGE_UNKNOWN',
}
PSYCH_TOKENS = {
    'SUICIDAL_IDEATION_PRESENT','SUICIDAL_IDEATION_DENIED','SUICIDE_ATTEMPT_CURRENT',
    'SUICIDE_ATTEMPT_HISTORY','HOMICIDAL_IDEATION_PRESENT','PSYCHOSIS_ACTIVE',
    'SELF_HARM_PRESENT','PSYCHIATRIC_HOLD',
}
CRIT_TOKENS = {
    'SEPSIS_PRESENT','INTUBATED_DURING_STAY','COMFORT_MEASURES_ONLY','DNR_PRESENT',
    'AKI_PRESENT','DELIRIUM_PRESENT','ICU_ADMISSION',
}
SUBST_TOKENS = {
    'ALCOHOL_WITHDRAWAL_ACTIVE','OPIOID_WITHDRAWAL_ACTIVE',
    'SUBSTANCE_USE_ACTIVE','NALOXONE_ADMINISTERED',
}

def pheno_domain(name):
    if name in CANCER_TOKENS: return 'Cancer'
    if name in PSYCH_TOKENS:  return 'Psychiatry'
    if name in CRIT_TOKENS:   return 'Critical Care'
    return 'Substance Use'


def load_model(device='cpu'):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    conf = DelphiConfigV4(**ckpt['model_args'])
    model = DelphiV4(conf)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, conf


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2f — v5 vs v6.1 AUC by ICD-10 chapter  (grouped horizontal barplot)
# ══════════════════════════════════════════════════════════════════════════════
def fig_chapter_auc():
    C_V5  = '#4878CF'
    C_V61 = '#D65F5F'

    def ch_stats(df):
        g = df.groupby('ICD-10 Chapter (short)')
        med = g['auc'].median()
        se  = g.apply(lambda x: np.sqrt(x['auc_variance_delong'].mean() / max(len(x), 1)),
                      include_groups=False)
        col = g['color'].first()
        return pd.DataFrame({'median_auc': med, 'se': se, 'color': col})

    stats_v5  = ch_stats(df_v5)
    stats_v61 = ch_stats(df_v61)

    # Order chapters by v6.1 median AUC, Death always last
    chapters = stats_v61.index.tolist()
    non_death = sorted([c for c in chapters if c != 'Death'],
                       key=lambda c: stats_v61.loc[c, 'median_auc'])
    order = non_death + ['Death']

    fig, ax = plt.subplots(figsize=(9, 8))
    h = 0.35
    y = np.arange(len(order))

    for i, ch in enumerate(order):
        auc_v5  = stats_v5.loc[ch,  'median_auc'] if ch in stats_v5.index  else np.nan
        auc_v61 = stats_v61.loc[ch, 'median_auc'] if ch in stats_v61.index else np.nan
        se_v5   = stats_v5.loc[ch,  'se']          if ch in stats_v5.index  else 0
        se_v61  = stats_v61.loc[ch, 'se']          if ch in stats_v61.index else 0

        is_death = (ch == 'Death')
        alpha_v5  = 1.0 if is_death else 0.75
        alpha_v61 = 1.0

        ax.barh(i - h/2, auc_v5,  height=h, color=C_V5,  alpha=alpha_v5,
                xerr=1.96*se_v5,  capsize=2,
                error_kw={'ecolor':'#444','lw':0.8})
        ax.barh(i + h/2, auc_v61, height=h, color=C_V61, alpha=alpha_v61,
                xerr=1.96*se_v61, capsize=2,
                error_kw={'ecolor':'#444','lw':0.8})

        delta = auc_v61 - auc_v5
        sign  = '+' if delta >= 0 else ''
        color = '#D65F5F' if delta > 0.005 else ('#4878CF' if delta < -0.005 else '#666')
        ax.text(max(auc_v5, auc_v61) + 0.015, i, f'{sign}{delta:.3f}',
                va='center', fontsize=7, color=color, fontweight='bold' if is_death else 'normal')

    ax.axvline(0.5, color='gray', ls='--', lw=0.8, alpha=0.6, label='Chance')
    ax.set_yticks(y)
    ax.set_yticklabels([c.split('. ', 1)[-1] for c in order], fontsize=8.5)
    ax.set_xlabel('Median AUC', fontsize=11)
    ax.set_title('Delphi v5 vs v6.1 — AUC by ICD-10 Chapter  (MIMIC-IV test)', fontsize=11)
    ax.set_xlim(0.45, 1.10)

    patch_v5  = mpatches.Patch(color=C_V5,  label='v5 (baseline)')
    patch_v61 = mpatches.Patch(color=C_V61, label='v6.1 Phase D-F')
    ax.legend(handles=[patch_v5, patch_v61], fontsize=9, loc='lower right')

    plt.tight_layout()
    out = FIG_DIR / 'fig2f_chapter_auc.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3b — UMAP of token embeddings (ICD + phenotype, v6.1 wte)
# ══════════════════════════════════════════════════════════════════════════════
def fig_embedding_umap():
    model, _ = load_model('cpu')
    wte = model.transformer.wte.weight.detach().numpy()   # (1570, 192)

    # ── ICD / lab tokens from df_v61 ──────────────────────────────────────────
    raw_toks   = df_v61['token'].values.astype(int)       # stored IDs
    model_idxs = raw_toks + 1                             # model IDs (stored+1)
    icd_emb    = wte[model_idxs]
    icd_colors = df_v61['color'].values
    icd_chaps  = df_v61['ICD-10 Chapter (short)'].values

    # ── Phenotype tokens ──────────────────────────────────────────────────────
    pheno_names  = list(PHENO_STORED.keys())
    pheno_mid    = [PHENO_STORED[n] + 1 for n in pheno_names]   # model IDs
    pheno_emb    = wte[pheno_mid]
    pheno_colors = [PHENO_DOMAIN_COLOR[pheno_domain(n)] for n in pheno_names]

    # ── Combined UMAP ─────────────────────────────────────────────────────────
    all_emb = np.vstack([icd_emb, pheno_emb])
    print(f"  Running UMAP on {len(all_emb)} embeddings "
          f"({len(icd_emb)} ICD/lab + {len(pheno_emb)} phenotype) …")
    emb_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       metric='cosine', random_state=42
                       ).fit_transform(StandardScaler().fit_transform(all_emb))

    icd_2d   = emb_2d[:len(icd_emb)]
    pheno_2d = emb_2d[len(icd_emb):]

    chapter_to_color = {ch: col for ch, col in zip(icd_chaps, icd_colors)}

    CHAPTER_ORDER = [
        'I. Infectious Diseases', 'II. Neoplasms', 'III. Blood & Immune Disorders',
        'IV. Metabolic Diseases', 'V. Mental Disorders', 'VI. Nervous System Diseases',
        'VII. Eye Diseases', 'VIII. Ear Diseases', 'IX. Circulatory Diseases',
        'X. Respiratory Diseases', 'XI. Digestive Diseases', 'XII. Skin Diseases',
        'XIII. Musculoskeletal Diseases', 'XIV. Genitourinary Diseases',
        'XV. Pregnancy & Childbirth', 'XVI. Perinatal Conditions',
        'XVII. Congenital Abnormalities',
    ]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(icd_2d[:, 0], icd_2d[:, 1], c=icd_colors,
               s=14, alpha=0.60, linewidths=0, zorder=2)

    # Phenotype tokens: per-domain colours, star marker
    for domain, dcolor in PHENO_DOMAIN_COLOR.items():
        mask = [pheno_domain(n) == domain for n in pheno_names]
        pts  = pheno_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=dcolor, marker='*',
                   s=140, alpha=1.0, linewidths=0.4, edgecolors='white',
                   zorder=5, label=f'Phenotype: {domain}')

    icd_handles = [mpatches.Patch(color=chapter_to_color[ch],
                                  label=ch.split('. ', 1)[-1])
                   for ch in CHAPTER_ORDER if ch in chapter_to_color]
    pheno_handles = [mpatches.Patch(color=c, label=f'★ {d}')
                     for d, c in PHENO_DOMAIN_COLOR.items()]

    leg1 = ax.legend(handles=icd_handles, fontsize=6.5, framealpha=0.85,
                     bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1,
                     title='ICD-10 Chapters', title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=pheno_handles, fontsize=7.5, framealpha=0.9,
              bbox_to_anchor=(1.01, 0.0), loc='lower left', ncol=1,
              title='Phenotype Tokens', title_fontsize=7.5)

    ax.set_title('UMAP of Token Embeddings  (Delphi-Px  —  ICD + Phenotype)', fontsize=12)
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    PDF_DIR = FIG_DIR.parent.parent / 'writing' / 'draft' / 'figure-pdf'
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.06, right=0.72, top=0.95, bottom=0.05)
    out = FIG_DIR / 'fig3b_embedding_umap.png'
    fig.savefig(out, dpi=150)
    fig.savefig(PDF_DIR / 'fig3b_embedding_umap.pdf')
    plt.close()
    print(f"  Saved {out} + PDF")
    del model


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3c — Attention heatmap (last transformer block, v6.1)
# ══════════════════════════════════════════════════════════════════════════════
def fig_attention_heatmap():
    model, conf = load_model('cpu')
    n_summary = conf.n_summary    # 4

    val_data = np.memmap(DATA_DIR / 'val.bin', dtype='uint32', mode='r').reshape(-1, 3)
    p2i      = get_p2i(val_data)

    # Patient with ≥ 40 real events
    rng       = np.random.default_rng(42)
    long_pids = np.where(p2i[:, 1] >= 40)[0]
    pid       = long_pids[rng.integers(0, len(long_pids))]

    BLOCK = 40
    with torch.no_grad():
        x, a, y, b_, ov_x, ov_a = get_batch_v4(
            [pid], val_data, p2i,
            block_size=BLOCK,
            n_summary=n_summary,
            n_overflow=conf.n_overflow,
            device='cpu',
            no_event_token_rate=0,
            select='right',
        )
        _, _, att = model(x, a, overflow_idx=ov_x, overflow_age=ov_a)

    # att: (n_layers, 1, n_heads, T, T)  where T = n_summary + BLOCK
    att_np   = att.detach().numpy()
    att_last = att_np[-1, 0]            # (n_heads, T, T)
    att_avg  = att_last.mean(axis=0)    # (T, T)

    # Skip summary tokens — display only real-event positions
    off      = n_summary
    T_show   = min(BLOCK, 40)
    att_show = att_avg[off:off+T_show, off:off+T_show]
    tok_ids  = x[0, off:off+T_show].numpy()   # model IDs
    ages_d   = a[0, off:off+T_show].numpy()

    def tok_label(model_tok, age_days):
        raw = int(model_tok) - 1
        age_yr = age_days / 365.25
        if model_tok <= 1:   return '[PAD]'
        if raw  == 1:        return f'NoEv\n{age_yr:.0f}y'
        if raw  == 2:        return 'F'
        if raw  == 3:        return 'M'
        if 4 <= raw <= 6:    return f'BMI\n{age_yr:.0f}y'
        if raw  == 7:        return f'ICU\n{age_yr:.0f}y'
        if raw  == 8:        return f'ED\n{age_yr:.0f}y'
        if raw >= FIRST_PHENO + 1:
            name = next((n for n, sid in PHENO_STORED.items() if sid + 1 == model_tok), None)
            return (f'{name[:10]}\n{age_yr:.0f}y' if name else f'P{raw}\n{age_yr:.0f}y')
        if raw < len(labels):
            code = labels.iloc[raw]['name']
            return f'{code}\n{age_yr:.0f}y'
        return f'T{raw}\n{age_yr:.0f}y'

    tick_lbls = [tok_label(t, a_) for t, a_ in zip(tok_ids, ages_d)]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(att_show, aspect='auto', cmap='hot_r', interpolation='nearest')
    ax.set_xticks(np.arange(T_show))
    ax.set_yticks(np.arange(T_show))
    ax.set_xticklabels(tick_lbls, rotation=90, fontsize=5.5)
    ax.set_yticklabels(tick_lbls, fontsize=5.5)
    ax.set_title('Attention Pattern — Last Transformer Block (mean over heads)  [Delphi v6.1]',
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.tight_layout()
    out = FIG_DIR / 'fig3c_attention_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")
    del model


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3d — Model-predicted risk curves: Diabetic vs Control  (v6.1)
# ══════════════════════════════════════════════════════════════════════════════
def fig_risk_curves():
    model, conf = load_model('cpu')
    n_summary = conf.n_summary

    val_data = np.memmap(DATA_DIR / 'val.bin', dtype='uint32', mode='r').reshape(-1, 3)
    p2i      = get_p2i(val_data)

    e11_rows = labels[labels['name'].str.match(r'^E11', na=False)]
    i21_rows = labels[labels['name'].str.match(r'^I2[0-9]', na=False)]

    if len(e11_rows) == 0:
        print("  E11 not found — skipping Fig 3d")
        return

    E11_raw = int(e11_rows['index'].iloc[0])
    if len(i21_rows) > 0:
        TARGET_raw  = int(i21_rows['index'].iloc[0])
        target_name = labels.iloc[TARGET_raw]['name']
    else:
        circ = df_v61[df_v61['ICD-10 Chapter (short)'] == 'IX. Circulatory Diseases']
        TARGET_raw  = int(circ.sort_values('auc', ascending=False).iloc[0]['token'])
        target_name = labels.iloc[TARGET_raw]['name']

    TARGET_model = TARGET_raw + 1
    print(f"  Cohort split on E11 (raw {E11_raw}); target: {target_name} (raw {TARGET_raw})")

    def has_token(pid, raw_tok):
        s, l = p2i[pid]
        return raw_tok in val_data[s:s+l, 2]

    all_pids      = np.arange(len(p2i))
    diabetic_pids = [p for p in all_pids if has_token(p, E11_raw)][:300]
    control_pids  = [p for p in all_pids if not has_token(p, E11_raw)][:300]

    print(f"  Diabetic: {len(diabetic_pids)}, Control: {len(control_pids)}")
    if len(diabetic_pids) < 10:
        print("  Too few diabetic patients — skipping Fig 3d")
        return

    age_bin_edges = np.arange(30, 90, 5)
    BLOCK = conf.block_size

    def risk_profile(pids, target_model_tok):
        bin_vals = {a: [] for a in age_bin_edges}
        for pid in pids:
            with torch.no_grad():
                x, a_, y, b_, ov_x, ov_a = get_batch_v4(
                    [pid], val_data, p2i,
                    block_size=BLOCK,
                    n_summary=n_summary,
                    n_overflow=conf.n_overflow,
                    device='cpu',
                    no_event_token_rate=5,
                    select='right',
                )
                logits, _, _ = model(x, a_, overflow_idx=ov_x, overflow_age=ov_a)

            # Real-event positions only (skip summary)
            log_p  = torch.log_softmax(logits[0], dim=-1)[:, target_model_tok].numpy()
            ages_y = a_[0].numpy() / 365.25
            valid  = (x[0].numpy() > 0)

            for ab in age_bin_edges:
                mask = valid & (ages_y >= ab) & (ages_y < ab + 5)
                if mask.sum() >= 1:
                    bin_vals[ab].append(float(log_p[mask].mean()))

        ages_out, risk_out, err_out = [], [], []
        for ab in age_bin_edges:
            vals = bin_vals[ab]
            if len(vals) >= 5:
                ages_out.append(ab + 2.5)
                risk_out.append(np.mean(vals))
                err_out.append(np.std(vals) / np.sqrt(len(vals)))
        return np.array(ages_out), np.array(risk_out), np.array(err_out)

    print("  Computing risk profiles …")
    a_d, r_d, e_d = risk_profile(diabetic_pids, TARGET_model)
    a_c, r_c, e_c = risk_profile(control_pids,  TARGET_model)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_d, r_d, 'o-', color='#d62728', label='Diabetic  (E11+)', lw=1.8)
    ax.fill_between(a_d, r_d - 1.96*e_d, r_d + 1.96*e_d, alpha=0.18, color='#d62728')
    ax.plot(a_c, r_c, 's--', color='#1f77b4', label='Control   (E11−)', lw=1.8)
    ax.fill_between(a_c, r_c - 1.96*e_c, r_c + 1.96*e_c, alpha=0.18, color='#1f77b4')

    ax.set_xlabel('Age (years)', fontsize=11)
    ax.set_ylabel(f'Mean log P(next = {target_name})', fontsize=10)
    ax.set_title(f'Model-predicted risk of {target_name}\n'
                 f'Diabetic (E11+) vs Control  (Delphi v6.1 — MIMIC-IV validation)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig3d_risk_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")
    del model


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.chdir(PIPE)
    print("=== Phase 7: Figure Reproduction (Delphi v6.1) ===\n")

    print("[1/4] Fig 2f — v5 vs v6.1 Chapter AUC comparison")
    fig_chapter_auc()

    print("[2/4] Fig 3b — Embedding UMAP (ICD + phenotype tokens)")
    fig_embedding_umap()

    print("[3/4] Fig 3c — Attention heatmap")
    fig_attention_heatmap()

    print("[4/4] Fig 3d — Predicted risk curves (Diabetic vs Control)")
    fig_risk_curves()

    print(f"\n=== Done. All figures saved to {FIG_DIR} ===")
