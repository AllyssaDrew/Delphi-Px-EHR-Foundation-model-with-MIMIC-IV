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
E4: Temporal Decay Curves
For diseases C25, C50, A41, I21, F32: find patients who have the diagnosis,
compute leave-one-out attribution of that token on Death logit at multiple
prediction timepoints after the diagnosis. 2 forward passes per point (full
vs masked), fast enough for 600+ points per disease.
Aggregate into time bins, plot decay curves with Risk Table and Nelson-Aalen.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PIPE   = PIPELINE_DIR
DELPHI = DELPHI_DIR
sys.path.insert(0, str(DELPHI))

CKPT       = DELPHI / 'checkpoints/mimic_v5_phase2b/ckpt.pt'
VAL_BIN    = PIPE / 'data/mimic_data_v5/val.bin'
META_PKL   = PIPE / 'data/mimic_data_v5/meta.pkl'
LABELS_CSV = PIPE / 'data/mimic_data_v5/mimic_labels.csv'
OUT_DIR    = PIPE / 'explainability/E4_temporal_decay'

BLOCK_SIZE  = 128
N_SUMMARY   = 4
N_OVERFLOW  = 64
MASK_TIME   = -10000.
NO_EVENT_MS = 2

ICD_START_STORED = 9
ICD_END_STORED   = 1509
DEATH_STORED     = 1510
DEATH_MODEL      = DEATH_STORED + 1  # 1511

MIN_N_PER_BIN = 50
BOOTSTRAP_N   = 200
RANDOM_SEED   = 42
MAX_PER_BIN   = 200  # up-sampled since each point is now cheap

MONTH_EDGES_DAYS = [0, 30, 90, 180, 365, 730, 1825]
BIN_LABELS = ['0-1m', '1-3m', '3-6m', '6-12m', '12-24m', '24-60m']
N_BINS = len(BIN_LABELS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

print("Loading model...", flush=True)
from model_v4 import DelphiV4, DelphiConfigV4
ckpt_data = torch.load(str(CKPT), map_location='cpu', weights_only=False)
config = DelphiConfigV4(**ckpt_data['model_args'])
model  = DelphiV4(config)
model.load_state_dict(ckpt_data['model'])
model.eval()
model.to(device)

print("Loading val.bin...", flush=True)
data = np.memmap(str(VAL_BIN), dtype=np.int32, mode='r').reshape(-1, 3)
from utils import get_p2i
p2i = get_p2i(data)
n_patients_total = len(p2i)
print(f"  {n_patients_total} patients", flush=True)

with open(META_PKL, 'rb') as f:
    meta = pickle.load(f)
labels = pd.read_csv(LABELS_CSV)

def name_to_stored_ids(prefix):
    return [sid for sid in range(ICD_START_STORED, ICD_END_STORED + 1)
            if pd.notna(labels.iloc[sid]['name']) and str(labels.iloc[sid]['name']).startswith(prefix)]

DISEASES = [
    {'name': 'C25', 'prefix': 'C25', 'color': '#d62728', 'label': 'C25 (Pancreatic Ca)'},
    {'name': 'C50', 'prefix': 'C50', 'color': '#ff7f0e', 'label': 'C50 (Breast Ca)'},
    {'name': 'A41', 'prefix': 'A41', 'color': '#2ca02c', 'label': 'A41 (Sepsis)'},
    {'name': 'I21', 'prefix': 'I21', 'color': '#1f77b4', 'label': 'I21 (MI)'},
    {'name': 'F32', 'prefix': 'F32', 'color': '#9467bd', 'label': 'F32 (Depression)'},
]
for d in DISEASES:
    d['stored_ids'] = set(name_to_stored_ids(d['prefix']))
    d['ms_ids']     = {s + 1 for s in d['stored_ids']}
    print(f"  {d['name']}: {len(d['stored_ids'])} stored IDs", flush=True)

# ── Sequence extraction ────────────────────────────────────────────────────────
def get_sequence_at_pos(pid, pred_pos):
    start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
    toks = data[start:start + length, 2].astype(np.int64)
    ages = data[start:start + length, 1].astype(np.float32)

    toks = toks[:pred_pos + 1]
    ages = ages[:pred_pos + 1]
    n = len(toks)
    if n < 2:
        return None

    w_start = max(0, n - BLOCK_SIZE)
    win_s = toks[w_start:]
    win_a = ages[w_start:]

    pad = BLOCK_SIZE - len(win_s)
    if pad > 0:
        win_s = np.concatenate([np.full(pad, -1, dtype=np.int64), win_s])
        win_a = np.concatenate([np.full(pad, MASK_TIME, dtype=np.float32), win_a])

    win_ms = (win_s + 1).clip(min=0)

    ov_end   = w_start
    ov_start = max(0, ov_end - N_OVERFLOW)
    ov_s     = toks[ov_start:ov_end]
    ov_a_raw = ages[ov_start:ov_end]

    ov_ms  = np.zeros(N_OVERFLOW, dtype=np.int64)
    ov_age = np.full(N_OVERFLOW, MASK_TIME, dtype=np.float32)
    if len(ov_s) > 0:
        n_act = len(ov_s)
        ov_ms[N_OVERFLOW - n_act:]  = ov_s + 1
        ov_age[N_OVERFLOW - n_act:] = ov_a_raw

    has_ov = bool(np.any(ov_ms > 0))
    if has_ov:
        valid = ov_age[ov_age > MASK_TIME + 1]
        last_ov_age = float(valid[-1]) if len(valid) > 0 else 0.

    return {
        'window_ms':    win_ms,
        'window_ages':  win_a,
        'overflow_ms':  ov_ms,
        'overflow_ages': ov_age,
        'has_ov':       has_ov,
        'last_ov_age':  last_ov_age if has_ov else 0.,
    }

# ── Leave-one-out attribution (2 forward passes) ──────────────────────────────
def run_model(win_ms, win_ages, ov_ms, ov_age, has_ov, last_ov_age):
    """Single forward pass, returns Death logit scalar."""
    bs = 1
    if has_ov:
        s_tok = np.full((bs, N_SUMMARY), NO_EVENT_MS, dtype=np.int64)
        s_age = np.full((bs, N_SUMMARY), last_ov_age, dtype=np.float32)
    else:
        s_tok = np.zeros((bs, N_SUMMARY), dtype=np.int64)
        s_age = np.full((bs, N_SUMMARY), MASK_TIME, dtype=np.float32)

    full_toks = np.concatenate([s_tok, win_ms.reshape(1, -1)], axis=1)
    full_ages = np.concatenate([s_age, win_ages.reshape(1, -1)], axis=1)

    x_t   = torch.tensor(full_toks, dtype=torch.long,  device=device)
    a_t   = torch.tensor(full_ages, dtype=torch.float, device=device)
    ov_xt = torch.tensor(ov_ms,  dtype=torch.long,  device=device).unsqueeze(0)
    ov_at = torch.tensor(ov_age, dtype=torch.float, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, *_ = model(x_t, a_t, overflow_idx=ov_xt, overflow_age=ov_at)

    return float(logits[0, -1, DEATH_MODEL].cpu())

def loo_attribution(seq, disease_ms_ids):
    """
    Leave-one-out: mean_{positions p of disease token} [f(x) - f(x with pos p masked)].
    Returns None if disease token not in window.
    """
    win_ms   = seq['window_ms'].copy()
    win_ages = seq['window_ages']
    ov_ms    = seq['overflow_ms']
    ov_age   = seq['overflow_ages']
    has_ov   = seq['has_ov']
    last_ov_age = seq['last_ov_age']

    pos_list = [p for p in range(BLOCK_SIZE) if int(win_ms[p]) in disease_ms_ids]
    if not pos_list:
        return None

    logit_full = run_model(win_ms, win_ages, ov_ms, ov_age, has_ov, last_ov_age)

    diffs = []
    for p in pos_list:
        masked = win_ms.copy()
        masked[p] = NO_EVENT_MS
        logit_masked = run_model(masked, win_ages, ov_ms, ov_age, has_ov, last_ov_age)
        diffs.append(logit_full - logit_masked)

    return float(np.mean(diffs))

# ── Scan patients ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)
all_results = []

for disease in DISEASES:
    print(f"\n{'='*60}", flush=True)
    print(f"Disease: {disease['name']}", flush=True)

    dis_stored_ids = disease['stored_ids']
    dis_ms_ids     = disease['ms_ids']

    records = []
    for pid in range(n_patients_total):
        start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
        toks = data[start:start + length, 2].astype(np.int64)
        ages = data[start:start + length, 1].astype(np.float32)

        occ_positions = [i for i, t in enumerate(toks) if t in dis_stored_ids]
        if not occ_positions:
            continue

        diag_pos = occ_positions[0]
        diag_age = float(ages[diag_pos])
        if diag_age <= 0:
            continue

        for pred_pos in range(diag_pos + 1, length):
            tok_ms = int(toks[pred_pos]) + 1
            is_icd   = (ICD_START_STORED + 1 <= tok_ms <= ICD_END_STORED + 1)
            is_death = (tok_ms == DEATH_MODEL)
            if not (is_icd or is_death):
                continue

            pred_age = float(ages[pred_pos])
            if pred_age <= 0:
                continue

            time_delta_days = pred_age - diag_age
            if time_delta_days < 0:
                continue

            bin_idx = -1
            for bi in range(N_BINS):
                if MONTH_EDGES_DAYS[bi] <= time_delta_days < MONTH_EDGES_DAYS[bi + 1]:
                    bin_idx = bi
                    break
            if bin_idx < 0:
                continue

            records.append({'pid': pid, 'pred_pos': pred_pos, 'bin_idx': bin_idx,
                             'time_delta_days': time_delta_days})

    print(f"  {len(records)} prediction points found", flush=True)
    if not records:
        continue

    df_rec = pd.DataFrame(records)
    sampled_rows = []
    for bi in range(N_BINS):
        sub = df_rec[df_rec['bin_idx'] == bi]
        if len(sub) > MAX_PER_BIN:
            sub = sub.sample(n=MAX_PER_BIN, random_state=RANDOM_SEED)
        sampled_rows.append(sub)
    df_sample = pd.concat(sampled_rows, ignore_index=True)
    print(f"  After sampling: {len(df_sample)} points", flush=True)

    shap_vals_by_bin = {bi: [] for bi in range(N_BINS)}
    t0 = time.time()

    for row_i, (_, row) in enumerate(df_sample.iterrows()):
        pid      = int(row['pid'])
        pred_pos = int(row['pred_pos'])
        bin_idx  = int(row['bin_idx'])

        seq = get_sequence_at_pos(pid, pred_pos)
        if seq is None:
            continue

        sv = loo_attribution(seq, dis_ms_ids)
        if sv is None:
            continue

        shap_vals_by_bin[bin_idx].append(sv)

        if (row_i + 1) % 50 == 0:
            print(f"    {row_i+1}/{len(df_sample)}  {time.time()-t0:.0f}s", flush=True)

    bin_means, bin_ci_lo, bin_ci_hi, bin_ns = [], [], [], []
    for bi in range(N_BINS):
        vals = np.array(shap_vals_by_bin[bi])
        n = len(vals)
        bin_ns.append(n)
        if n == 0:
            bin_means.append(np.nan); bin_ci_lo.append(np.nan); bin_ci_hi.append(np.nan)
        else:
            m = float(np.mean(vals))
            bin_means.append(m)
            if n >= 5:
                bs_means = [float(np.mean(rng.choice(vals, size=n, replace=True)))
                            for _ in range(BOOTSTRAP_N)]
                bin_ci_lo.append(float(np.percentile(bs_means, 2.5)))
                bin_ci_hi.append(float(np.percentile(bs_means, 97.5)))
            else:
                bin_ci_lo.append(m); bin_ci_hi.append(m)

    for bi in range(N_BINS):
        all_results.append({
            'disease':   disease['name'],
            'label':     disease['label'],
            'color':     disease['color'],
            'bin_idx':   bi,
            'bin_label': BIN_LABELS[bi],
            'mean_shap': bin_means[bi],
            'ci_lo':     bin_ci_lo[bi],
            'ci_hi':     bin_ci_hi[bi],
            'n':         bin_ns[bi],
        })
    print(f"  N per bin: {bin_ns}", flush=True)
    print(f"  Means: {[f'{m:.3f}' if not np.isnan(m) else 'nan' for m in bin_means]}", flush=True)

# ── Nelson-Aalen ──────────────────────────────────────────────────────────────
na_results = []
for disease in DISEASES:
    dis_stored_ids = disease['stored_ids']
    bin_events = np.zeros(N_BINS)
    bin_atrisk = np.zeros(N_BINS)

    for pid in range(n_patients_total):
        start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
        toks = data[start:start + length, 2].astype(np.int64)
        ages = data[start:start + length, 1].astype(np.float32)

        occ = [i for i, t in enumerate(toks) if t in dis_stored_ids]
        if not occ:
            continue
        diag_age = float(ages[occ[0]])
        if diag_age <= 0:
            continue

        death_pos = [i for i in range(occ[0] + 1, length) if int(toks[i]) == DEATH_STORED]
        if death_pos:
            time_to = float(ages[death_pos[0]]) - diag_age
            event = True
        else:
            time_to = max(0., float(ages[-1]) - diag_age)
            event = False

        for bi in range(N_BINS):
            lo, hi = MONTH_EDGES_DAYS[bi], MONTH_EDGES_DAYS[bi + 1]
            if time_to >= lo:
                bin_atrisk[bi] += 1
            if event and lo <= time_to < hi:
                bin_events[bi] += 1

    cum_h = 0.
    for bi in range(N_BINS):
        if bin_atrisk[bi] > 0:
            cum_h += bin_events[bi] / bin_atrisk[bi]
        na_results.append({'disease': disease['name'], 'bin_idx': bi,
                            'bin_label': BIN_LABELS[bi], 'cum_hazard': cum_h,
                            'at_risk': int(bin_atrisk[bi])})

df_results = pd.DataFrame(all_results)
df_na      = pd.DataFrame(na_results)
df_results.to_csv(OUT_DIR / 'decay_curves.csv', index=False)
df_na.to_csv(OUT_DIR / 'nelson_aalen_reference.csv', index=False)
print("CSVs saved.", flush=True)

# ── Main figure: decay + Risk Table ──────────────────────────────────────────
x = np.arange(N_BINS)
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_main  = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])

for disease in DISEASES:
    dname = disease['name']
    sub   = df_results[df_results['disease'] == dname].sort_values('bin_idx')
    if sub.empty:
        continue
    means = sub['mean_shap'].values
    ci_lo = sub['ci_lo'].values
    ci_hi = sub['ci_hi'].values
    ns    = sub['n'].values
    color = disease['color']

    cutoff = next((bi for bi in range(N_BINS) if ns[bi] < MIN_N_PER_BIN), N_BINS)
    valid_x = x[:cutoff][~np.isnan(means[:cutoff])]
    valid_m = means[:cutoff][~np.isnan(means[:cutoff])]
    valid_lo = ci_lo[:cutoff][~np.isnan(ci_lo[:cutoff])]
    valid_hi = ci_hi[:cutoff][~np.isnan(ci_hi[:cutoff])]

    if len(valid_x) == 0:
        continue
    ax_main.plot(valid_x, valid_m, 'o-', color=color, label=disease['label'], linewidth=2, markersize=5)
    if len(valid_x) > 1:
        ax_main.fill_between(valid_x, valid_lo, valid_hi, alpha=0.2, color=color)
    if cutoff < N_BINS:
        ax_main.axvline(cutoff - 0.5, color=color, linestyle='--', alpha=0.4, linewidth=1)

ax_main.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax_main.set_ylabel('LOO attribution on Death logit (log-rate)', fontsize=11)
ax_main.set_title('Disease Diagnosis → Death Attribution: Temporal Decay', fontsize=12, fontweight='bold')
ax_main.set_xticks(x)
ax_main.set_xticklabels(['' for _ in BIN_LABELS])
ax_main.legend(fontsize=9, loc='upper right')
ax_main.grid(True, alpha=0.3)
ax_main.axhline(0, color='black', linewidth=0.8)

ax_table.axis('off')
cell_text = []
for disease in DISEASES:
    sub = df_results[df_results['disease'] == disease['name']].sort_values('bin_idx')
    if sub.empty:
        cell_text.append(['—'] * N_BINS); continue
    row = []
    for bi in range(N_BINS):
        s = sub[sub['bin_idx'] == bi]
        n = int(s['n'].values[0]) if not s.empty else 0
        row.append(f'{n}*' if 0 < n < MIN_N_PER_BIN else ('—' if n == 0 else str(n)))
    cell_text.append(row)

tbl = ax_table.table(cellText=cell_text, rowLabels=[d['label'] for d in DISEASES],
                     colLabels=BIN_LABELS, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.4)
ax_table.set_title('Risk Table (N; * = N<50, curve truncated)', fontsize=9, pad=2)

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'figures' / 'temporal_decay_with_risktable.pdf'), bbox_inches='tight', dpi=150)
plt.close()
print("Decay figure saved.", flush=True)

# ── Nelson-Aalen comparison figure ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for disease in DISEASES:
    dname = disease['name']; color = disease['color']; label = disease['label']
    sub = df_results[df_results['disease'] == dname].sort_values('bin_idx')
    if not sub.empty:
        ns = sub['n'].values; means = sub['mean_shap'].values
        cutoff = next((bi for bi in range(N_BINS) if ns[bi] < MIN_N_PER_BIN), N_BINS)
        vx = x[:cutoff][~np.isnan(means[:cutoff])]
        vm = means[:cutoff][~np.isnan(means[:cutoff])]
        if len(vx) > 0:
            axes[0].plot(vx, vm, 'o-', color=color, label=label, linewidth=2)
    sub_na = df_na[df_na['disease'] == dname].sort_values('bin_idx')
    if not sub_na.empty:
        axes[1].plot(x, sub_na['cum_hazard'].values, 's--', color=color, label=label, linewidth=2)

for ax, title, ylabel in zip(axes,
    ['LOO Attribution on Death logit', 'Nelson-Aalen Cumulative Hazard'],
    ['Mean LOO diff (log-rate)', 'Cumulative hazard']):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
axes[0].axhline(0, color='black', linewidth=0.8)

plt.suptitle('LOO Decay vs Nelson-Aalen Reference', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'figures' / 'shap_vs_nelson_aalen.pdf'), bbox_inches='tight', dpi=150)
plt.close()
print("Nelson-Aalen comparison figure saved.", flush=True)
print("\nE4 complete.", flush=True)
