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
E2: SHAP Contrastive Case Pairs
Selects matched patient pairs for 3 disease groups, computes per-token
SHAP values via PermutationExplainer (numpy array masking), generates
paired waterfall plots.

Note: shap.maskers.Text is NOT used because SHAP 0.51's Text masker
concatenates mask_token directly to adjacent token text, producing
out-of-range IDs.  Instead the model_fn takes (batch, BLOCK_SIZE) numpy
arrays and the background is an all-NO_EVENT array.
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPE   = PIPELINE_DIR
DELPHI = DELPHI_DIR
sys.path.insert(0, str(DELPHI))

CKPT       = DELPHI / 'checkpoints/mimic_v5_phase2b/ckpt.pt'
VAL_BIN    = PIPE / 'data/mimic_data_v5/val.bin'
META_PKL   = PIPE / 'data/mimic_data_v5/meta.pkl'
LABELS_CSV = PIPE / 'data/mimic_data_v5/mimic_labels.csv'
OUT_DIR    = PIPE / 'explainability/E2_shap_contrastive'

BLOCK_SIZE  = 128
N_SUMMARY   = 4
N_OVERFLOW  = 64
MASK_TIME   = -10000.
NO_EVENT_MS = 2   # model-space NO_EVENT

# Stored token ID conventions (stored + 1 = model space)
FEMALE_STORED    = 1
MALE_STORED      = 2
ICD_START_STORED = 9
ICD_END_STORED   = 1509
DEATH_STORED     = 1510

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model from phase2b checkpoint...")
from model_v4 import DelphiV4, DelphiConfigV4
ckpt   = torch.load(str(CKPT), map_location='cpu', weights_only=False)
config = DelphiConfigV4(**ckpt['model_args'])
model  = DelphiV4(config)
model.load_state_dict(ckpt['model'])
model.eval()
model.to(device)
print(f"  vocab_size={config.vocab_size}, block_size={config.block_size}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading val.bin...")
data = np.memmap(str(VAL_BIN), dtype=np.int32, mode='r').reshape(-1, 3)
from utils import get_p2i
p2i = get_p2i(data)
n_patients = len(p2i)
print(f"  {n_patients} patients, {len(data)} rows")

with open(META_PKL, 'rb') as f:
    meta = pickle.load(f)
labels = pd.read_csv(LABELS_CSV)

# ── ICD code lookup ───────────────────────────────────────────────────────────
def name_to_stored_id(name_prefix):
    """Return stored IDs for rows whose 'name' column starts with name_prefix."""
    result = []
    for stored_id in range(ICD_START_STORED, ICD_END_STORED + 1):
        row = labels.iloc[stored_id]
        if pd.notna(row['name']) and str(row['name']).startswith(name_prefix):
            result.append(stored_id)
    return result

def ms_to_name(model_id):
    """Human-readable name for a model-space token ID."""
    if model_id == 0:          return 'PAD'
    stored = model_id - 1
    if stored == 0:            return 'NO_EVENT'
    if stored == FEMALE_STORED: return 'Female'
    if stored == MALE_STORED:   return 'Male'
    if stored < ICD_START_STORED: return f'Demog({stored})'
    if stored == DEATH_STORED: return 'Death'
    if ICD_START_STORED <= stored <= ICD_END_STORED:
        row = labels.iloc[stored]
        code = str(row['name']).split()[0] if pd.notna(row['name']) else f'ICD{stored}'
        return code
    for name, info in meta['lab_vocab'].items():
        if info['token_id'] == stored:
            return name
    return f'tok{model_id}'

# ── Build patient index ────────────────────────────────────────────────────────
print("Building patient index (may take ~1 min)...")
patient_attrs = []
for pid in range(n_patients):
    start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
    toks = data[start:start + length, 2].astype(np.int64)
    ages = data[start:start + length, 1].astype(np.float32)

    sex = ('F' if np.any(toks == FEMALE_STORED) else
           'M' if np.any(toks == MALE_STORED) else '?')

    icd_mask = (toks >= ICD_START_STORED) & (toks <= ICD_END_STORED)
    icd_set  = set(toks[icd_mask].tolist())

    real_ages = ages[ages > 0]
    age_mid_yr = float(np.median(real_ages)) / 365.25 if len(real_ages) > 0 else 0.

    patient_attrs.append({
        'sex':        sex,
        'age_mid_yr': age_mid_yr,
        'icd_set':    icd_set,
        'seq_len':    int(length),
        'has_death':  bool(np.any(toks == DEATH_STORED)),
    })
print(f"  Done.")

# ── Pair selection ─────────────────────────────────────────────────────────────
def match_pairs(a_ids, b_ids, max_age_diff=10., min_shared=3, max_seq_ratio=1.6, max_pairs=3):
    """Match each A candidate to best B by sex + age + shared ICD codes."""
    pairs, used_b = [], set()
    for a in a_ids:
        pa = patient_attrs[a]
        if pa['sex'] == '?':
            continue
        best_b, best_score = None, -1
        for b in b_ids:
            if b in used_b or b == a:
                continue
            pb = patient_attrs[b]
            if pb['sex'] != pa['sex']:
                continue
            if abs(pb['age_mid_yr'] - pa['age_mid_yr']) > max_age_diff:
                continue
            shared = len(pa['icd_set'] & pb['icd_set'])
            if shared < min_shared:
                continue
            ratio = max(pa['seq_len'], pb['seq_len']) / max(min(pa['seq_len'], pb['seq_len']), 1)
            if ratio > max_seq_ratio:
                continue
            score = shared - 0.1 * abs(pb['age_mid_yr'] - pa['age_mid_yr'])
            if score > best_score:
                best_score, best_b = score, b
        if best_b is not None:
            used_b.add(best_b)
            pairs.append((a, best_b))
            if len(pairs) >= max_pairs:
                break
    return pairs

# ── Sequence extraction ────────────────────────────────────────────────────────
def get_patient_sequence(pid, truncate_before=None):
    """
    Extract model-space window (BLOCK_SIZE) + overflow (N_OVERFLOW).

    truncate_before: set of stored token IDs — truncate sequence before
    the last occurrence of any of them (used for patient A to exclude
    the target event itself from the input window).
    """
    start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
    toks = data[start:start + length, 2].astype(np.int64)
    ages = data[start:start + length, 1].astype(np.float32)

    if truncate_before:
        trunc = len(toks)
        for sid in truncate_before:
            occ = np.where(toks == sid)[0]
            if len(occ) > 0:
                trunc = min(trunc, int(occ[-1]))
        if trunc <= 1:
            return None
        toks = toks[:trunc]
        ages = ages[:trunc]

    n = len(toks)
    if n < 2:
        return None

    # Window = last BLOCK_SIZE tokens
    w_start = max(0, n - BLOCK_SIZE)
    win_s   = toks[w_start:]
    win_a   = ages[w_start:]

    pad = BLOCK_SIZE - len(win_s)
    if pad > 0:
        win_s = np.concatenate([np.full(pad, -1, dtype=np.int64), win_s])
        win_a = np.concatenate([np.full(pad, MASK_TIME, dtype=np.float32), win_a])

    win_ms = (win_s + 1).clip(min=0)  # stored+1; -1 → 0 (PAD)

    # Overflow = up to N_OVERFLOW tokens before window
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

    return {'window_ms': win_ms, 'window_ages': win_a,
            'overflow_ms': ov_ms, 'overflow_ages': ov_age}

# ── SHAP model wrapper ────────────────────────────────────────────────────────
def make_shap_fn(seq, target_ms_id):
    """
    Build a batched model_fn for shap.PermutationExplainer.
    Input:  X  shape (batch, BLOCK_SIZE) numpy int64 array of model-space token IDs.
            Masked positions are filled with NO_EVENT_MS by the Independent masker.
    Output: shape (batch,) float64 target logits.
    """
    win_ages = seq['window_ages'].copy()
    ov_ms    = seq['overflow_ms'].copy()
    ov_age   = seq['overflow_ages'].copy()

    has_ov = bool(np.any(ov_ms > 0))
    if has_ov:
        valid = ov_age[ov_age > -9000]
        last_ov_age = float(valid[-1]) if len(valid) > 0 else 0.

    def model_fn(X):
        # X: (batch, BLOCK_SIZE)
        bs = X.shape[0]
        if has_ov:
            s_tok = np.full((bs, N_SUMMARY), NO_EVENT_MS, dtype=np.int64)
            s_age = np.full((bs, N_SUMMARY), last_ov_age, dtype=np.float32)
        else:
            s_tok = np.zeros((bs, N_SUMMARY), dtype=np.int64)
            s_age = np.full((bs, N_SUMMARY), MASK_TIME, dtype=np.float32)

        full_toks = np.concatenate([s_tok, X.astype(np.int64)], axis=1)
        full_ages = np.concatenate([s_age, np.tile(win_ages, (bs, 1))], axis=1)

        x_t   = torch.tensor(full_toks, dtype=torch.long,  device=device)
        a_t   = torch.tensor(full_ages, dtype=torch.float, device=device)
        ov_xt = torch.tensor(ov_ms,  dtype=torch.long,  device=device).unsqueeze(0).repeat(bs, 1)
        ov_at = torch.tensor(ov_age, dtype=torch.float, device=device).unsqueeze(0).repeat(bs, 1)

        with torch.no_grad():
            logits, *_ = model(x_t, a_t, overflow_idx=ov_xt, overflow_age=ov_at)

        return logits[:, -1, target_ms_id].cpu().numpy()

    return model_fn

# ── Waterfall plot ─────────────────────────────────────────────────────────────
TOP_N = 10

def plot_waterfall_pair(pair_idx, group_name, target_name,
                        a_seq, b_seq, a_shap, b_shap, a_pid, b_pid):
    pa, pb = patient_attrs[a_pid], patient_attrs[b_pid]

    def items_from_shap(seq, shap_exp):
        win_ms   = seq['window_ms']
        win_ages = seq['window_ages']
        v = shap_exp.values[0]
        vals = v[:, 0] if v.ndim == 2 else v   # (BLOCK_SIZE,)
        items    = []
        for i in range(BLOCK_SIZE):
            ms = int(win_ms[i])
            if ms == 0:
                continue   # PAD
            age_yr = float(win_ages[i]) / 365.25
            name   = ms_to_name(ms)
            items.append({'label': f'{name}@{age_yr:.0f}y', 'shap': float(vals[i])})
        return items

    a_items = items_from_shap(a_seq, a_shap)
    b_items = items_from_shap(b_seq, b_shap)

    # Union of top-10 by difference first, then by individual magnitude
    a_top10 = {x['label'] for x in sorted(a_items, key=lambda x: abs(x['shap']), reverse=True)[:TOP_N]}
    b_top10 = {x['label'] for x in sorted(b_items, key=lambda x: abs(x['shap']), reverse=True)[:TOP_N]}
    candidate_labels = list(a_top10 | b_top10)

    a_by_label = {x['label']: x['shap'] for x in a_items}
    b_by_label = {x['label']: x['shap'] for x in b_items}

    a_v = np.array([a_by_label.get(l, 0.) for l in candidate_labels])
    b_v = np.array([b_by_label.get(l, 0.) for l in candidate_labels])

    # Sort by absolute difference to put the most contrastive tokens on top
    diff_order = np.argsort(np.abs(a_v - b_v))[::-1][:TOP_N]
    top_labels = [candidate_labels[i] for i in diff_order]
    a_plot     = a_v[diff_order]
    b_plot     = b_v[diff_order]

    # Shared-token validation: compute mean abs difference on shared tokens
    shared_labels = a_top10 & b_top10
    if shared_labels:
        shared_diffs = [abs(a_by_label.get(l, 0.) - b_by_label.get(l, 0.))
                        for l in shared_labels]
        shared_mean_val = np.mean([abs(a_by_label.get(l, 0.)) for l in shared_labels])
        if shared_mean_val > 0:
            shared_pct = np.mean(shared_diffs) / shared_mean_val * 100
            print(f"    Shared-token SHAP divergence: {shared_pct:.1f}%  (check: should be <20% ideally)")

    a_base = float(np.atleast_1d(a_shap.base_values.ravel())[0])
    b_base = float(np.atleast_1d(b_shap.base_values.ravel())[0])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    y = np.arange(len(top_labels))

    def _draw(ax, vals, base, title_suffix, pid_label):
        colors = ['#d62728' if v >= 0 else '#1f77b4' for v in vals]
        ax.barh(y, vals, color=colors, height=0.65, edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(top_labels[::-1] if False else top_labels, fontsize=9)
        ax.set_xlabel('SHAP value (log-rate)', fontsize=10)
        ax.invert_yaxis()

        for i, v in enumerate(vals):
            mult = np.exp(v)
            lbl  = f'×{mult:.2f}'
            offset = max(abs(v) * 0.05, 0.01)
            ax.text(v + (offset if v >= 0 else -offset), i, lbl,
                    va='center', ha='left' if v >= 0 else 'right', fontsize=7.5)

        pred_logit = base + float(np.sum([a_by_label.get(l, 0.) for l in top_labels]
                                         if 'A' in pid_label else
                                         [b_by_label.get(l, 0.) for l in top_labels]))
        ax.text(0.03, 0.97,
                f'Baseline: ×{np.exp(base):.2f}\nPrediction (top tokens): ×{np.exp(pred_logit):.2f}',
                transform=ax.transAxes, fontsize=8.5, va='top', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.set_title(title_suffix, fontsize=10, fontweight='bold', pad=6)

    _draw(ax_a, a_plot, a_base,
          f"Patient A  (has {target_name})\n"
          f"Sex={pa['sex']}, Age≈{pa['age_mid_yr']:.0f}y, N_ICD={len(pa['icd_set'])}",
          'A')
    _draw(ax_b, b_plot, b_base,
          f"Patient B  (no {target_name})\n"
          f"Sex={pb['sex']}, Age≈{pb['age_mid_yr']:.0f}y, N_ICD={len(pb['icd_set'])}",
          'B')

    shared_n = len(pa['icd_set'] & pb['icd_set'])
    fig.suptitle(f'{group_name} — Pair {pair_idx + 1} — Target: {target_name}  '
                 f'(shared ICD codes: {shared_n})', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f'contrastive_waterfall_{group_name}_pair{pair_idx + 1}.pdf'
    out   = OUT_DIR / 'figures' / fname
    plt.savefig(str(out), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved: {out.name}")

# ── Group definitions ─────────────────────────────────────────────────────────
GROUPS = [
    {
        'name':           'chapterIX_cardiovascular',
        'csv_name':       'pairs_chapterIX.csv',
        'required':       ['I10', 'E11'],     # both patients must have these
        'target_prefix':  'I21',
        'target_name':    'I21 (MI)',
    },
    {
        'name':           'chapterII_tumor',
        'csv_name':       'pairs_chapterII.csv',
        'required':       ['K86'],
        'target_prefix':  'C25',
        'target_name':    'C25 (Pancreatic Cancer)',
    },
    {
        'name':           'death_mortality',
        'csv_name':       'pairs_death.csv',
        'required':       ['A41'],
        'target_prefix':  None,               # Death token, not ICD
        'target_stored':  DEATH_STORED,
        'target_name':    'Death',
    },
]

# ── Main loop ─────────────────────────────────────────────────────────────────
for group in GROUPS:
    print(f"\n{'='*60}\nGroup: {group['name']}")

    # Resolve target stored/model IDs
    if group['target_prefix'] is not None:
        tgt_stored_ids = set(name_to_stored_id(group['target_prefix']))
        if not tgt_stored_ids:
            print(f"  No stored IDs for prefix {group['target_prefix']}, skipping")
            continue
        tgt_ms_id = next(iter(tgt_stored_ids)) + 1
        print(f"  Target: {group['target_prefix']}  ({len(tgt_stored_ids)} stored IDs), ms={tgt_ms_id}")
    else:
        tgt_stored_ids = {group['target_stored']}
        tgt_ms_id      = group['target_stored'] + 1
        print(f"  Target: Death  (model space {tgt_ms_id})")

    # Resolve required code stored IDs
    req_groups = [set(name_to_stored_id(c)) for c in group['required']]

    # Classify patients
    a_ids, b_ids = [], []
    for pid, attrs in enumerate(patient_attrs):
        icd = attrs['icd_set']
        has_req = all(len(icd & rg) > 0 for rg in req_groups)
        if not has_req:
            continue
        has_tgt = (len(icd & tgt_stored_ids) > 0
                   if group['target_prefix'] else attrs['has_death'])
        (a_ids if has_tgt else b_ids).append(pid)

    print(f"  A candidates: {len(a_ids)},  B candidates: {len(b_ids)}")
    if not a_ids or not b_ids:
        print("  Insufficient candidates, skipping")
        continue

    pairs = match_pairs(a_ids, b_ids)
    print(f"  Matched pairs: {len(pairs)}")
    if not pairs:
        print("  No pairs found, skipping")
        continue

    # Save pair metadata
    pair_rows = []
    for a, b in pairs:
        pa, pb = patient_attrs[a], patient_attrs[b]
        pair_rows.append({
            'patient_A': a, 'patient_B': b,
            'A_sex': pa['sex'], 'B_sex': pb['sex'],
            'A_age_yr': round(pa['age_mid_yr'], 1), 'B_age_yr': round(pb['age_mid_yr'], 1),
            'A_seq_len': pa['seq_len'], 'B_seq_len': pb['seq_len'],
            'shared_icd': len(pa['icd_set'] & pb['icd_set']),
        })
    pd.DataFrame(pair_rows).to_csv(OUT_DIR / 'pair_selection' / group['csv_name'], index=False)
    print(f"  Pair selection saved.")

    # Background for PermutationExplainer: all positions = NO_EVENT_MS
    background = np.full((1, BLOCK_SIZE), NO_EVENT_MS, dtype=np.int64)

    # SHAP + plot per pair
    for pair_idx, (a_pid, b_pid) in enumerate(pairs):
        print(f"\n  Pair {pair_idx + 1}: A={a_pid}, B={b_pid}")

        a_seq = get_patient_sequence(a_pid, truncate_before=tgt_stored_ids)
        b_seq = get_patient_sequence(b_pid)

        if a_seq is None:
            print("    A sequence extraction failed, skipping")
            continue
        if b_seq is None:
            print("    B sequence extraction failed, skipping")
            continue

        a_window = a_seq['window_ms'].reshape(1, -1)  # (1, BLOCK_SIZE)
        b_window = b_seq['window_ms'].reshape(1, -1)

        print(f"    Running SHAP for A...")
        a_fn  = make_shap_fn(a_seq, tgt_ms_id)
        a_exp = shap.PermutationExplainer(a_fn, background)
        a_sv  = a_exp(a_window, max_evals=2 * BLOCK_SIZE + 1)

        print(f"    Running SHAP for B...")
        b_fn  = make_shap_fn(b_seq, tgt_ms_id)
        b_exp = shap.PermutationExplainer(b_fn, background)
        b_sv  = b_exp(b_window, max_evals=2 * BLOCK_SIZE + 1)

        # Persist SHAP values
        json_path = OUT_DIR / 'shap_values' / f'pair_{group["name"]}_{pair_idx + 1:02d}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'group': group['name'], 'pair_idx': pair_idx,
                'target_name': group['target_name'], 'target_ms_id': tgt_ms_id,
                'A': {'patient_id': int(a_pid),
                      'base_value': float(a_sv.base_values.ravel()[0]),
                      'shap_values': a_sv.values[0].ravel().tolist(),
                      'window_ms': a_seq['window_ms'].tolist(),
                      'window_ages': a_seq['window_ages'].tolist()},
                'B': {'patient_id': int(b_pid),
                      'base_value': float(b_sv.base_values.ravel()[0]),
                      'shap_values': b_sv.values[0].ravel().tolist(),
                      'window_ms': b_seq['window_ms'].tolist(),
                      'window_ages': b_seq['window_ages'].tolist()},
            }, f, indent=2)

        plot_waterfall_pair(pair_idx, group['name'], group['target_name'],
                            a_seq, b_seq, a_sv, b_sv, a_pid, b_pid)

print("\nE2 complete.")
