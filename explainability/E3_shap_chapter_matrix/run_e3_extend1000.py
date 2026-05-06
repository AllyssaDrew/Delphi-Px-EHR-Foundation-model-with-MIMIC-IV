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
E3 Extension: resume from N=500, add patients 501-1000 using same seed.
Loads existing cumulative sums, processes only the new 500 patients.
Snapshots at N=600/700/800/900/1000.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PIPE   = PIPELINE_DIR
DELPHI = DELPHI_DIR
sys.path.insert(0, str(DELPHI))

CKPT       = DELPHI / 'checkpoints/mimic_v5_phase2b/ckpt.pt'
VAL_BIN    = PIPE / 'data/mimic_data_v5/val.bin'
META_PKL   = PIPE / 'data/mimic_data_v5/meta.pkl'
LABELS_CSV = PIPE / 'data/mimic_data_v5/mimic_labels.csv'
OUT_DIR    = PIPE / 'explainability/E3_shap_chapter_matrix'

BLOCK_SIZE  = 128
N_SUMMARY   = 4
N_OVERFLOW  = 64
MASK_TIME   = -10000.
NO_EVENT_MS = 2

ICD_START_STORED = 9
ICD_END_STORED   = 1509
DEATH_STORED     = 1510
DEATH_MODEL      = DEATH_STORED + 1
ICD_MODEL_START  = ICD_START_STORED + 1
ICD_MODEL_END    = ICD_END_STORED + 1
N_ICD_TARGETS    = ICD_END_STORED - ICD_START_STORED + 1

N_TOTAL      = 1000   # final target
N_DONE       = 500    # already processed
SNAPSHOTS    = {600, 700, 800, 900, 1000}
RANDOM_SEED  = 42

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
print(f"  vocab_size={config.vocab_size}", flush=True)

print("Loading val.bin...", flush=True)
data = np.memmap(str(VAL_BIN), dtype=np.int32, mode='r').reshape(-1, 3)
from utils import get_p2i
p2i = get_p2i(data)
n_patients_total = len(p2i)
print(f"  {n_patients_total} patients", flush=True)

with open(META_PKL, 'rb') as f:
    meta = pickle.load(f)
labels = pd.read_csv(LABELS_CSV)

CHAPTER_RANGES = [
    ('A', 'B', 0,  'I: Infectious'),
    ('C', 'D', 1,  'II: Neoplasms'),
    ('D', 'D', 2,  'III: Blood'),
    ('E', 'E', 3,  'IV: Endocrine'),
    ('F', 'F', 4,  'V: Mental'),
    ('G', 'G', 5,  'VI: Nervous'),
    ('H', 'H', 6,  'VII/VIII: Eye/Ear'),
    ('I', 'I', 7,  'IX: Circulatory'),
    ('J', 'J', 8,  'X: Respiratory'),
    ('K', 'K', 9,  'XI: Digestive'),
    ('L', 'L', 10, 'XII: Skin'),
    ('M', 'M', 11, 'XIII: Musculoskeletal'),
    ('N', 'N', 12, 'XIV: Genitourinary'),
    ('O', 'O', 13, 'XV: Pregnancy'),
    ('P', 'P', 14, 'XVI: Perinatal'),
    ('Q', 'Q', 15, 'XVII: Congenital'),
    ('R', 'Z', 16, 'XVIII+: Other'),
]
N_CHAPTERS = 17
CHAPTER_NAMES = [r[3] for r in CHAPTER_RANGES]

def icd_to_chapter(code_str):
    if not code_str or len(code_str) < 1:
        return -1
    c = code_str[0].upper()
    num_part = code_str[1:3] if len(code_str) >= 3 else ''
    if c == 'D':
        try:
            n = int(num_part)
            return 2 if n >= 50 else 1
        except ValueError:
            return 1
    for lo, hi, idx, _ in CHAPTER_RANGES:
        if c == 'D':
            continue
        if lo <= c <= hi:
            return idx
    return -1

stored_id_to_chapter = {}
for stored_id in range(ICD_START_STORED, ICD_END_STORED + 1):
    row = labels.iloc[stored_id]
    if pd.notna(row['name']):
        code = str(row['name']).split()[0]
        ch = icd_to_chapter(code)
        stored_id_to_chapter[stored_id] = ch
    else:
        stored_id_to_chapter[stored_id] = -1

target_chapters = np.array([
    stored_id_to_chapter.get(ICD_START_STORED + i, -1)
    for i in range(N_ICD_TARGETS)
], dtype=np.int32)

def get_patient_sequence(pid):
    start, length = int(p2i[pid, 0]), int(p2i[pid, 1])
    toks = data[start:start + length, 2].astype(np.int64)
    ages = data[start:start + length, 1].astype(np.float32)
    n = len(toks)
    if n < 2:
        return None

    seq_model = toks + 1
    if seq_model[-1] == DEATH_MODEL:
        icd_mask = (seq_model >= ICD_MODEL_START) & (seq_model <= ICD_MODEL_END)
        if not icd_mask.any():
            return None
        pred_pos = int(np.where(icd_mask)[0][-1])
    else:
        pred_pos = n - 1

    toks = toks[:pred_pos + 1]
    ages = ages[:pred_pos + 1]
    n = len(toks)

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

    real_win_ages = win_a[win_a > MASK_TIME + 1]
    pred_age = float(real_win_ages[-1]) if len(real_win_ages) > 0 else 0.

    return {
        'window_ms':    win_ms,
        'window_ages':  win_a,
        'overflow_ms':  ov_ms,
        'overflow_ages': ov_age,
        'pred_age':     pred_age,
    }

def make_shap_fn_multi(seq):
    win_ages = seq['window_ages'].copy()
    ov_ms    = seq['overflow_ms'].copy()
    ov_age   = seq['overflow_ages'].copy()

    has_ov = bool(np.any(ov_ms > 0))
    if has_ov:
        valid = ov_age[ov_age > MASK_TIME + 1]
        last_ov_age = float(valid[-1]) if len(valid) > 0 else 0.

    def model_fn(X):
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

        return logits[:, -1, ICD_MODEL_START:ICD_MODEL_END + 1].cpu().numpy()

    return model_fn

def aggregate_to_chapters(shap_values, win_ms, win_ages, pred_age):
    full_mat  = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.float64)
    short_mat = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.float64)
    long_mat  = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.float64)
    full_cnt  = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.int32)
    short_cnt = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.int32)
    long_cnt  = np.zeros((N_CHAPTERS, N_CHAPTERS), dtype=np.int32)

    for pos in range(BLOCK_SIZE):
        ms = int(win_ms[pos])
        if ms == 0:
            continue
        stored = ms - 1
        if stored < ICD_START_STORED or stored > ICD_END_STORED:
            continue
        pred_ch = stored_id_to_chapter.get(stored, -1)
        if pred_ch < 0:
            continue

        sv = shap_values[pos]
        age = float(win_ages[pos])
        time_delta = pred_age - age if age > MASK_TIME + 1 else np.inf

        for tgt_ch in range(N_CHAPTERS):
            mask = (target_chapters == tgt_ch)
            if not mask.any():
                continue
            mean_sv = float(sv[mask].mean())
            full_mat[pred_ch, tgt_ch]  += mean_sv
            full_cnt[pred_ch, tgt_ch]  += 1
            if time_delta < 30:
                short_mat[pred_ch, tgt_ch] += mean_sv
                short_cnt[pred_ch, tgt_ch] += 1
            elif time_delta > 180:
                long_mat[pred_ch, tgt_ch] += mean_sv
                long_cnt[pred_ch, tgt_ch] += 1

    def safe_div(a, b):
        out = np.zeros_like(a)
        mask = b > 0
        out[mask] = a[mask] / b[mask]
        return out

    return safe_div(full_mat, full_cnt), safe_div(short_mat, short_cnt), safe_div(long_mat, long_cnt)

# ── Load existing N=500 means, recover cumsums ────────────────────────────────
print(f"Loading N=500 matrices as starting point...", flush=True)
mat500_full  = np.load(OUT_DIR / 'shap_aggregated.npy')
mat500_short = np.load(OUT_DIR / 'shap_aggregated_shortterm.npy')
mat500_long  = np.load(OUT_DIR / 'shap_aggregated_longterm.npy')

# mat500 is the mean over 500 patients → cumsum = mean * 500
# (safe because aggregate_to_chapters returns per-patient mean, then main loop sums them)
cumsum_full  = mat500_full  * N_DONE
cumsum_short = mat500_short * N_DONE
cumsum_long  = mat500_long  * N_DONE
print(f"  Cumsums recovered from N={N_DONE} snapshots.", flush=True)

# ── Sample all 1000 patients with same seed, take [500:1000] ─────────────────
rng = np.random.default_rng(RANDOM_SEED)
all_pids = rng.choice(n_patients_total, size=min(N_TOTAL, n_patients_total), replace=False)
new_pids = all_pids[N_DONE:]  # patients 501-1000
print(f"New patients to process: {len(new_pids)}", flush=True)

background = np.full((1, BLOCK_SIZE), NO_EVENT_MS, dtype=np.int64)

processed = N_DONE  # total including N=500 already done
t0 = time.time()

for i, pid in enumerate(new_pids):
    seq = get_patient_sequence(int(pid))
    if seq is None:
        print(f"  [+{i+1}/{len(new_pids)}] pid={pid}: skipped (bad seq)", flush=True)
        continue

    window_in = seq['window_ms'].reshape(1, -1)

    t1 = time.time()
    fn = make_shap_fn_multi(seq)
    exp = shap.PermutationExplainer(fn, background)
    sv_obj = exp(window_in, max_evals=2 * BLOCK_SIZE + 1)
    elapsed = time.time() - t1

    sv = sv_obj.values[0]
    p_full, p_short, p_long = aggregate_to_chapters(
        sv, seq['window_ms'], seq['window_ages'], seq['pred_age']
    )

    cumsum_full  += p_full
    cumsum_short += p_short
    cumsum_long  += p_long
    processed += 1

    print(f"  [+{i+1}/{len(new_pids)}] pid={pid} done in {elapsed:.1f}s  (total_n={processed})", flush=True)

    if processed in SNAPSHOTS:
        snap = cumsum_full / processed
        np.save(OUT_DIR / 'convergence_check' / f'matrix_n{processed}.npy', snap)
        print(f"  >> Snapshot saved: N={processed}", flush=True)

print(f"\nTotal processed: {processed}  New time: {(time.time()-t0)/60:.1f} min", flush=True)

if processed <= N_DONE:
    print("No new patients processed, exiting.")
    sys.exit(1)

# Final matrices
mat_full  = cumsum_full  / processed
mat_short = cumsum_short / processed
mat_long  = cumsum_long  / processed

np.save(OUT_DIR / 'shap_aggregated.npy',           mat_full)
np.save(OUT_DIR / 'shap_aggregated_shortterm.npy', mat_short)
np.save(OUT_DIR / 'shap_aggregated_longterm.npy',  mat_long)
print("Aggregated matrices saved (N=1000).", flush=True)

# ── Convergence curve (all snapshots 100-1000) ────────────────────────────────
all_snaps = []
for n in [100, 200, 300, 500, 600, 700, 800, 900, 1000]:
    p = OUT_DIR / 'convergence_check' / f'matrix_n{n}.npy'
    if p.exists():
        all_snaps.append((n, np.load(str(p))))

if len(all_snaps) >= 2:
    frob_dists = []
    for j in range(1, len(all_snaps)):
        n_prev, m_prev = all_snaps[j-1]
        n_curr, m_curr = all_snaps[j]
        frob_dists.append((n_curr, float(np.linalg.norm(m_curr - m_prev, 'fro'))))

    ns_plot, ds_plot = zip(*frob_dists)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns_plot, ds_plot, 'o-', color='#2c7bb6')
    ax.axvline(500, color='gray', linestyle='--', alpha=0.5, label='N=500 (original)')
    ax.set_xlabel('N patients')
    ax.set_ylabel('Frobenius distance (vs previous snapshot)')
    ax.set_title('Chapter Matrix Convergence (N=100 to 1000)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'convergence_check' / 'convergence_curve_1000.pdf'), bbox_inches='tight')
    plt.close()
    print("Convergence curve (1000) saved.", flush=True)

    if len(ds_plot) >= 2:
        ratio = ds_plot[-1] / ds_plot[0] if ds_plot[0] > 0 else float('inf')
        print(f"  Convergence ratio (last/first Frob dist): {ratio:.3f}  "
              f"({'CONVERGED' if ratio < 0.20 else 'NOT CONVERGED'})", flush=True)

    # Also report 500→1000 sub-ratio
    d500 = next((d for n, d in frob_dists if n == 500), None)
    d1000 = next((d for n, d in frob_dists if n == 1000), None)
    if d500 and d1000:
        print(f"  500→1000 Frob dist: {d1000:.4f}  (vs 400→500: {d500:.4f})", flush=True)

# ── Heatmap plotting ──────────────────────────────────────────────────────────
short_labels = [n.split(':')[0] for n in CHAPTER_NAMES]

def plot_heatmap(mat, title, out_path, vmax=None):
    exp_mat = np.exp(mat)
    fig, ax = plt.subplots(figsize=(12, 10))

    mat_log = mat.copy()
    if vmax is None:
        vmax = max(abs(mat_log[mat_log != 0]).max(), 0.1) if (mat_log != 0).any() else 1.0

    im = ax.imshow(mat_log, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean SHAP (log-rate)', fontsize=10)

    for r in range(N_CHAPTERS):
        for c in range(N_CHAPTERS):
            v = mat_log[r, c]
            if abs(v) > 0.01:
                txt = f'×{exp_mat[r,c]:.2f}'
                color = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(c, r, txt, ha='center', va='center', fontsize=5.5, color=color)

    ax.set_xticks(range(N_CHAPTERS))
    ax.set_yticks(range(N_CHAPTERS))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel('Target chapter (predicted)', fontsize=10)
    ax.set_ylabel('Predictor chapter (history)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {Path(out_path).name}", flush=True)

vmax_global = max(
    abs(mat_full[mat_full != 0]).max()  if (mat_full  != 0).any() else 0.1,
    abs(mat_short[mat_short != 0]).max() if (mat_short != 0).any() else 0.1,
    abs(mat_long[mat_long != 0]).max()   if (mat_long  != 0).any() else 0.1,
)

plot_heatmap(mat_full,  f'Chapter SHAP Matrix (N={processed}, all time)',
             OUT_DIR / 'figures' / 'chapter_matrix_full.pdf', vmax=vmax_global)
plot_heatmap(mat_short, f'Chapter SHAP Matrix (N={processed}, short-term <30d)',
             OUT_DIR / 'figures' / 'chapter_matrix_shortterm.pdf', vmax=vmax_global)
plot_heatmap(mat_long,  f'Chapter SHAP Matrix (N={processed}, long-term >180d)',
             OUT_DIR / 'figures' / 'chapter_matrix_longterm.pdf', vmax=vmax_global)

print("\nE3 extension to N=1000 complete.", flush=True)
