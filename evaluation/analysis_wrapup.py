"""
Delphi v6.1 — Wrap-up Analysis Script

Analysis A: Traditional ML baseline comparison
  LR  (age + sex + 33 phenotype binary)
  XGB (LR features + ICD bag-of-words)

Analysis B: Survival analysis metrics
  C-index (Weibull time head, concordance)
  30-day AUC (in-hospital mortality)

Outputs → Phase_E/results/wrapup/
  ml_baseline_results.csv
  survival_metrics.csv
  analysis_summary.txt
"""
import os
from pathlib import Path

# ── Portable path configuration ────────────────────────────────────────────────
# Set DELPHI_PROJECT_ROOT to the directory that contains both
# mimic_pipeline/ and Delphi/Delphi-main/ as siblings.
#   export DELPHI_PROJECT_ROOT=/your/project/root
# Alternatively MIMIC_PIPELINE_DIR and DELPHI_DIR can be set individually.
_ROOT        = Path(os.environ.get('DELPHI_PROJECT_ROOT',
                                    Path(__file__).resolve().parents[1]))
PIPELINE_DIR = Path(os.environ.get('MIMIC_PIPELINE_DIR',
                                    _ROOT / 'mimic_pipeline'))
DELPHI_DIR   = Path(os.environ.get('DELPHI_DIR',
                                    _ROOT / 'Delphi' / 'Delphi-main'))
# ──────────────────────────────────────────────────────────────────────────────


import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore')

DELPHI = DELPHI_DIR
PIPE   = PIPELINE_DIR
sys.path.insert(0, str(DELPHI))

from model_v4 import DelphiV4, DelphiConfigV4
from utils import get_batch_v4, get_p2i

OUT_DIR = PIPE / 'clinical_phenotyping' / 'Phase_E' / 'results' / 'wrapup'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Shared data loading
# ══════════════════════════════════════════════════════════════════════════════
DATA_V5  = PIPE / 'data' / 'mimic_data_v5'
DATA_V61 = PIPE / 'data' / 'mimic_data_v61'
PHENO_CSV = PIPE / 'clinical_phenotyping' / 'Phase_B' / 'phenotype_tokens.csv'

meta_v61 = pickle.load(open(DATA_V61 / 'meta_v61.pkl', 'rb'))
meta_v5  = pickle.load(open(DATA_V5  / 'meta.pkl',    'rb'))

DEATH_STORED = 1510  # same in v5 and v61
RESERVED     = 9
FEMALE_STORED, MALE_STORED = 2, 3
PHENO_STORED = meta_v61['PHENOTYPE_TOKENS']  # name → stored_id
PHENO_NAMES  = sorted(PHENO_STORED, key=lambda n: PHENO_STORED[n])  # sorted by stored_id

splits_df = pd.read_csv(DATA_V61 / 'patient_splits.csv')
cuts_df   = pd.read_csv(DATA_V61 / 'test_cutoffs.csv')   # same for v5 and v61

TRAIN_SIDS = sorted(splits_df[splits_df['split'] == 'train']['subject_id'])
TEST_SIDS  = sorted(splits_df[splits_df['split'] == 'test']['subject_id'])
TRAIN_SID2PIDX = {sid: i for i, sid in enumerate(TRAIN_SIDS)}
TEST_SID2PIDX  = {sid: i for i, sid in enumerate(TEST_SIDS)}

# patient_idx → subject_id for test
TEST_PIDX2SID  = {v: k for k, v in TEST_SID2PIDX.items()}

print(f"Train: {len(TRAIN_SIDS)} patients | Test: {len(TEST_SIDS)} patients", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. Feature extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_phenotype_features(sids_set):
    """Returns dict: subject_id → {pheno_name: 1/0} for patients in sids_set."""
    print("  Loading phenotype features from CSV …", flush=True)
    pheno_df = pd.read_csv(PHENO_CSV,
                           dtype={'subject_id': int, 'hadm_id': int, 'tokens': str})
    # Aggregate to patient level: union of tokens across all admissions
    sid_to_phenos = defaultdict(set)
    for _, row in pheno_df.iterrows():
        sid = int(row['subject_id'])
        if sid not in sids_set:
            continue
        if pd.isna(row['tokens']) or row['tokens'].strip() == '':
            continue
        for t in str(row['tokens']).split():
            sid_to_phenos[sid].add(int(t))
    return sid_to_phenos   # stored_id sets


def extract_features_from_bin(bin_path, pidx_to_sid, n_icd=1501):
    """
    Extract per-patient features from a binary file:
      - sex (0/1 Female)
      - age_days (last event)
      - ICD sparse bag-of-words (stored tokens 9 .. 9+n_icd-1)
      - death label (stored token DEATH_STORED present)
    Returns: pid_list, ages, sexes, icd_csr, death_labels
    """
    print(f"  Reading {bin_path.name} …", flush=True)
    data = np.memmap(str(bin_path), dtype=np.uint32, mode='r').reshape(-1, 3)
    p2i  = get_p2i(data)
    n_pats = len(p2i)

    ages   = np.zeros(n_pats, dtype=np.float32)
    sexes  = np.zeros(n_pats, dtype=np.float32)  # 1=Female
    deaths = np.zeros(n_pats, dtype=np.int8)

    rows_icd, cols_icd, vals_icd = [], [], []

    for pidx in tqdm(range(n_pats), desc=f'  {bin_path.name}', leave=False):
        start, length = p2i[pidx]
        events = data[start: start + length]  # (L, 3)

        toks = events[:, 2].astype(int)
        ags  = events[:, 1].astype(float)

        # Age: last non-padding age
        valid_age = ags[ags > 0]
        ages[pidx] = float(valid_age[-1]) if len(valid_age) > 0 else 0.0

        # Sex: Female=stored 2, Male=stored 3
        sexes[pidx]  = float(np.any(toks == FEMALE_STORED))

        # Death label
        deaths[pidx] = int(np.any(toks == DEATH_STORED))

        # ICD bag-of-words: stored 9 .. 9+n_icd-1
        icd_mask = (toks >= RESERVED) & (toks < RESERVED + n_icd)
        icd_toks = toks[icd_mask]
        if len(icd_toks) > 0:
            for tok, cnt in zip(*np.unique(icd_toks, return_counts=True)):
                col = int(tok) - RESERVED
                if 0 <= col < n_icd:
                    rows_icd.append(pidx)
                    cols_icd.append(col)
                    vals_icd.append(int(cnt))

    icd_csr = sp.csr_matrix((vals_icd, (rows_icd, cols_icd)),
                             shape=(n_pats, n_icd), dtype=np.float32)
    return p2i, ages, sexes, icd_csr, deaths


# ══════════════════════════════════════════════════════════════════════════════
# Analysis A — ML Baseline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("=== Analysis A: Traditional ML Baseline ===", flush=True)
print("="*70, flush=True)

n_icd   = 1501          # stored 9..1509
n_pheno = len(PHENO_NAMES)  # 33

# ── A1. Phenotype features ────────────────────────────────────────────────────
train_sids_set = set(TRAIN_SIDS)
test_sids_set  = set(TEST_SIDS)
all_sids_set   = train_sids_set | test_sids_set
pheno_map      = load_phenotype_features(all_sids_set)  # sid → set of stored_ids

def sid_to_pheno_vec(sid):
    present = pheno_map.get(int(sid), set())
    return np.array([1.0 if PHENO_STORED[n] in present else 0.0
                     for n in PHENO_NAMES], dtype=np.float32)

# ── A2. Binary features from train.bin ───────────────────────────────────────
tr_p2i, tr_ages, tr_sexes, tr_icd, tr_deaths = \
    extract_features_from_bin(DATA_V61 / 'train.bin',
                               {i: sid for i, sid in enumerate(TRAIN_SIDS)})

# ── A3. Binary features from test_input.bin ───────────────────────────────────
te_p2i, te_ages, te_sexes, te_icd, te_deaths_input = \
    extract_features_from_bin(DATA_V61 / 'test_input.bin',
                               TEST_PIDX2SID)

# Override test death labels with test_future.bin
print("  Loading test Death labels from test_future.bin …", flush=True)
fut_data   = np.fromfile(str(DATA_V61 / 'test_future.bin'), dtype=np.uint32).reshape(-1, 3)
te_deaths  = np.zeros(len(te_p2i), dtype=np.int8)
for row in fut_data:
    pidx, _, tok = int(row[0]), int(row[1]), int(row[2])
    if tok == DEATH_STORED and pidx < len(te_deaths):
        te_deaths[pidx] = 1

n_te_dead = te_deaths.sum()
print(f"  Test Death labels: {n_te_dead}/{len(te_deaths)} ({n_te_dead/len(te_deaths)*100:.1f}%)",
      flush=True)

# ── A4. Build phenotype feature vectors ──────────────────────────────────────
print("  Building phenotype feature matrices …", flush=True)
tr_pheno = np.vstack([sid_to_pheno_vec(TRAIN_SIDS[i]) for i in range(len(TRAIN_SIDS))])
te_pheno = np.vstack([sid_to_pheno_vec(TEST_SIDS[i])  for i in range(len(TEST_SIDS))])

# Age in years, sex binary
tr_demo = np.column_stack([tr_ages / 365.25, tr_sexes])
te_demo = np.column_stack([te_ages / 365.25, te_sexes])

# ── A5. Build feature matrices ────────────────────────────────────────────────
# LR: age + sex + 33 phenotype (35-dim dense)
X_train_lr = np.hstack([tr_demo, tr_pheno])
X_test_lr  = np.hstack([te_demo, te_pheno])

# XGB: LR features + ICD bag-of-words (35 + 1501 = 1536 features, sparse)
from scipy.sparse import hstack as sp_hstack
X_train_xgb = sp_hstack([sp.csr_matrix(np.hstack([tr_demo, tr_pheno])), tr_icd])
X_test_xgb  = sp_hstack([sp.csr_matrix(np.hstack([te_demo, te_pheno])), te_icd])

y_train = tr_deaths.astype(int)
y_test  = te_deaths.astype(int)

print(f"\n  Train: {y_train.sum()} deaths / {len(y_train)} patients "
      f"({y_train.mean()*100:.1f}%)", flush=True)
print(f"  Test:  {y_test.sum()} deaths / {len(y_test)} patients "
      f"({y_test.mean()*100:.1f}%)", flush=True)

# ── A6. Train and evaluate models ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

print("\n  [LR] Logistic Regression …", flush=True)
scaler_lr  = StandardScaler()
Xtr_lr_s   = scaler_lr.fit_transform(X_train_lr)
Xte_lr_s   = scaler_lr.transform(X_test_lr)
lr_model   = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                 class_weight='balanced')
lr_model.fit(Xtr_lr_s, y_train)
lr_proba   = lr_model.predict_proba(Xte_lr_s)[:, 1]
lr_auc     = roc_auc_score(y_test, lr_proba)
print(f"  LR Death AUC = {lr_auc:.4f}", flush=True)

print("\n  [HGBT] HistGradientBoostingClassifier …", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
# Convert sparse matrix to dense for HGBT (it does not natively accept csr)
X_train_xgb_dense = X_train_xgb.toarray()
X_test_xgb_dense  = X_test_xgb.toarray()
xgb_model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=6, learning_rate=0.05,
    class_weight='balanced', random_state=42,
)
xgb_model.fit(X_train_xgb_dense, y_train)
xgb_proba = xgb_model.predict_proba(X_test_xgb_dense)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"  HGBT Death AUC = {xgb_auc:.4f}", flush=True)

ml_results = pd.DataFrame([
    {'model': 'LR (age+sex+33 phenotype)',        'death_auc': lr_auc,  'n_features': 35},
    {'model': 'HGBT (+ICD bag-of-words)',            'death_auc': xgb_auc, 'n_features': 1536},
    {'model': 'Delphi v5 (sequence, no text)',     'death_auc': 0.6796,  'n_features': None},
    {'model': 'Delphi v6.1 D-F (sequence+pheno)', 'death_auc': 0.7206,  'n_features': None},
])
ml_results.to_csv(OUT_DIR / 'ml_baseline_results.csv', index=False)
print(f"\n  Saved {OUT_DIR}/ml_baseline_results.csv", flush=True)
print(ml_results.to_string(index=False), flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis B — C-index and 30-day AUC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("=== Analysis B: Survival Analysis Metrics ===", flush=True)
print("="*70, flush=True)

# ── B0. Build death time arrays for test patients ─────────────────────────────
# event_times: days from cutoff to death (or censoring)
# For dead patients: death_age (from test_future) - cutoff_age
# For censored:      last_adm_age - cutoff_age
sid_to_cut = dict(zip(cuts_df['subject_id'], cuts_df['cutoff_age']))
sid_to_last = dict(zip(cuts_df['subject_id'], cuts_df['last_adm_age']))

# Death age per test patient from test_future.bin
te_death_age = {}
for row in fut_data:
    pidx, age_days, tok = int(row[0]), int(row[1]), int(row[2])
    if tok == DEATH_STORED and pidx < len(TEST_SIDS):
        sid = TEST_SIDS[pidx]
        te_death_age[sid] = age_days

n_test = len(TEST_SIDS)
event_times    = np.zeros(n_test)
event_observed = np.zeros(n_test, dtype=bool)

for i, sid in enumerate(TEST_SIDS):
    cut  = sid_to_cut.get(sid, 0)
    last = sid_to_last.get(sid, cut)
    if sid in te_death_age:
        event_times[i]    = max(te_death_age[sid] - cut, 1)
        event_observed[i] = True
    else:
        event_times[i]    = max(last - cut, 1)
        event_observed[i] = False

print(f"  Events: {event_observed.sum()} deaths, {(~event_observed).sum()} censored",
      flush=True)
print(f"  Median follow-up: {np.median(event_times):.0f} days", flush=True)

# 30-day label
labels_30d = (event_observed & (event_times <= 30)).astype(int)
print(f"  30-day deaths: {labels_30d.sum()} / {n_test}", flush=True)


def load_delphi(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    conf = DelphiConfigV4(**ckpt['model_args'])
    m = DelphiV4(conf)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    m.load_state_dict(sd)
    m.eval().to(DEVICE)
    return m, conf


def infer_weibull(model, conf, data_dir, batch_size=32):
    """
    Returns per-test-patient (k, lam) from time_head at last sequence position.
    time_head is only called during training (inside if targets is not None), so
    we hook transformer.ln_f to capture hidden states and call time_head manually.
    """
    test_bin = Path(data_dir) / 'test_input.bin'
    data = np.memmap(str(test_bin), dtype=np.uint32, mode='r').reshape(-1, 3)
    p2i  = get_p2i(data)
    n    = len(p2i)

    k_out   = np.zeros(n, dtype=np.float32)
    lam_out = np.zeros(n, dtype=np.float32)

    # Hook transformer.ln_f to get final hidden states (shape: b, T, n_embd)
    hidden_cache = {}
    def hook_ln_f(module, inp, out):
        hidden_cache['x'] = out  # keep on device to pass into time_head
    handle = model.transformer.ln_f.register_forward_hook(hook_ln_f)

    pids = np.arange(n)
    for start in tqdm(range(0, n, batch_size), desc='  Weibull inference', leave=False):
        batch_pids = pids[start: start + batch_size].tolist()
        with torch.no_grad():
            x_in, a_in, _, _, ov_x, ov_a = get_batch_v4(
                batch_pids, data, p2i,
                block_size=conf.block_size,
                n_summary=conf.n_summary,
                n_overflow=conf.n_overflow,
                device=DEVICE,
                no_event_token_rate=1,
                select='right',
            )
            model(x_in, a_in, overflow_idx=ov_x, overflow_age=ov_a)
            # hidden_cache['x'] now has shape (b, T, n_embd) on DEVICE
            x_hidden = hidden_cache['x']            # still on DEVICE
            time_out = model.time_head(x_hidden)    # (b, T, 2)  on DEVICE
            time_last = time_out[:, -1, :].cpu()    # (b, 2)

        k_raw   = time_last[:, 0].numpy()
        lam_raw = time_last[:, 1].numpy()
        k_out[start: start + len(batch_pids)]   = np.exp(k_raw)
        lam_out[start: start + len(batch_pids)] = np.exp(lam_raw)

    handle.remove()
    return k_out, lam_out


def _concordance_index_numpy(event_times, risk_scores, event_observed):
    """Harrell C-index without lifelines. Higher risk_score = more at risk."""
    et = np.asarray(event_times,   dtype=np.float64)
    rs = np.asarray(risk_scores,   dtype=np.float64)
    eo = np.asarray(event_observed, dtype=bool)
    concordant = discordant = tied = 0.0
    for i in np.where(eo)[0]:
        admissible = (et > et[i]) | ((et == et[i]) & (~eo))
        rj = rs[admissible]
        concordant += float((rs[i] > rj).sum())
        discordant += float((rs[i] < rj).sum())
        tied       += float((rs[i] == rj).sum())
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5


def compute_survival_metrics(k, lam, event_times, event_observed, labels_30d):
    from sklearn.metrics import roc_auc_score

    # Median survival = lam * (log2)^(1/k)
    median_surv = lam * (np.log(2) ** (1.0 / k))
    risk_score  = 1.0 / np.clip(median_surv, 1e-6, None)

    cindex = _concordance_index_numpy(event_times, risk_score, event_observed)

    # 30-day mortality AUC
    prob_30d = 1.0 - np.exp(-(30.0 / np.clip(lam, 1e-6, None)) ** k)
    n_pos = labels_30d.sum()
    if n_pos >= 5:
        auc_30d = roc_auc_score(labels_30d, prob_30d)
    else:
        auc_30d = float('nan')

    return cindex, auc_30d, prob_30d


surv_rows = []

for model_name, ckpt_rel, data_dir in [
    ('v5 Phase 2b',   'checkpoints/mimic_v5_phase2b/ckpt.pt',    DATA_V5),
    ('v6.1 Phase D-F', 'checkpoints/mimic_v61_phase_f/ckpt.pt',  DATA_V61),
]:
    print(f"\n  [{model_name}] Loading checkpoint …", flush=True)
    ckpt_path = DELPHI / ckpt_rel
    model, conf = load_delphi(ckpt_path)
    print(f"  [{model_name}] Running Weibull inference …", flush=True)
    k, lam = infer_weibull(model, conf, data_dir, batch_size=64)
    print(f"  [{model_name}] k: mean={k.mean():.3f}  lam: mean={lam.mean():.1f} days",
          flush=True)
    cindex, auc_30d, prob_30d = compute_survival_metrics(
        k, lam, event_times, event_observed, labels_30d)
    print(f"  [{model_name}] C-index = {cindex:.4f}  30-day AUC = {auc_30d:.4f}",
          flush=True)

    surv_rows.append({
        'model':         model_name,
        'c_index':       cindex,
        'auc_30d_mort':  auc_30d,
        'n_test':        n_test,
        'n_events':      int(event_observed.sum()),
        'n_30d_events':  int(labels_30d.sum()),
    })
    del model

surv_df = pd.DataFrame(surv_rows)
surv_df.to_csv(OUT_DIR / 'survival_metrics.csv', index=False)
print(f"\n  Saved {OUT_DIR}/survival_metrics.csv", flush=True)
print(surv_df.to_string(index=False), flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary output
# ══════════════════════════════════════════════════════════════════════════════
summary_lines = [
    "=" * 70,
    "DELPHI v6.1 WRAP-UP ANALYSIS — SUMMARY",
    "=" * 70,
    "",
    "--- Analysis A: ML Baseline Comparison (Death AUC) ---",
]
for _, row in ml_results.iterrows():
    summary_lines.append(f"  {row['model']:<45s}  AUC = {row['death_auc']:.4f}")
summary_lines += [
    "",
    "--- Analysis B: Survival Analysis Metrics ---",
]
for _, row in surv_df.iterrows():
    summary_lines.append(
        f"  {row['model']:<25s}  C-index = {row['c_index']:.4f}  "
        f"30-day AUC = {row['auc_30d_mort']:.4f}  "
        f"(N={row['n_test']}, deaths={row['n_events']}, 30d={row['n_30d_events']})"
    )
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text, flush=True)

with open(OUT_DIR / 'analysis_summary.txt', 'w') as f:
    f.write(summary_text + "\n")

print(f"\nAll results saved to {OUT_DIR}", flush=True)
