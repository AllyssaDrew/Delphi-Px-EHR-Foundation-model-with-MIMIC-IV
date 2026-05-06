"""
Bootstrap 95% CI for C-index and 30-day AUC (Weibull head).
1,000 stratified resamples on the test split.
Outputs: bootstrap_ci_results.csv
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


import sys, pickle, warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

DELPHI = DELPHI_DIR
PIPE   = PIPELINE_DIR
sys.path.insert(0, str(DELPHI))

from model_v4 import DelphiV4, DelphiConfigV4
from utils import get_batch_v4, get_p2i

OUT_DIR = PIPE / 'clinical_phenotyping' / 'Phase_E' / 'results' / 'wrapup'
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
N_BOOT  = 1000
RNG     = np.random.default_rng(42)

# ── Shared setup ──────────────────────────────────────────────────────────────
meta_v61 = pickle.load(open(PIPE / 'data/mimic_data_v61/meta_v61.pkl', 'rb'))
DEATH_STORED = 1510

splits_df = pd.read_csv(PIPE / 'data/mimic_data_v61/patient_splits.csv')
cuts_df   = pd.read_csv(PIPE / 'data/mimic_data_v61/test_cutoffs.csv')
TEST_SIDS = sorted(splits_df[splits_df['split'] == 'test']['subject_id'])
n_test    = len(TEST_SIDS)

sid_to_cut  = dict(zip(cuts_df['subject_id'], cuts_df['cutoff_age']))
sid_to_last = dict(zip(cuts_df['subject_id'], cuts_df['last_adm_age']))

fut_data = np.fromfile(str(PIPE / 'data/mimic_data_v61/test_future.bin'),
                       dtype=np.uint32).reshape(-1, 3)

te_death_age = {}
for row in fut_data:
    pidx, age_days, tok = int(row[0]), int(row[1]), int(row[2])
    if tok == DEATH_STORED and pidx < n_test:
        te_death_age[TEST_SIDS[pidx]] = age_days

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

labels_30d = (event_observed & (event_times <= 30)).astype(int)
print(f"Test set: {n_test} patients, {event_observed.sum()} deaths, "
      f"{labels_30d.sum()} 30-day deaths", flush=True)


# ── Model inference ───────────────────────────────────────────────────────────
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    conf = DelphiConfigV4(**ckpt['model_args'])
    m = DelphiV4(conf)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    m.load_state_dict(sd)
    m.eval().to(DEVICE)
    return m, conf


def infer_weibull(model, conf, data_dir, batch_size=64):
    test_bin = Path(data_dir) / 'test_input.bin'
    data = np.memmap(str(test_bin), dtype=np.uint32, mode='r').reshape(-1, 3)
    p2i  = get_p2i(data)
    n    = len(p2i)
    k_out   = np.zeros(n, dtype=np.float32)
    lam_out = np.zeros(n, dtype=np.float32)

    hidden_cache = {}
    def hook_ln_f(module, inp, out):
        hidden_cache['x'] = out
    handle = model.transformer.ln_f.register_forward_hook(hook_ln_f)

    for start in tqdm(range(0, n, batch_size), desc='  inference', leave=False):
        batch_pids = np.arange(start, min(start + batch_size, n)).tolist()
        with torch.no_grad():
            x_in, a_in, _, _, ov_x, ov_a = get_batch_v4(
                batch_pids, data, p2i,
                block_size=conf.block_size, n_summary=conf.n_summary,
                n_overflow=conf.n_overflow, device=DEVICE,
                no_event_token_rate=1, select='right',
            )
            model(x_in, a_in, overflow_idx=ov_x, overflow_age=ov_a)
            time_out = model.time_head(hidden_cache['x'])
            time_last = time_out[:, -1, :].cpu()
        k_out[start: start + len(batch_pids)]   = np.exp(time_last[:, 0].numpy())
        lam_out[start: start + len(batch_pids)] = np.exp(time_last[:, 1].numpy())

    handle.remove()
    return k_out, lam_out


# ── Metrics ───────────────────────────────────────────────────────────────────
def c_index(event_times, risk_scores, event_observed):
    et, rs, eo = (np.asarray(x) for x in (event_times, risk_scores, event_observed))
    concordant = discordant = tied = 0.0
    for i in np.where(eo)[0]:
        admissible = (et > et[i]) | ((et == et[i]) & (~eo))
        rj = rs[admissible]
        concordant += float((rs[i] > rj).sum())
        discordant += float((rs[i] < rj).sum())
        tied       += float((rs[i] == rj).sum())
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5


def auc_30d(prob_30d, labels_30d):
    from sklearn.metrics import roc_auc_score
    if labels_30d.sum() < 5:
        return float('nan')
    return roc_auc_score(labels_30d, prob_30d)


def bootstrap_metrics(k, lam, event_times, event_observed, labels_30d, n_boot):
    """Stratified bootstrap: maintain event rate in each resample."""
    n = len(k)
    # Precompute point estimates
    median_surv = lam * (np.log(2) ** (1.0 / k))
    risk        = 1.0 / np.clip(median_surv, 1e-6, None)
    prob30      = 1.0 - np.exp(-(30.0 / np.clip(lam, 1e-6, None)) ** k)

    ci_boot = np.zeros(n_boot)
    auc_boot = np.zeros(n_boot)

    event_idx   = np.where(event_observed)[0]
    censor_idx  = np.where(~event_observed)[0]
    n_ev  = len(event_idx)
    n_cen = len(censor_idx)

    for b in tqdm(range(n_boot), desc='  bootstrap', leave=False):
        # Stratified resample: keep same event/censored counts
        ev_samp  = RNG.choice(event_idx,  size=n_ev,  replace=True)
        cen_samp = RNG.choice(censor_idx, size=n_cen, replace=True)
        idx      = np.concatenate([ev_samp, cen_samp])

        ci_boot[b]  = c_index(event_times[idx], risk[idx], event_observed[idx])
        auc_boot[b] = auc_30d(prob30[idx], labels_30d[idx])

    return ci_boot, auc_boot


# ── Run ───────────────────────────────────────────────────────────────────────
rows = []
for model_name, ckpt_rel, data_dir in [
    ('v5 Phase 2b',    'checkpoints/mimic_v5_phase2b/ckpt.pt',   PIPE / 'data/mimic_data_v5'),
    ('v6.1 Phase D-F', 'checkpoints/mimic_v61_phase_f/ckpt.pt',  PIPE / 'data/mimic_data_v61'),
]:
    print(f"\n[{model_name}] Loading …", flush=True)
    model, conf = load_model(DELPHI / ckpt_rel)
    k, lam = infer_weibull(model, conf, data_dir)
    del model

    median_surv = lam * (np.log(2) ** (1.0 / k))
    risk        = 1.0 / np.clip(median_surv, 1e-6, None)
    prob30      = 1.0 - np.exp(-(30.0 / np.clip(lam, 1e-6, None)) ** k)

    pt_ci  = c_index(event_times, risk, event_observed)
    pt_auc = auc_30d(prob30, labels_30d)
    print(f"  Point: C-index={pt_ci:.4f}  30d-AUC={pt_auc:.4f}", flush=True)

    print(f"  Running {N_BOOT} bootstrap resamples …", flush=True)
    ci_boot, auc_boot = bootstrap_metrics(
        k, lam, event_times, event_observed, labels_30d, N_BOOT)

    rows.append({
        'model':         model_name,
        'c_index':       pt_ci,
        'c_index_lo':    float(np.percentile(ci_boot, 2.5)),
        'c_index_hi':    float(np.percentile(ci_boot, 97.5)),
        'auc_30d':       pt_auc,
        'auc_30d_lo':    float(np.percentile(auc_boot[~np.isnan(auc_boot)], 2.5)),
        'auc_30d_hi':    float(np.percentile(auc_boot[~np.isnan(auc_boot)], 97.5)),
        'k_mean':        float(k.mean()),
        'lam_mean':      float(lam.mean()),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'bootstrap_ci_results.csv', index=False)
print(f"\nSaved {OUT_DIR}/bootstrap_ci_results.csv")
print(df.to_string(index=False))
