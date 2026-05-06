"""
Phase D: Expand v5 checkpoint to v6.1 vocabulary.

v5  wte shape: (1537, 192)   [model IDs 0..1536]
v61 wte shape: (1537+N, 192) [model IDs 0..1536+N]

New token rows initialised as mean of ICD code embeddings (stored 9-1509)
plus small Gaussian noise (std=0.01, seed=42) — same strategy as v5→v6.

Saves to: Delphi/Delphi-main/checkpoints/mimic_v61_init/ckpt.pt
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


import sys, pickle
import numpy as np
import torch

DELPHI = DELPHI_DIR
PIPE   = PIPELINE_DIR

sys.path.insert(0, str(DELPHI))
sys.path.insert(0, str(PIPE / 'clinical_phenotyping/Phase_A'))

from model_v4 import DelphiV4, DelphiConfigV4
from phenotype_dict import PHENOTYPE_TOKENS, NEW_VOCAB_SIZE_STORED

V5_CKPT  = DELPHI / 'checkpoints/mimic_v5_phase2b/ckpt.pt'
OUT_DIR  = DELPHI / 'checkpoints/mimic_v61_init'
META_V61 = PIPE / 'data/mimic_data_v61/meta_v61.pkl'

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading v5 checkpoint: {V5_CKPT}", flush=True)
    ckpt = torch.load(str(V5_CKPT), map_location='cpu', weights_only=False)
    model_args = ckpt['model_args']

    old_vocab = model_args['vocab_size']          # model-space, should be 1537
    n_new     = len(PHENOTYPE_TOKENS)             # 33
    new_vocab  = NEW_VOCAB_SIZE_STORED + 1        # stored 1569 → model 1570

    print(f"  v5 vocab (model-space): {old_vocab}", flush=True)
    print(f"  Adding {n_new} phenotype tokens", flush=True)
    print(f"  v61 vocab (model-space): {new_vocab}", flush=True)

    # ── Expand embedding weight ───────────────────────────────────────────────
    sd = ckpt['model']
    wte_key = 'transformer.wte.weight'   # tied with lm_head.weight
    wte_old = sd[wte_key].float()        # (old_vocab, 192)
    n_embd  = wte_old.shape[1]

    # ICD code rows in model-space: stored 9-1509 → model 10-1510
    icd_mean = wte_old[10:1511].mean(0)  # (192,)

    rng = np.random.default_rng(seed=42)
    noise = torch.from_numpy(
        rng.normal(0, 0.01, (n_new, n_embd)).astype(np.float32)
    )
    new_rows = icd_mean.unsqueeze(0) + noise     # (n_new, 192)
    wte_new  = torch.cat([wte_old, new_rows], dim=0)  # (new_vocab, 192)

    assert wte_new.shape == (new_vocab, n_embd), \
        f"Expected ({new_vocab}, {n_embd}), got {wte_new.shape}"

    sd[wte_key]           = wte_new
    sd['lm_head.weight']  = wte_new   # weight tying

    # ── Update model_args ────────────────────────────────────────────────────
    model_args_new = dict(model_args)
    model_args_new['vocab_size'] = new_vocab

    # ── Verify by loading into DelphiV4 ──────────────────────────────────────
    conf  = DelphiConfigV4(**model_args_new)
    model = DelphiV4(conf)
    model.load_state_dict(sd)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_total/1e6:.3f}M  (added "
          f"{n_new * n_embd / 1e6:.3f}M for new embeddings)", flush=True)

    # Verify new rows are noise around ICD mean (not zero)
    new_check = wte_new[-n_new:]
    print(f"  New embedding rows  norm mean: {new_check.norm(dim=-1).mean():.4f}",
          flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_ckpt = {
        'model':         sd,
        'model_args':    model_args_new,
        'iter_num':      0,
        'best_val_loss': 1e9,
    }
    out_path = OUT_DIR / 'ckpt.pt'
    torch.save(out_ckpt, str(out_path))
    print(f"\nSaved → {out_path}", flush=True)

    # ── Print token table ─────────────────────────────────────────────────────
    print("\nNew token IDs (stored → model):")
    for name, sid, desc in PHENOTYPE_TOKENS:
        print(f"  stored {sid:4d} → model {sid+1:4d}  {name}")


if __name__ == '__main__':
    main()
