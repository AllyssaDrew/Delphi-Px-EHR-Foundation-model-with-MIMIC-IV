# Delphi-Px: Discrete Phenotype Tokens for EHR Mortality Prediction

Code for the paper **"Discrete Phenotype Tokens Bridge Clinical Notes and EHR Sequences for Trustworthy Mortality Prediction"**.

We extend the [Delphi](https://github.com/gerstung-lab/DELPHI) EHR sequence model with 33 discrete clinical phenotype tokens extracted from MIMIC-IV discharge summaries. The tokens are inserted directly into the patient event timeline, preserving the model's causal event interface while adding auditable note-derived clinical states.

---

## Model Overview

| Model | Description | Death AUC |
|---|---|---|
| **Delphi-Base** | Structured-only EHR sequence model (5.83 M params, 1 537 tokens) | 0.680 |
| **Delphi-Px** | Delphi-Base + 33 discrete clinical phenotype tokens (5.84 M params, 1 570 tokens) | **0.721** |
| Logistic Regression baseline | Sex + age + ICD features | 0.558 |
| Histogram GBT baseline | Sex + age + ICD features | 0.561 |

All results are on the MIMIC-IV held-out test split (*N* = 9 915). External validation is required before clinical deployment.

---

## Repository Structure

```
mimic_pipeline/
├── preprocessing/          # Stage 1 – MIMIC-IV → Delphi binary sequences
│   ├── 02_preprocess_v5.py         ← Delphi-Base dataset (main)
│   ├── 04_make_mimic_labels.py
│   ├── legacy/02_preprocess.py     ← v1–v4 provenance scripts
│   └── ...
├── configs/
│   ├── final/                      ← production training configs
│   │   ├── train_delphi_mimic_v5_phase2b.py    ← Delphi-Base
│   │   ├── train_delphi_mimic_v6_phase_e_v3.py ← Delphi-Px D-E warmup
│   │   └── train_delphi_mimic_v6_phase_f_v3.py ← Delphi-Px D-F joint ft
│   └── legacy/                     ← historical/ablation configs
├── multimodal_notes/       # Stage 2 – Continuous note-fusion branch (NoteProjector)
│   ├── Phase_A/run_phase_a.py          ← discharge-note linkage
│   ├── Phase_B/run_phase_b_shard.py    ← Clinical-Longformer embeddings
│   ├── Phase_C/run_phase_c.py          ← insert NOTE token into sequences
│   └── Phase_E–H/                      ← NoteProjector training iterations
├── clinical_phenotyping/   # Stage 3 – Discrete phenotype token model (Delphi-Px)
│   ├── Phase_A/phenotype_dict.py       ← 33-token dictionary + regex patterns
│   ├── Phase_A/extract_phenotypes.py   ← single-note extraction function
│   ├── Phase_B/run_extraction.py       ← parallel extraction over all notes
│   ├── Phase_C/build_v61_dataset.py    ← insert phenotype events into sequences
│   ├── Phase_D/expand_vocab_v5_to_v61.py  ← expand checkpoint vocab 1537→1570
│   ├── Phase_D/train_phase_d_e.sbatch  ← D-E embedding warmup (1 000 steps)
│   ├── Phase_D/train_phase_d_f.sbatch  ← D-F joint fine-tune (3 000 steps)
│   └── Phase_E/                        ← evaluation + SHAP + figures
├── explainability/         # SHAP and UMAP analyses
├── evaluation/             # ML baselines, survival metrics, bootstrap CIs
├── figures/                # Manuscript figure reproduction
├── slurm/                  # HPC job scripts (preprocess / train / eval / figures)
└── docs/data_access.md     # MIMIC-IV access instructions
```

---

## Environment

```bash
conda create -n delphi python=3.11
conda activate delphi
pip install -r requirements.txt
```

The Delphi backbone model code must be cloned separately:

```bash
git clone https://github.com/gerstung-lab/DELPHI.git
```

Set environment variables before running any script:

```bash
export DELPHI_PROJECT_ROOT=/path/to/your/project   # parent of mimic_pipeline/ and DELPHI/
export MIMIC_DATA_DIR=/path/to/mimic-iv            # MIMIC-IV root (contains hosp/, icu/)
export MIMIC_NOTE_DIR=/path/to/mimic-iv-note       # MIMIC-IV-Note root (contains note/)
export DELPHI_DIR=${DELPHI_PROJECT_ROOT}/DELPHI/Delphi-main
export MIMIC_PIPELINE_DIR=${DELPHI_PROJECT_ROOT}/mimic_pipeline
export PYTHON=$(which python)                       # or full path to conda env python
```

---

## Data Access

MIMIC-IV and MIMIC-IV-Note are controlled-access datasets. See [`docs/data_access.md`](docs/data_access.md) for PhysioNet credentialing instructions. Raw patient data cannot be redistributed.

---

## Reproduction

### Stage 1 – Delphi-Base structured dataset

```bash
python preprocessing/02_preprocess_v5.py
python preprocessing/04_make_mimic_labels.py
```

### Stage 2 – Continuous note-fusion branch (NoteProjector — abandoned)

This branch is documented for reproducibility. It extracts Clinical-Longformer embeddings and inserts a dense NOTE token into the event sequence. The approach was ultimately abandoned due to representation collapse (pairwise cosine similarity 0.9977) and no net gain in ICD prediction.

```bash
python multimodal_notes/Phase_A/run_phase_a.py     # discharge-note linkage
sbatch multimodal_notes/Phase_B/phase_b_array.sbatch   # Longformer inference (GPU)
python multimodal_notes/Phase_B/merge_shards.py
python multimodal_notes/Phase_C/run_phase_c.py     # build v6 dataset
```

### Stage 3 – Delphi-Px discrete phenotype token model

```bash
# 1. Extract phenotype tokens from all discharge summaries
python clinical_phenotyping/Phase_B/run_extraction.py

# 2. Build Delphi-Px dataset (insert phenotype events into sequences)
python clinical_phenotyping/Phase_C/build_v61_dataset.py

# 3. Expand Delphi-Base checkpoint vocabulary 1537 → 1570
python clinical_phenotyping/Phase_D/expand_vocab_v5_to_v61.py

# 4. Phase D-E: embedding warmup (backbone frozen, 1 000 steps, ~5 min)
sbatch clinical_phenotyping/Phase_D/train_phase_d_e.sbatch

# 5. Phase D-F: joint fine-tune (all params, 3 000 steps, ~10 min)
sbatch clinical_phenotyping/Phase_D/train_phase_d_f.sbatch
```

### Stage 4 – Evaluation and explainability

```bash
sbatch clinical_phenotyping/Phase_E/evaluate_v61.sbatch
sbatch clinical_phenotyping/Phase_E/shap_v61.sbatch
python clinical_phenotyping/Phase_E/plot_shap_v61.py

python evaluation/analysis_wrapup.py    # ML baselines + survival metrics
python evaluation/bootstrap_ci.py       # bootstrap CIs for C-index and 30-day AUC
python figures/05_figures.py            # manuscript figures
```

---

## Phenotype Token Dictionary

33 discrete tokens across four clinical domains:

| Domain | Count | Example tokens |
|---|---|---|
| Cancer status & treatment | 14 | `CANCER_STAGE_IV`, `CANCER_METASTATIC`, `CHEMO_RECEIVED` |
| Psychiatric crisis | 8 | `SUICIDAL_IDEATION_PRESENT`, `PSYCHOSIS_ACTIVE`, `PSYCHIATRIC_HOLD` |
| Substance use | 4 | `ALCOHOL_WITHDRAWAL_ACTIVE`, `NALOXONE_ADMINISTERED` |
| Critical care | 7 | `COMFORT_MEASURES_ONLY`, `DNR_PRESENT`, `ICU_ADMISSION`, `SEPSIS_PRESENT` |

Extraction uses section-aware regex with negation handling and temporal-intent guards. See [`clinical_phenotyping/Phase_A/phenotype_dict.py`](clinical_phenotyping/Phase_A/phenotype_dict.py).

Top SHAP contributors to Death prediction (mean |SHAP|):

| Token | Mean SHAP |
|---|---|
| `COMFORT_MEASURES_ONLY` | +0.408 |
| `ICU_ADMISSION` | +0.330 |
| `INTUBATED_DURING_STAY` | +0.328 |
| `DNR_PRESENT` | +0.312 |
| `SEPSIS_PRESENT` | +0.241 |

---

## Hardware

- **Preprocessing**: 30 CPU workers, ~188 seconds for phenotype extraction over 331 793 notes
- **Training**: single NVIDIA L40S (46 GB VRAM); Phase D-E < 5 min, Phase D-F < 10 min
- **Bootstrap CI**: ~2 hours on CPU (1 000 resamples)

---

## Citation

```bibtex
@inproceedings{anonymous2026delphipx,
  title     = {Discrete Phenotype Tokens Bridge Clinical Notes and {EHR} Sequences
               for Trustworthy Mortality Prediction},
  author    = {ganzeyu},
  booktitle = {BIOS740},
  year      = {2026}
}
```

---

## License

Code: MIT License. See [LICENSE](LICENSE).

Data: MIMIC-IV is governed by the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/). No patient data is included in this repository.
