# Data Access

This pipeline requires two controlled-access datasets from PhysioNet.
No patient data is included in this repository.

---

## MIMIC-IV (structured EHR)

**PhysioNet page**: https://physionet.org/content/mimiciv/

1. Complete the CITI "Data or Specimens Only Research" training.
2. Sign the PhysioNet Credentialed Health Data License 1.5.0.
3. Download MIMIC-IV and set the environment variable:

```bash
export MIMIC_DATA_DIR=/path/to/mimic-iv
```

The preprocessing scripts expect the following tables under `$MIMIC_DATA_DIR/hosp/`:

```
admissions.csv.gz
patients.csv.gz
diagnoses_icd.csv.gz
procedures_icd.csv.gz
labevents.csv.gz
d_labitems.csv.gz
transfers.csv.gz
```

---

## MIMIC-IV-Note (discharge summaries)

**PhysioNet page**: https://physionet.org/content/mimic-iv-note/

Required for Stage 2 (NoteProjector) and Stage 3 (Delphi-Px phenotype extraction).

```bash
export MIMIC_NOTE_DIR=/path/to/mimic-iv-note
```

The scripts expect `$MIMIC_NOTE_DIR/note/discharge.csv.gz`.

---

## Citation

```bibtex
@article{johnson2023mimiciv,
  author  = {Johnson, Alistair E. W. and others},
  title   = {{MIMIC-IV}, a freely accessible electronic health record dataset},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {1},
  year    = {2023},
  doi     = {10.1038/s41597-022-01899-x}
}
```
