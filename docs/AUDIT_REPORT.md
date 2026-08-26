# Packaging and reproducibility audit

Audit date: 2026-08-26

## Authoritative evidence

- 10 core notebooks parse as valid notebook JSON.
- All core notebooks contain stored execution outputs.
- No core notebook contains a stored exception.
- Most core notebooks include one unexecuted empty trailing cell; this does not affect the stored run.
- The saved test target and every model probability/prediction array contain 10,281 observations on the common held-out test set.
- Recomputed AUC, accuracy, F1, precision, recall, and Brier score values agree with `validated_primary_metrics.csv` to machine precision.

## Corrections preserved in the release

- Corrected stacking AUC: `0.800367` rather than the earlier mismatched-configuration result.
- The statistical implementation uses tied-value midranks and paired DeLong covariance; it does not fall back to an independent-variance calculation.
- Calibration uses exact saved primary probabilities.
- Feature ablation is guideline-defined rather than circularly selected.
- VIME and SAINT claims are relabelled as inspired implementations.
- SAINT-inspired batch-size and row-order sensitivity is reported.

## Issues found and handled

1. The supplied correction scripts used absolute `/mnt/data/...` paths. The release versions accept command-line paths and use repository-relative defaults.
2. The supplied statistics script expected `proba_stack.npy`, while the corrected artifact is `proba_stack_corrected.npy`. The portable release explicitly maps the Stacking Ensemble to the corrected file.
3. The supplied final-package checksum list referred to earlier manuscript filenames and did not describe the current archive. A new deterministic release manifest replaces it.
4. The supplied `Heart_disease_clean.csv` is the 68,653-row stage-1 file, while the analytical cohort is 68,534 after a second BMI/pulse-pressure check. The new data guide and preparation script make both stages explicit.
5. The source dataset's redistribution licence could not be verified from the available Kaggle record. Patient-level CSV files are excluded from the public release.
6. Several notebooks in `coding.zip` are incomplete, include hard-coded local paths, or contain stored errors. They are isolated under `archive/legacy-notebooks/` and are not part of the evidentiary workflow.

## Legacy notebook status

| Notebook | Executed code cells | Stored errors | Status |
| --- | ---: | ---: | --- |
| `Copy of Final Updated Code.ipynb` | 25 / 60 | 2 | Incomplete; provenance only |
| `Copy of Final code of Heart disease.ipynb` | 18 / 29 | 0 | Partial; provenance only |
| `Final Updated Code.ipynb` | 17 / 46 | 0 | Partial; provenance only |
| `New Begining.ipynb` | 35 / 35 | 1 | Contains stored exception |
| `New app.ipynb` | 2 / 2 | 0 | Small prototype |
| `Tasting models Accuracy.ipynb` | 5 / 5 | 0 | Small exploratory check |

## Remaining author actions

- Add ORCID identifiers if available.
- Add the final GitHub URL and Zenodo DOI after repository creation and release.
- Confirm each author approves the MIT/CC BY 4.0 mixed licensing.
- Confirm journal policy before publicly posting the manuscript DOCX.
- Complete any required TRIPOD+AI and PROBAST+AI forms.
