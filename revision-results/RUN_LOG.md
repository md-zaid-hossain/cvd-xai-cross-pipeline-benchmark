# Revision analysis run log

Per `docs/ANALYSIS_PROTOCOL.md` Section 6.

## 2026-11-24 — Section 3: retraining fidelity gate

- **Protocol commit:** `16108b4` (locking commit `4366b20`, pushed to `origin/main`).
- **Script:** `src/revision/section3_retraining_fidelity.py`.
- **Data file:** `Data/Heart_disease_clean.csv` (outside the repository; patient-level CSVs are not redistributed, per `data/README.md`). SHA-256 `8b1322989836cf85444ba7b693c28cd9a43188730b380cbe301da75d6569eb5f`, which matches the documented stage-1 hash in `data/README.md` exactly. Shape (68653, 11), target distribution 34,706 / 33,947, matching the stored notebook output.
- **Split identity check (Section 1.2):** PASS — regenerated `y_test` (10,281 records) equals `artifacts/predictions/y_test.npy` element-wise.
- **Environment:**
  - Python 3.12.10, Windows-11-10.0.26300-SP0.
  - numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2 (pinned to the original-environment versions in `environment/ORIGINAL_ENVIRONMENT.md`, per instruction, so the comparison tests the pipeline rather than a library upgrade).
  - xgboost 3.1.3 (no original-environment XGBoost version is recorded anywhere in the repository; this is the version pinned in `environment/requirements-corrections.txt` for the correction environment, used here for lack of a documented original pin — recorded as a deviation, not matched to "original").
  - GPU: NVIDIA GeForce GTX 1650 (driver 576.02, CUDA 12.9) — the same GPU model used for the original notebook runs (per `environment/ORIGINAL_ENVIRONMENT.md` and the manuscript).
- **Device policy (Section 6):** Random Forest and MLP on CPU (scikit-learn); XGBoost on GPU (`device='cuda'`), per "prefer, for each model, the device the original used for that model."
- **Result:** all three models PASS the Section 3 fidelity gate (|ΔAUC| ≤ 0.001 and correlation ≥ 0.999). Random Forest variant C passed on the first attempt; variant S was not fitted. XGBoost `best_iteration` = 185, matching the published value exactly. Full results in `revision-results/fidelity/retrain_fidelity_seed42.csv` and `revision-results/fidelity/rf_variant_check.csv`.
- **Rerun (same day, before push):** the first run wrote its two CSV outputs with Windows CRLF line endings (pandas `to_csv` default on Windows), which conflicts with `.gitattributes` (`*.csv eol=lf`) and made the checksums fail in a fresh LF clone. The script was changed to write CSVs with `lineterminator="\n"` (no other change) and rerun. All results are identical to the first run.
- **Finding from the rerun: Random Forest predictions vary in the last floating-point bit between runs.** The fitted RF is deterministic (`rf_seed42.joblib` is byte-identical across runs), but with the protocol's `n_jobs=-1` scikit-learn sums the 300 trees' probabilities across threads in a varying order. Between the two runs, 2,428 of 10,281 test probabilities differ, by at most 4.4×10⁻¹⁶; the AUC (0.7995334583278106) and the comparison with the deposited array (max |Δp| 4.4×10⁻¹⁶) are identical. Predicting the same model with `n_jobs=1` gives bit-identical arrays. This also explains the 4.4×10⁻¹⁶ residual against the deposited array. `n_jobs=-1` is kept because the protocol fixes it (Section 2.1); `proba_rf_retrained_seed42.npy` is therefore reproducible to floating-point precision, not bit-for-bit. All other outputs (MLP, XGBoost, the scaler and all model files) were byte-identical across runs.
- **Deviations:** none from the protocol. No amendment required.
- **Outputs:** listed with SHA-256 in `revision-results/MANIFEST_SHA256.txt` (9 files: 4 under `artifacts/models-revision/`, 5 under `revision-results/fidelity/`).

## 2026-11-24 — Section 3A(a): stacking correction impact (before vs after)

- **Protocol commit:** `16108b4`; Section 3 results committed in `e885309`.
- **Script:** `src/revision/section3A_a_stacking_before_after.py` (imports `MODELS` and `delong_test` from `src/run_validated_stats.py` unchanged; writes no `__pycache__`).
- **Inputs:** deposited arrays only (`artifacts/predictions/`). Identity check: `proba_stack_primary_original.npy` and `pred_stack_primary_original.npy` are byte-identical to their `_stack_original` counterparts (recorded in the output).
- **Provenance of "original":** `proba_stack_original.npy` reproduces the original `Stacking_Ensemble_CVD_GPU.ipynb` stored output (cells 27 and 41) on all five reported metrics (AUC 0.8005, accuracy 0.7335, F1 0.7171, precision 0.7546, recall 0.6831); that notebook saved it as `proba_stack.npy` (cell 43). The "0.8011" stacking AUC in `TabNet_CVD_Clean.ipynb` cell 41 is a hand-typed placeholder table ("Update these with your actual values…"), not the output of any run, and is not used.
- **Environment:** as Section 3, plus matplotlib 3.11.2 and scipy 1.18.1 (matplotlib is imported by `run_validated_stats.py` but not used by DeLong).
- **Bootstrap:** 2,000 paired resamples, `numpy.random.default_rng(42)`, sequential draws, one shared index vector per replicate (Section 3A(b)); 0 single-class replicates discarded.
- **Result:** corrected − original ΔAUC = −0.000110 (95% bootstrap CI −0.000820 to +0.000566; DeLong z = −0.308, p = 0.758). Both versions rank first among the 15 models; top-six membership unchanged. Full output: `revision-results/lowcompute/stacking_before_after.csv`.
- **Note:** the corrected stacking AUC bootstrap CI here (0.791751–0.808880, 2,000 resamples, seed 42) differs slightly from the published CI (0.791538–0.809095), which used a different resample count and seed; this is expected and is not a discrepancy.
- **Deviations / reruns:** none. Committed after author review (commit message "Section 3A(a): stacking before/after correction …").
