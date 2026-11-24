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

## 2026-11-24 — Sections 3A(b)–3A(f)

- **Protocol commit:** `16108b4`; preceding results committed through `8f4bde9` (Section 3A(a)).
- **Scripts:** `src/revision/section3A_b_delong_effect_sizes.py`, `section3A_c_omnibus_roc.py`, `section3A_d_decision_curve.py`, `section3A_e_prevalence_transport.py`, `section3A_f_profile_collision.py`. Run in that order. `MODELS`, `delong_test`, `fast_delong` and `fdr_bh` are imported unchanged from `src/run_validated_stats.py`. All CSV/JSON outputs written with LF line endings; no `__pycache__` written.
- **Environment:** as Section 3A(a) (numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.18.1, matplotlib 3.11.2; Python 3.12.10, Windows). No GPU used.
- **3A(b):** 2,000 paired resamples, `default_rng(42)`, sequential draws, one shared index vector per replicate; 0 single-class replicates discarded. The Stacking Ensemble bootstrap CI (0.791751–0.808880) is identical to Section 3A(a), confirming both used the same replicates. DeLong p, BH q and ΔAUC reproduce the committed `results/tables/validated_delong_top6_bh.csv` and `validated_delong_vs_mlp_bh.csv` (max |Δp| 2.1×10⁻¹⁴, max |Δq| 4.4×10⁻¹⁵, ΔAUC exact). Pair order follows `run_validated_stats.py` (model A = higher primary AUC; vs-MLP family: model B = MLP).
- **3A(c):** one joint `fast_delong` call on all 15 models. Contrast row i = e_i − e_k against the last model in `MODELS` order (15-model test) and against the last of the six in `MODELS` order (top-six test). Both L S Lᵀ matrices full rank (14 and 5); **pseudo-inverse fallback not used**.
- **3A(d):** "exceeds both reference strategies" applied strictly (NB > treat-all and NB > 0) at each grid threshold; ranges are maximal runs of consecutive grid points. Below p_t ≈ 0.10 and above ≈ 0.85 the margins over the better reference strategy are ≤ 0.0005, so the fragmented edge ranges in `decision_curve_summary.csv` reflect near-ties, not meaningful benefit.
- **3A(e): AUC-unchanged check — 56 of 60 model × target cells pass (|Δ| < 1e-12); the 4 Naive Bayes cells do not (|Δ| = 1.542×10⁻⁵ at every target).** Cause, verified: the protocol's prescribed clip `np.clip(proba, 1e-6, 1 - 1e-6)` maps 344 Naive Bayes probabilities above 1 − 10⁻⁶ (320 distinct values) to a single value, creating ties; clipping alone changes the AUC by +1.542×10⁻⁵, while the logit shift without clipping leaves AUC exactly unchanged (Δ = 0). Decision Tree and k-NN also have values beyond the clip bounds, but they are exactly 0 or 1 (already tied), so clipping does not affect them. The method was not changed (the clip is prespecified); the failure is reported as found. PPV is undefined (NaN) where a model predicts no positives at threshold 0.5 after transport to low prevalence; this is expected behaviour, not an error.
- **3A(f):** cohort and test set built with the Section 3 functions (Sections 1.1–1.2); split identity check passed. Manuscript counts (3,333 shared profiles; 1,574 with both outcome labels) **reproduced exactly**. Grouping on the 10 recorded predictors gives the identical partition to the 13-feature grouping, as the protocol states.
- **Deviations / reruns:** none from the protocol. Committed after author review (commit message "Section 3A(b)-(f): …").

## 2026-11-24 — Section 3A(g): grouped haemodynamic permutation importance

- **Protocol commit:** `16108b4`; preceding results committed through `bd892aa` (Sections 3A(b)–(f)).
- **Script:** `src/revision/section3A_g_grouped_haemodynamic_permutation.py`.
- **Model:** `artifacts/models-revision/rf_seed42.joblib` (Section 3, variant C). **Data:** seed-42 `X_test_sc` (10,281 records) rebuilt with the Section 3 functions from `Data/Heart_disease_clean.csv` (same file and SHA-256 as Section 3); split identity check passed. Baseline AUC 0.7995334583, equal to the Section 3 retrained RF.
- **Environment:** as Section 3A(b)–(f); CPU (scikit-learn), per Section 6.
- **Loop order (protocol leaves the nesting open):** block — protocol code block exactly (`default_rng(42)`, 10 repeats). Single features — a fresh `default_rng(42)`, features in column order 0–12 as the outer loop and the 10 repeats as the inner loop, one `rng.permutation(n)` per feature per repeat (the nesting used by `sklearn.inspection.permutation_importance`). SD with ddof = 1.
- **Result:** block mean AUC drop 0.2129 (SD 0.0071); sum of the four single-feature drops (SBP 0.0538, MAP 0.0135, PP 0.0046, DBP 0.0025) = 0.0744, i.e. 34.9% of the block; block / sum = 2.86. Block ranks 1st among the block plus the nine non-haemodynamic features; Age 2nd; Cholesterol 3rd (also 3rd among the 13 single features).
- **Note:** RF predictions carry the ≤4.4×10⁻¹⁶ thread-order jitter recorded under Section 3 (`n_jobs=-1`); it does not affect the reported precision.
- **Deviations / reruns:** none from the protocol. Committed after author review, together with the Amendment 2 controls.

## 2026-11-24 — Section 3A(g), Amendment 2: independent and partly-redundant controls

- **Amendment:** Section 8, row 2, committed alone in `efb13a4` (committer time 2026-11-24T06:58:17Z) **before either control was run**.
- **Script:** `src/revision/section3A_g_controls.py`. Same Section 3 Random Forest, same seed-42 `X_test_sc`, 10 repeats. Each block permuted as one unit with the Section 3A(g) code, each with its own fresh `default_rng(42)`, so all three blocks receive the same ten row permutations. Single-feature importances recomputed with the Section 3A(g) nesting; they match `grouped_haemodynamic_permutation.csv` to 9.1×10⁻¹⁷. The haemodynamic block reproduces Section 3A(g) exactly (0.212913). Output file name `grouped_permutation_controls.csv` chosen here (the amendment names no file).
- **Redundancy (pairwise |Spearman ρ|, test set):** Control A max 0.170 (mean 0.077); Control B max 0.657 (mean 0.216); haemodynamic block max 0.921 (mean 0.700).
- **Result (block drop / sum of four singles / ratio):** Control A 0.0635 / 0.0610 / 1.04; Control B 0.0693 / 0.0614 / 1.13; haemodynamic block 0.2129 / 0.0744 / 2.86.
- **Deviations / reruns:** none. Committed after author review, together with Section 3A(g).
