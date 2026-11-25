# Code–manuscript discrepancies in the explainability pipelines

| Field | Value |
| --- | --- |
| Date identified | Items 1–3: 2026-11-24, while writing `docs/ANALYSIS_PROTOCOL.md`. Item 4: 2026-11-24, while running Protocol §4.2–4.3. Item 5: 2026-11-24, while preparing Protocol §5. Items 6–7: 2026-11-25, while compiling manuscript Appendix Table S16 from the executed notebooks. Item 8: 2026-11-25, while correcting the Table 2 KernelSHAP row (archive-wide search). Item 9: 2026-11-25, in the feasibility check for Protocol Amendment 12. Item 10: 2026-11-26, while preparing vector figures for submission. |
| Manuscript version checked | Submitted manuscript, 17 Oct 2026 (CHBR-D-26-02504): Sections 3.8–3.8.5 and Table 2 (items 1–3); Abstract, §4.6, Table 7, Figure 6, §5.4, §5.5, §6, Appendix Table S13 and the graphical abstract (item 4) |
| Code checked | Executed notebooks in `notebooks/core/` with their stored outputs |
| Effect on primary benchmark | Items 1–4 affect only the explainability rankings (Table 7 / Figure 6) and how they are described. Items 6 and 7 affect how the VIME-inspired and LightGBM models' **training is described**; their effect on test AUC is not quantified (item 6: not quantifiable from the deposited artefacts; item 7: not measured). Item 5 affects how the SAINT-inspired model's **training is described**, not any reported number: the reported SAINT AUC (0.7996) is correct for the deposited model as it was actually trained. Item 9 is a reproducibility gap in the ablation archive (no models or predictions kept); the refits under Amendment 12 reproduced every recorded ablation AUC exactly. Item 10 was a provenance gap for Figure 2 and Table 4: their 1,000-resample intervals were in no deposited output file. The manuscript now reports the deposited 2,000-resample intervals, which differ by at most 0.0008. No AUC, DeLong, calibration or ablation value is affected. **Item 4 is the most severe: it changes a headline explainability claim.** |

"Protocol §4" below means the common-sample explainability analysis prespecified in `docs/ANALYSIS_PROTOCOL.md` Section 4. All five explanation pipelines are recomputed on one shared stratified sample of n = 1,000 test records (sampling seed 42).

A point that applies to items 2 and 3: `train_test_split` shuffles by default. The "first N rows" of `X_test` are therefore a seed-42 random draw from the test set, not a block of consecutive source records. They are still not the stratified or explicitly drawn sample the text implies, and different pipelines used different N.

---

## 1. TabNet attention importance is computed on training data, not test data

| | |
| --- | --- |
| **Manuscript claims** | §3.8.4: "Global importance was computed as the mean normalized attention weight across all five steps and **all 10,281 test records** … following the pytorch-tabnet library implementation." Table 2 (TabNet native attention row): "… across 5 decision steps and **all 10,281 test records**, computed during forward propagation." |
| **Code does** | `TabNet_CVD_Clean.ipynb` cell 21 takes the ranking from `tabnet.feature_importances_`. pytorch-tabnet fills this attribute inside `fit()` from the training matrix passed to it. Cell 15 calls `tabnet.fit(X_train=X_train_sc, …)`, so the importances are aggregated over the **47,971 training records**. No test-set aggregation is used for the ranking. Cells 24–25 do call `tabnet.explain(X_test_sc[:500])`, but only for the step-wise heatmap figure, not the ranking. |
| **Verification (2026-11-24)** | Confirmed against the installed library in the original environment: pytorch-tabnet 4.1.0 `fit()` sets `self.feature_importances_ = self._compute_feature_importances(X_train)`. Applying that formula to the training set with the deposited model reproduces the notebook's stored TabNet importances on all 13 features to 4 decimals (Protocol §4.2 run; `revision-results/xai/pipeline_run_records.json`). |
| **Section(s) to correct** | §3.8.4; Table 2 (TabNet row); the Table 7 / Figure 6 caption if it describes the TabNet column as test-set attention |
| **Resolved by the Protocol §4 analysis?** | **Partly.** The Protocol §4 analysis computes TabNet attention with `tabnet.explain()` on the 1,000-record test sample, which gives a test-set ranking for the revised comparison. The manuscript text must still be corrected to say the *originally reported* TabNet ranking used training-set attention. The Protocol §4 analysis changes the new result; it does not make the old description true. |

## 2. FT-Transformer KernelSHAP uses the first 300 test rows and a random (not k-means) background

| | |
| --- | --- |
| **Manuscript claims** | §3.8.3: "All three used **300 test records** and **200 background samples from the training set selected by k-means summarisation**." Table 2 (SHAP KernelExplainer row): "300 test samples, **200 k-means background samples**." |
| **Code does** | `FT_Transformer_CVD.ipynb` cell 21: `np.random.seed(RANDOM_STATE); bg_idx = np.random.choice(len(X_train_sc), 200, replace=False)`. That is 200 training rows drawn **uniformly at random**; no `shap.kmeans` call is made. The explained rows are `X_test_sc[:300]`, the **first 300 rows** of the test set, not a separately drawn sample. `shap_values(..., nsamples=100)` is used, and the manuscript does not report `nsamples`. The TabNet KernelSHAP run (supplementary) prints SHAP's own warning suggesting `shap.kmeans`, which suggests it also used an unsummarised background. |
| **Section(s) to correct** | §3.8.3 (background method, row selection, add `nsamples = 100`); Table 2 (KernelExplainer row). Check that the supplementary TabNet and VIME KernelSHAP descriptions say the same thing. |
| **Resolved by the Protocol §4 analysis?** | **Partly.** The Protocol §4 analysis explains the shared stratified 1,000-record sample instead of the first 300 rows, which resolves the sample issue for the revised comparison. That analysis deliberately keeps the **executed** background procedure: 200 random training rows, seed 42, `nsamples = 100` (Protocol §4.2, row 4). So the k-means claim is not resolved by the Protocol §4 analysis. It is resolved only by correcting the text to describe random background sampling. |

## 3. SAINT-inspired permutation importance uses the first 1,000 test rows with no dedicated shuffle seed

| | |
| --- | --- |
| **Manuscript claims** | §3.8.5: "… on **a 1,000-record test sample** due to computational cost. Each feature was shuffled 10 times …" Table 2 (SAINT permutation row): "… across **a 1,000-record test-set sample**." |
| **Code does** | `SAINT_CVD_Complete.ipynb` cell 25: `X_importance = X_test_scaled[:n_samples]` with `n_samples = min(1000, len(X_test_scaled))`, i.e. the **first 1,000 rows** of the test set. Each repeat calls `np.random.shuffle(X_permuted[:, feat_idx])` with **no dedicated seed or generator**. It draws from the global NumPy stream seeded at the top of the notebook (cell 3), so the permutations depend on every random call executed before cell 25. The stored values can be reproduced only by re-running the whole notebook in the same order and environment. Inference used `batch_size = 256` in fixed row order. Because of intersample attention, the result is also conditional on that batching, which the manuscript already notes. |
| **Section(s) to correct** | §3.8.5; Table 2 (SAINT permutation row). Add a reproducibility caveat to §3.10 or Appendix Table S10 (implementation-fidelity audit). |
| **Resolved by the Protocol §4 analysis?** | **Yes for the revised analysis; text still needs correcting.** The Protocol §4 analysis uses the shared stratified 1,000-record sample and a dedicated generator, `np.random.default_rng(42)`, with batch size and row order fixed (Protocol §4.2, row 5). The original ranking in Table 7 / Figure 6 cannot be regenerated exactly from a dedicated seed. §3.8.5 should state how it was produced, and the response letter should present the Protocol §4 values as the reproducible replacement. |

## 4. SAINT-inspired importances were printed under the wrong feature names — MOST SEVERE OF THE FOUR

| | |
| --- | --- |
| **Severity** | **Most severe of the four.** It is the source of the manuscript's claim that the SAINT-inspired pipeline diverged from the other four, and of the published Kendall's W = 0.657. |
| **Manuscript claims** | The SAINT-inspired pipeline ranked Cholesterol 13th and Alcohol 4th, its correlations with the other four pipelines were weak (Spearman ρ = 0.187–0.330), and overall concordance was moderate (Kendall's W = 0.657). "The SAINT-inspired Cholesterol permutation estimate was −0.000842 ± 0.001302 and was not distinguishable from zero." |
| **Code does** | `SAINT_CVD_Complete.ipynb` builds the model's matrix in cell 10 with `FEATURE_COLS` (canonical order: …DBP, Cholesterol, Glucose, Smoking, Alcohol, Active, PP, MAP, RiskCount). Cell 25 (executed 16th) permutes the columns of that matrix but prints each result under `feature_cols` from cell 4 (executed 3rd), an earlier, superseded order (…DBP, PP, MAP, Cholesterol, Glucose, Smoking, Alcohol, Active, RiskCount). **Positions 6–12 carry the wrong names.** The model, its training and its predictions are unaffected (the deposited `proba_saint.npy` is reproduced exactly). |
| **Position → name mapping** | 6: printed "PP" = **Cholesterol** (+0.0388) · 7: "MAP" = **Glucose** (+0.0048) · 8: "Cholesterol" = **Smoking** (−0.0008) · 9: "Glucose" = **Alcohol** (+0.0002) · 10: "Smoking" = **Active** (+0.0028) · 11: "Alcohol" = **PP** (+0.0213) · 12: "Active" = **MAP** (+0.0076). Positions 1–5 and 13 are correctly named. Full table: `revision-results/xai/saint_label_correction.csv`. |
| **Evidence** | (a) The label printed at every position equals `feature_cols[k]` from cell 4. (b) The corrected importance vector, re-sorted, is exactly the set of originally reported values — only the names move. (c) The submitted Table 7 SAINT column equals the ranks of the mislabelled vector. (d) The canonical-order model, evaluated on the first 1,000 scaled test rows that cell 25 used, gives baseline AUC 0.81199, matching the notebook's printed 0.8120. (Amendment 3; `revision-results/xai/concordance_decomposition.json`.) |
| **Consequence** | Correctly labelled, the original SAINT ranking is Cholesterol, Age, SBP, PP, DBP… and agrees with the other pipelines. Kendall's W: (i) submitted 0.657 → (ii) original ranks with SAINT labels corrected **0.838** → (iii) common sample 0.836. The label correction accounts for the whole change (+0.181); the common sample changes W by −0.003. The four SAINT Spearman values change from 0.319 / 0.330 / 0.187 / 0.214 (with MLP / XGBoost / TabNet / FT-Transformer) to 0.951 / 0.819 / 0.791 / 0.753. |
| **Manuscript locations affected** | **Abstract** ("while the SAINT-inspired pipeline diverged (rho = 0.187-0.330); overall concordance was moderate (Kendall's W = 0.657)"). **Table 7** SAINT column (positions 6–12), its ‡ (Cholesterol) and † (Alcohol) markers and the Avg. Rank column. **Figure 6** heatmap and caption ("…SAINT-inspired divergence"). **Kendall's W** = 0.657, χ²(12) = 39.43 wherever reported. **The four SAINT Spearman values** (0.319, 0.330, 0.187, 0.214) in §4.6 and Appendix Table S13. **Graphical abstract** ("Kendall's W = 0.657", the mean ranks shown, and "Clay tick: SAINT-inspired cholesterol estimate not distinguishable from zero", which is Smoking's estimate). **§4.6** (four-pipeline cluster "with one clearly divergent pipeline"; average ranks MAP/PP 4.8 and Cholesterol 5.2; the −0.000842 ± 0.001302 Cholesterol estimate; "ranked Alcohol fourth"). **§5.4** ("weak correlations from SAINT-inspired permutation tests show that explanatory agreement was incomplete"). **§5.5** ("weak agreement between those four and the SAINT-inspired pipeline"). **§6 Conclusion** ("while the SAINT-inspired pipeline diverged (0.187-0.330); aggregate concordance was moderate (Kendall's W = 0.657)"). |
| **Also affected (repository, not modified)** | `results/tables/cross_pipeline_spearman_matrix.csv` in the v1.0.0 release carries the Spearman values derived from the mislabelled ranking. It is left unchanged under the protocol; the revision release should mark it as superseded. Supplementary §S5 also records that the original checkpoint metadata listed PP and MAP before Cholesterol and Glucose — the same superseded list. |
| **Resolved by the Protocol §4 analysis?** | **Yes for the revised analysis; every location above needs correcting.** The Protocol §4.2 SAINT pipeline uses the canonical feature order from the corrected checkpoint (verified by exact reproduction of `proba_saint.npy`). The claim that SAINT diverges must be withdrawn, not merely updated: the batch-dependence of SAINT probabilities remains a valid caveat, but it is not evidence of attribution divergence. |

## 5. SAINT-inspired training restores the final-epoch weights, not the best-epoch weights

| | |
| --- | --- |
| **Manuscript claims** | §3.7.5: "…a maximum of 50 epochs, **early-stopping patience 10**, and batch_size = 256." Stating early stopping with patience implies that the weights from the best validation epoch are restored. |
| **Code does** | `SAINT_CVD_Complete.ipynb` cell 19 saves the "best" weights with `best_model_state = model.state_dict().copy()`. `state_dict()` returns references to the live parameter tensors and `.copy()` copies only the dictionary, so the snapshot keeps changing as training continues. `model.load_state_dict(best_model_state)` therefore restores the **final-epoch** weights. (The FT-Transformer notebook uses `copy.deepcopy`, which is correct.) |
| **Evidence** | Cell 19's stored output: best validation AUC at **epoch 19 (0.8027)**; early stopping triggered after **epoch 29** (validation AUC 0.8020). The deposited checkpoint and `proba_saint.npy` (test AUC 0.7996) therefore come from the **epoch-29** weights. A direct test in the original environment confirms that a `state_dict().copy()` snapshot follows later in-place parameter updates. |
| **Effect** | Patience 10 means the restored weights are 10 epochs past the best validation epoch. The early-stopping rule still decided *when* training stopped, but not *which* weights were kept. The effect on test performance is quantified by the Protocol Amendment 4(b) diagnostic at seed 42 (best-epoch vs final-epoch test AUC; `revision-results/five_seed/five_seed_run_manifest.json`). |
| **Section(s) to correct** | §3.7.5 (state that the final-epoch weights were used, or that the best-weight restoration was ineffective); **Appendix Table S10** (implementation-fidelity audit), as a known deviation. |
| **Resolved by the Protocol analysis?** | **No change to the published model, by design.** Protocol Amendment 4(a) keeps the executed behaviour in the §5 five-seed analysis, so that only the seed varies and the seed-42 retrain stays comparable with the deposited arrays. The text must describe what was executed. |

## 6. VIME-inspired fine-tuning restores the final-epoch weights, not the best-epoch weights

| | |
| --- | --- |
| **Manuscript claims** | §3.7.5 (submitted) gave no early-stopping details for the VIME-inspired model. The revised Appendix Table S16 describes its fine-tuning schedule, which uses early stopping with patience 15 on validation AUC. Stating early stopping with patience implies that the weights from the best validation epoch are restored. |
| **Code does** | `VIME_Pretraining_CVD_FIXED.ipynb` cell 26 saves the "best" weights with `best_model_state = vime_model.state_dict().copy()` and later calls `vime_model.load_state_dict(best_model_state)`. As in item 5, `state_dict()` returns references to the live parameter tensors and `.copy()` copies only the dictionary, so the restored weights are the **final-epoch** weights. |
| **Evidence** | Cell 26's stored output: "Early stopping at epoch 63"; "Fine-tuning Complete! Best Val AUC: 0.8025". With `patience = 15` and the counter reset at each improvement, the last improvement, and therefore the best epoch, was **epoch 48**. The notebook does not print that epoch: it prints validation AUC only every tenth epoch (epoch 50: 0.8017; epoch 60: 0.8018). The **final-epoch (63) validation AUC was not logged.** The deposited VIME probabilities (test AUC 0.7990) therefore come from the epoch-63 weights. |
| **Effect** | Not quantifiable from the deposited artefacts: neither the epoch-63 validation AUC nor a best-epoch test AUC is recorded, and establishing it would require retraining, which was not done. |
| **Section(s) to correct** | §3.7.5 (state that the final-epoch weights were used); Appendix Table S16 (VIME-inspired row); Appendix A12.4. |
| **Resolved?** | **Text corrected, model unchanged.** The deposited model and all reported VIME results stand as executed. |

## 7. LightGBM early stopping monitored binary log-loss, not AUC

| | |
| --- | --- |
| **Manuscript claims** | §3.7.2 (submitted): the three gradient-boosting frameworks were trained "with early stopping monitored on the validation set. **The checkpoint at the best validation AUC iteration** was used for test evaluation." |
| **Code does** | `Stacking_Ensemble_CVD_GPU.ipynb` cell 16 builds `lgb.LGBMClassifier(...)` with **no `metric` parameter** and calls `lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])` with **no `eval_metric`**. LightGBM therefore evaluates only its objective's default metric, **binary log-loss**, and early stopping monitors that metric. XGBoost (cell 15: `eval_metric = 'auc'`, `early_stopping_rounds = 30`, `eval_set=[(X_val, y_val)]`) and CatBoost (cell 17: `eval_metric = 'AUC'`, `early_stopping_rounds = 30`, `eval_set=(X_val, y_val)`) did monitor validation AUC, as stated. |
| **Evidence** | The executed environment's `lightgbm` version is 4.6.0 (cell 3 output). Running the same `LGBMClassifier` configuration and `fit()` call under lightgbm 4.6.0 on synthetic data records only `binary_logloss` in `evals_result_` and `best_score_`, confirming the default. No benchmark model was refitted for this check. Cell 16's stored output: `Best iter=102`, which is the log-loss-selected iteration. |
| **Effect** | Not measured. An AUC-monitored run could select a different iteration; the LightGBM test AUC (0.7995) and the stacking tree count (103) derive from the log-loss-selected iteration. |
| **Section(s) to correct** | §3.7.2 (state per booster what was monitored); Appendix Table S16 (LightGBM row); Appendix A12.4. |
| **Resolved?** | **Text corrected, model unchanged.** |

## 8. A supplementary VIME-inspired KernelSHAP analysis was described but never performed

| | |
| --- | --- |
| **Manuscript claims** | §3.8 (submitted): "Additional KernelSHAP analyses for TabNet and the VIME-inspired model were provided as supplementary material". §3.8.3: KernelExplainer was applied "to TabNet and VIME as supplementary checks". Table 2 (KernelExplainer row): "TabNet, VIME (supplementary)". |
| **Code does** | Nothing. `VIME_Pretraining_CVD_FIXED.ipynb` contains no SHAP or KernelExplainer code and no SHAP output. |
| **Evidence** | **Not found** in an archive-wide search on 2026-11-25. The search covered: every tracked file name; the text of every tracked CSV, JSON, Markdown and log file; every cell and stored output of all 11 notebooks in `notebooks/core/`; and the 10 executed PDFs in `notebooks/executed-pdf/`. The only notebook hit was base64 image data. The authors' local project folder, including the members of every zip archive and the 26 August 2026 supplementary file, was also searched, with no hit. |
| **Effect** | None on any reported number: the VIME analysis was never part of the main average-rank table. The manuscript described an analysis that was not performed. |
| **Section(s) to correct** | §3.8, §3.8.3, Table 2 (remove the claim); Appendix A12.4 (record it). |
| **Resolved?** | **Yes: claim removed from the revised manuscript and recorded in A12.4.** |

---

## 9. The feature-ablation script deposited neither its fitted models nor its predictions

| | |
| --- | --- |
| **Manuscript claims** | Appendix A4 and Table S7 report the five guideline-defined ablation AUCs; Table S8 gives `results/tables/guideline_defined_feature_ablation.csv` as the deposited output. The manuscript does not claim that the ablation models or predictions were deposited. |
| **Code does** | `src/run_guideline_ablation.py` fits each Random Forest, computes `p = m.predict_proba(Xte)[:, 1]` in memory and writes only the AUC table. Neither the fitted models nor the test-set predictions are saved. |
| **Evidence** | The script itself; no ablation model or prediction array exists under `artifacts/` or `revision-results/` before Amendment 12. The deposited `artifacts/predictions/proba_rf.npy` is not the 13-feature ablation model: it is the Section 3 Random Forest fitted on scaled inputs (AUC 0.7995334583 against the table's 0.7995343667). The Section 3A(h) 11-feature model was deposited (`artifacts/models-revision/rf_11feat_seed42.joblib` with its scaler), but its predictions were not. |
| **Effect** | A reproducibility gap, not an error in any reported number. Confidence intervals for the ablation AUCs, requested by Reviewer 3, could not be computed from the archive as it stood. |
| **What was done** | Protocol Amendment 12 (prespecified, pushed alone in `c5f2041`) refitted all five ablation models with the script's seed, split and hyperparameters and scored the deposited 11-feature model. **All six reproduced their recorded AUCs exactly**, at full double precision for the five refits and for the 11-feature model against `pp_map_removed_rf_summary.json`. The six test-set probability arrays are now deposited in `revision-results/ablation/`, with the interval outputs. The fitted ablation models are not deposited (about 22 MB each); the arrays and `src/revision/amendment12_ablation_and_gender_ci.py` reproduce every reported quantity. |
| **Section(s) to correct** | None for the reported values. Appendix A4 and Table S7 now give the intervals and state that the models were refitted because the original run kept neither models nor predictions. |
| **Resolved?** | **Yes: arrays deposited under Amendment 12; every refit matched.** |

---

## 10. Figure 2 had no deposited script, and its plotted intervals are in no deposited output file

| | |
| --- | --- |
| **Manuscript claims** | Figure 2 shows each model's test-set AUC with a 95% bootstrap interval. §3.9 and the Table 4 caption state 1,000 resamples. The README lists the figure's source as "Saved predictions + bootstrap". |
| **Code does** | No deposited script or notebook draws Figure 2: its in-image title occurs in no file in the repository. The plotted 1,000-resample intervals survive only as printed output in `notebooks/core/Statistical_Significance_Tests.ipynb` (cell 9; 14 models to four decimals). That cell's stacking line is the original, pre-correction ensemble (0.8005 [0.7915–0.8093]). The corrected stacking interval that is plotted (0.791538–0.809095) appears only as text in `README.md` and `.zenodo.json`. |
| **Evidence** | Archive-wide search for the figure title. The manuscript figure was digitised from `results/figures/main/Figure_2_Corrected_AUC_95CI.png`, and its intervals match Table 4 to four decimals for all 15 models. |
| **Effect** | A reproducibility gap, not an error in a point estimate. The deposited intervals, Protocol §3A(b) in `revision-results/lowcompute/bootstrap_auc_by_model.csv` (2,000 resamples, seed 42), differ from the plotted 1,000-resample limits by at most 0.0008 (k-NN lower limit). The point AUCs are identical. |
| **What was done** | `src/revision/figure2_auc_ci.py` draws Figure 2 from `results/tables/validated_primary_metrics.csv` and `bootstrap_auc_by_model.csv` only. It stops if the two disagree on any AUC and asserts that every drawn point and interval equals the deposited value. Output: `results/figures/main/Figure_2_AUC_95CI.pdf` (vector, 190 mm wide, no in-image title). The submitted PNG is left unchanged as the record of the submitted figure. |
| **Section(s) corrected** | 2026-11-26: the authors adopted the deposited 2,000-resample intervals. The revised manuscript replaces the 1,000-resample values in Figure 2 (now drawn by `figure2_auc_ci.py`), all 15 rows of Table 4 and its caption ("1,000 resamples" → "2,000 resamples"), §3.9 ("1,000 test-set bootstrap resamples" → "2,000"), the abstract and §4.2 (stacking 0.7915–0.8091 → 0.7918–0.8089; §4.2 also MLP 0.7892–0.8065 → 0.7895–0.8064), and Appendix A8 ("interval for corrected stacking used 1,000 resamples" → "intervals for all 15 models used 2,000 resamples"). Largest change in any limit: 0.0008 (k-NN lower limit, 0.7281 → 0.7289). Point AUCs are unchanged. |
| **Resolved?** | **Yes: the manuscript now reports only deposited intervals.** |

---

## Summary

| # | Pipeline | Manuscript section(s) | Protocol §4 analysis resolves? | Text correction still needed? |
| --- | --- | --- | --- | --- |
| 1 | TabNet attention | §3.8.4, Table 2 | Partly (gives test-set ranking) | Yes: original ranking was training-set based |
| 2 | FT-Transformer KernelSHAP | §3.8.3, Table 2 | Partly (sample only) | Yes: background was random, not k-means; add `nsamples=100` |
| 3 | SAINT-inspired permutation | §3.8.5, Table 2, §3.10/S10 | Yes, for the new analysis | Yes: original used first 1,000 rows and an unseeded shuffle |
| **4** | **SAINT-inspired permutation — most severe** | **Abstract, Table 7, Figure 6, Kendall's W, four Spearman values (§4.6, Table S13), graphical abstract, §4.6, §5.4, §5.5, §6** | **Yes, for the new analysis** | **Yes: positions 6–12 mislabelled; withdraw the SAINT-divergence claim; W 0.657 → 0.838 with labels corrected** |
| 5 | SAINT-inspired training (benchmark model) | §3.7.5, Appendix Table S10 | No change, by design (Amendment 4(a)); effect quantified at seed 42 (4(b)) | Yes: final-epoch, not best-epoch, weights were restored |
| 6 | VIME-inspired training (benchmark model) | §3.7.5, Appendix Table S16 | No (effect not quantifiable without retraining) | Yes: final-epoch, not best-epoch, weights were restored |
| 7 | LightGBM training (benchmark model and stacking base learner) | §3.7.2, Appendix Table S16 | No (effect not measured) | Yes: early stopping monitored binary log-loss, not AUC |
| 8 | VIME-inspired KernelSHAP (claimed supplementary analysis) | §3.8, §3.8.3, Table 2 | Not applicable: no analysis exists | Yes: claim removed; recorded in Appendix A12.4 |
| 9 | Guideline-defined feature ablation (Random Forest) | Appendix A4, Table S7, Table S8 | Yes: Amendment 12 refitted the models (all matched) and deposited the predictions | No change to any value; A4 and Table S7 now state the refit and give the intervals |
| 10 | Figure 2 and Table 4 (AUC with 95% intervals) | Figure 2, Table 4, §3.9, §4.2, abstract, A8 | Yes: script deposited; the manuscript now reports the deposited 2,000-resample intervals | Done: 1,000-resample values replaced (largest change 0.0008) |

These corrections should be listed in the response to reviewers as author-identified corrections, separate from the reviewer-requested changes.
