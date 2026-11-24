# Code–manuscript discrepancies in the explainability pipelines

| Field | Value |
| --- | --- |
| Date identified | Items 1–3: 2026-11-24, while writing `docs/ANALYSIS_PROTOCOL.md`. Item 4: 2026-11-24, while running Protocol §4.2–4.3. |
| Manuscript version checked | Submitted manuscript, 17 Oct 2026 (CHBR-D-26-02504): Sections 3.8–3.8.5 and Table 2 (items 1–3); Abstract, §4.6, Table 7, Figure 6, §5.4, §5.5, §6, Appendix Table S13 and the graphical abstract (item 4) |
| Code checked | Executed notebooks in `notebooks/core/` with their stored outputs |
| Effect on primary benchmark | None. All four affect only the explainability rankings (Table 7 / Figure 6) and how they are described. No AUC, DeLong, calibration or ablation value is affected. **Item 4 is the most severe: it changes a headline explainability claim.** |

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

---

## Summary

| # | Pipeline | Manuscript section(s) | Protocol §4 analysis resolves? | Text correction still needed? |
| --- | --- | --- | --- | --- |
| 1 | TabNet attention | §3.8.4, Table 2 | Partly (gives test-set ranking) | Yes: original ranking was training-set based |
| 2 | FT-Transformer KernelSHAP | §3.8.3, Table 2 | Partly (sample only) | Yes: background was random, not k-means; add `nsamples=100` |
| 3 | SAINT-inspired permutation | §3.8.5, Table 2, §3.10/S10 | Yes, for the new analysis | Yes: original used first 1,000 rows and an unseeded shuffle |
| **4** | **SAINT-inspired permutation — most severe** | **Abstract, Table 7, Figure 6, Kendall's W, four Spearman values (§4.6, Table S13), graphical abstract, §4.6, §5.4, §5.5, §6** | **Yes, for the new analysis** | **Yes: positions 6–12 mislabelled; withdraw the SAINT-divergence claim; W 0.657 → 0.838 with labels corrected** |

These corrections should be listed in the response to reviewers as author-identified corrections, separate from the reviewer-requested changes.
