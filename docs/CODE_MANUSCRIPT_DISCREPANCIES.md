# Code–manuscript discrepancies in the explainability pipelines

| Field | Value |
| --- | --- |
| Date identified | 2026-11-24, while writing `docs/ANALYSIS_PROTOCOL.md` |
| Manuscript version checked | Submitted manuscript, 17 Oct 2026 (CHBR-D-26-02504), Sections 3.8–3.8.5 and Table 2 |
| Code checked | Executed notebooks in `notebooks/core/` with their stored outputs |
| Effect on primary benchmark | None. All three affect only the explainability rankings (Table 7 / Figure 6) and how they are described. No AUC, DeLong, calibration or ablation value is affected. |

"Protocol §4" below means the common-sample explainability analysis prespecified in `docs/ANALYSIS_PROTOCOL.md` Section 4. All five explanation pipelines are recomputed on one shared stratified sample of n = 1,000 test records (sampling seed 42).

A point that applies to items 2 and 3: `train_test_split` shuffles by default. The "first N rows" of `X_test` are therefore a seed-42 random draw from the test set, not a block of consecutive source records. They are still not the stratified or explicitly drawn sample the text implies, and different pipelines used different N.

---

## 1. TabNet attention importance is computed on training data, not test data

| | |
| --- | --- |
| **Manuscript claims** | §3.8.4: "Global importance was computed as the mean normalized attention weight across all five steps and **all 10,281 test records** … following the pytorch-tabnet library implementation." Table 2 (TabNet native attention row): "… across 5 decision steps and **all 10,281 test records**, computed during forward propagation." |
| **Code does** | `TabNet_CVD_Clean.ipynb` cell 21 takes the ranking from `tabnet.feature_importances_`. pytorch-tabnet fills this attribute inside `fit()` from the training matrix passed to it. Cell 15 calls `tabnet.fit(X_train=X_train_sc, …)`, so the importances are aggregated over the **47,971 training records**. No test-set aggregation is used for the ranking. Cells 24–25 do call `tabnet.explain(X_test_sc[:500])`, but only for the step-wise heatmap figure, not the ranking. |
| **Caveat** | The notebooks did not record the pytorch-tabnet version. This was checked against the library's documented `fit()` behaviour (`compute_importance=True` aggregates over `X_train`). Confirm against the installed version before the response letter is finalised. |
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

---

## Summary

| # | Pipeline | Manuscript section(s) | Protocol §4 analysis resolves? | Text correction still needed? |
| --- | --- | --- | --- | --- |
| 1 | TabNet attention | §3.8.4, Table 2 | Partly (gives test-set ranking) | Yes: original ranking was training-set based |
| 2 | FT-Transformer KernelSHAP | §3.8.3, Table 2 | Partly (sample only) | Yes: background was random, not k-means; add `nsamples=100` |
| 3 | SAINT-inspired permutation | §3.8.5, Table 2, §3.10/S10 | Yes, for the new analysis | Yes: original used first 1,000 rows and an unseeded shuffle |

These corrections should be listed in the response to reviewers as author-identified corrections, separate from the reviewer-requested changes.
