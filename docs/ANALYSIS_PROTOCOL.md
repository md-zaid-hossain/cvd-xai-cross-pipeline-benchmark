# Prespecified analysis protocol: journal revision (CHBR-D-26-02504, round 1)

| Field | Value |
| --- | --- |
| Protocol version | 1.0 |
| Date written | 2026-11-24 |
| Lock timestamp | The header date 2026-11-24 is Bangladesh local time (UTC+6). The locking commit `4366b206485c34b29776e4d3ca58eddc53766d88` has committer timestamp **2026-11-23T21:34:22Z**, i.e. 2026-11-24 03:34 UTC+6. Both refer to the same moment. |
| Status | **Locked when committed.** No analysis in this document had been run when it was written. |
| Applies to | Revision analyses requested by the editor and Reviewers 1–3 (decision dated 2026-11-20) |
| Baseline release | `cvd-xai-benchmark` v1.0.0 (Zenodo 10.5281/zenodo.22114347) |
| Sources read | `src/run_validated_stats.py`; `src/run_corrected_stacking.py`; `src/run_saint_sensitivity.py`; `notebooks/core/*.ipynb`; manuscript Sections 3.5–3.9 (submitted version, 17 Oct 2026) |

## 0. Purpose and locking rule

This protocol fixes every analytical choice for the revision analyses **before any of them is run**. The commit that adds this file is the lock. Its hash and timestamp will be quoted in the response to reviewers.

- Analyses run exactly as written here. Any later change is recorded in Section 8 (Amendment log) with a date and reason. The amendment is committed **before** the affected analysis runs. Results from both the original plan and the amended plan are reported.
- Every prespecified output (Section 7) is reported in the revised manuscript or the supplement. This holds whether the result supports the manuscript's claims or weakens them. No output is dropped for being unfavourable.
- The primary benchmark (the deposited 15-model probability arrays and `results/tables/validated_primary_metrics.csv`) is **not** replaced by any analysis here. The new analyses are secondary.
- The protocol covers eleven analysis blocks:
  - the retraining fidelity gate (Section 3);
  - eight low-compute analyses (Section 3A, (a)–(h)): stacking correction impact, DeLong effect sizes, omnibus ROC test, decision-curve analysis, prevalence transport, the exact-profile collision bound, grouped haemodynamic permutation, and the PP/MAP-removed Random Forest importance check;
  - the common-sample explainability analysis (Section 4);
  - the five-seed stability analysis (Section 5).

  Any analysis not listed in Sections 3–5 is outside this protocol and needs an amendment before it runs.

## 1. Data, cohort and primary split (as executed)

### 1.1 Cohort

These steps are copied from `notebooks/core/Classical_Models_CVD.ipynb` cells 5–9 and are identical in `src/run_corrected_stacking.py`:

1. Load the stage-1 file (`Heart_disease_clean.csv` in the notebooks; `data/local/Heart_disease_clean_stage1.csv` in `src/`). It has 68,653 rows and 11 columns.
2. Target: `target = (df['cardiovascular disease'] == 'has cardiovascular disease').astype(int)`
3. Keep rows with `BMI.between(10, 60)` and `(SBP - DBP) > 0`, then `reset_index(drop=True)`. This leaves **68,534** rows.
4. Engineer the derived features:
   - `PP = SBP - DBP`
   - `MAP = (SBP + 2*DBP) / 3`
   - `risk_count = (cholesterol>1) + (glucose>1) + smoking + alcohol + (1 - active)`
5. The feature matrix uses this exact column order (13 features):

```python
FEATURES = [
    'Gender (0: female, 1: male)', 'age(years)', 'BMI',
    'ap_hi(Systolic blood pressure)', 'ap_lo(Diastolic blood pressure)',
    'cholesterol(1: normal, 2: above normal, 3: well above normal)',
    'Glucose(1: normal, 2: above normal, 3: well above normal)',
    'smoking', 'alcohol', 'active', 'PP', 'MAP', 'risk_count'
]
FEAT_NAMES = ['Gender','Age','BMI','SBP','DBP','Cholesterol','Glucose',
              'Smoking','Alcohol','Active','PP','MAP','RiskCount']
X = df[FEATURES].values
y = df['target'].values
```

### 1.2 Stratified 70/15/15 split (base seed 42)

Copied verbatim from `Classical_Models_CVD.ipynb` cell 11. The same code appears in the Stacking, TabNet, FT-Transformer and SAINT notebooks and in `src/run_corrected_stacking.py`:

```python
RANDOM_STATE = 42
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_tmp
)
# Scale — fitted on train only
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
```

- This gives Train = 47,971 (70.0%), Validation = 10,282 (15.0%) and Test = 10,281 (15.0%). These are the counts in the stored notebook output and manuscript Section 3.5.
- Step 2 uses `test_size=0.1765`, which is 0.15/0.85 rounded. It does not use exactly 0.15/0.85.
- **Split identity check (mandatory for every seed-42 run):** the regenerated `y_test` must equal `artifacts/predictions/y_test.npy` element-wise. This is the same `assert` used in `src/run_corrected_stacking.py`. If the check fails, the run stops.
- Inputs by model family (manuscript Section 3.6): RF, MLP, TabNet, FT-Transformer and SAINT-inspired use the scaled matrices. XGBoost, LightGBM and CatBoost use the raw matrices. The corrected stacking script uses raw matrices for all four base learners.

## 2. Published configurations of models to be retrained

Random Forest, MLP and XGBoost have **no saved artefact** in `artifacts/models/`. That folder contains only FT-Transformer, SAINT-inspired, TabNet and VIME-inspired checkpoints. These three models must be retrained before the analyses in Sections 3A(g)–(h), 4 and 5. The configurations below are copied verbatim from the executed notebook code that wrote the deposited arrays. Each one was checked against manuscript Sections 3.7.1 and 3.7.2 and matches.

### 2.1 Random Forest

Source: `Classical_Models_CVD.ipynb` cell 17. That notebook writes `proba_rf.npy` in cell 38. Manuscript text (3.7.1): *"n_estimators = 300, max_depth = 10, min_samples_leaf = 5, max_features = 'sqrt', n_jobs = −1, fixed random state."*

```python
rf_model = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 10,
    min_samples_leaf = 5,
    max_features     = 'sqrt',
    random_state     = RANDOM_STATE,
    n_jobs           = -1
)
rf_model.fit(X_train_sc, y_train)
```

Note: `Stacking_Ensemble_CVD_GPU.ipynb` cells 18 and 43 fit an identically configured RF on `X_train_sc` and also write `proba_rf.npy`. The fidelity gate (Section 3) determines whether the retrained RF matches the deposited array.

### 2.2 MLP

Source: `Classical_Models_CVD.ipynb` cell 19, which writes `proba_mlp.npy` in cell 38. Manuscript text (3.7.1): *"hidden_layer_sizes = (16, 16), activation = 'relu', L2 regularisation alpha = 1×10⁻⁵, Adam optimiser, learning_rate_init = 0.001, max_iter = 500, early_stopping = True, validation_fraction = 0.1, fixed random state."*

```python
mlp_model = MLPClassifier(
    hidden_layer_sizes = (16, 16),
    activation         = 'relu',
    alpha              = 1e-5,      # L2 regularisation
    learning_rate_init = 0.001,
    solver             = 'adam',
    max_iter           = 500,
    early_stopping     = True,
    validation_fraction= 0.1,
    random_state       = RANDOM_STATE
)
mlp_model.fit(X_train_sc, y_train)
```

`early_stopping=True` with `validation_fraction=0.1` makes the MLP hold out 10% of `X_train_sc` internally. It does not use `X_val`.

### 2.3 XGBoost

Source: `Stacking_Ensemble_CVD_GPU.ipynb` cell 15, which writes `proba_xgb.npy` in cell 43. Manuscript text (3.7.2): *"n_estimators = 500, max_depth = 5, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 3, gamma = 0.1, reg_alpha = 0.1, reg_lambda = 1.5, eval_metric = 'auc', early_stopping_rounds = 30, tree_method = 'hist', device = 'cuda'. Best iteration = 185."*

```python
xgb_model = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 5,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    gamma                 = 0.1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.5,
    eval_metric           = 'auc',
    early_stopping_rounds = 30,
    random_state          = RANDOM_STATE,
    verbosity             = 0,
    # ── GPU params ──────────────────────────────────────────
    tree_method           = 'hist',          # required for GPU
    device                = 'cuda' if GPU_AVAILABLE else 'cpu'
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

The original run used `device='cuda'` on a GTX 1650. Section 3 retrains XGBoost on the GPU where available (Section 6, device policy) and records the device used. Retraining on another device may cause small numerical differences from the deposited array (see Section 3).

## 3. Retraining fidelity gate (runs first, before Sections 3A(g)–(h), 4 and 5)

1. Retrain RF, MLP and XGBoost at seed 42 using Sections 1–2.
2. For each model, compare the retrained test probabilities with the deposited `artifacts/predictions/proba_{rf,mlp,xgb}.npy`. Report:
   - maximum absolute difference
   - Pearson correlation
   - deposited AUC and retrained AUC, and their difference
   - for XGBoost, `best_iteration` (published value: 185)
   - the device and library versions used for each model (Section 6, device policy)
3. Decision rule, fixed now:
   - **Pass:** |ΔAUC| ≤ 0.001 and correlation ≥ 0.999. The retrained model is used for Sections 3A(g)–(h), 4 and 5.
   - **Fail:** the retrained model is still used, because no deposited model object exists. The discrepancy is reported in the supplement and named in the response to reviewers.
   - The deposited arrays stay the primary benchmark either way.
4. **Random Forest: two possible sources.** `proba_rf.npy` is written by two notebooks:
   - `Classical_Models_CVD.ipynb` cell 38, from the RF fitted in cell 17 (variant C);
   - `Stacking_Ensemble_CVD_GPU.ipynb` cell 43, from the RF fitted in cell 18 (variant S).

   The deposited array may come from either. The gate first uses variant C (Section 2.1). If variant C fails, variant S is fitted exactly as in its notebook cell, on that notebook's own split and scaler code, and also compared with the deposited array. The result table records the max |Δp|, correlation and ΔAUC of each variant, and which one matches (it may be neither or both). The RF is declared a discrepancy only if **both** variants fail. If variant S passes and variant C does not, variant S is the retrained RF used in Sections 3A, 4 and 5.
5. The retrained objects are saved under `artifacts/models-revision/` (Section 7) so later analyses do not retrain again.

## 3A. Low-compute analyses on deposited arrays

Analyses (a)–(e) use only the deposited files in `artifacts/predictions/` (`y_test.npy` plus the `proba_*`/`pred_*` arrays). Analysis (f) also needs the local analytical cohort (Section 1.1). Analysis (g) needs the cohort and the retrained RF from Section 3, so it runs after Section 3. Analysis (h) needs the cohort, the Section 3 RF and the Section 4.1 common sample, so it runs after both of those.

Conventions for all of 3A:
- **The 15 models** and their array suffixes are the `MODELS` dictionary in `src/run_validated_stats.py`. Its order is also the model order for every matrix below. The Stacking Ensemble means `proba_stack_corrected.npy` throughout, except in (a).
- **The two comparison families** are exactly as in `src/run_validated_stats.py`:
  - **top-six family:** the six highest primary AUCs, i.e. Stacking Ensemble, XGBoost, CatBoost, FT-Transformer, SAINT-inspired and Random Forest (15 pairs);
  - **vs-MLP family:** XGBoost, LightGBM, CatBoost, Stacking Ensemble, TabNet, FT-Transformer, VIME-inspired SSL, SAINT-inspired and Random Forest, each against MLP (9 comparisons).
- **DeLong:** the midrank implementation `fast_delong` / `delong_test` in `src/run_validated_stats.py`, used unchanged.
- **Sign:** ΔAUC = AUC(Model A) − AUC(Model B). Model A has the higher primary AUC, except in the vs-MLP family, where Model B is always MLP.
- **Prevalence:** π_s = `y_test.mean()`, the observed test-set prevalence. It is about 0.4944, matching the 49.44% cohort prevalence reported in the manuscript.

### 3A(a) Stacking correction impact (before vs after)

- **Inputs:** `proba_stack_original.npy` / `pred_stack_original.npy` (original notebook stacking, mismatched configuration) against `proba_stack_corrected.npy` / `pred_stack_corrected.npy`. `proba_stack_primary_original.npy` is byte-identical to `proba_stack_original.npy`, so it is not analysed separately; the identity is checked and recorded.
- **Reported for each version:**
  - AUC with 95% CI from the 3A(b) bootstrap (same 2,000 resamples);
  - accuracy, F1, precision and recall at the 0.5 threshold;
  - Brier score;
  - calibration intercept and slope (as in `run_validated_stats.py`);
  - rank among the 15 models when that version stands in for the Stacking Ensemble.
- **Reported for the comparison:**
  - ΔAUC (corrected − original) with its paired bootstrap 95% CI (3A(b) method);
  - DeLong z and p for the pair;
  - mean and max |Δp|, and the Pearson correlation of the two probability vectors;
  - the number of records whose 0.5-threshold class changes;
  - whether membership of the top six changes.

### 3A(b) DeLong effect sizes with bootstrap CIs

- **Families:** the top-six family (15 pairs) and the vs-MLP family (9 pairs), as defined above.
- **Point estimate:** ΔAUC on the full 10,281-record test set.
- **Primary interval:** a paired percentile bootstrap with **2,000 resamples and seed 42**:

```python
rng = np.random.default_rng(42)
n = len(y_test)                       # 10,281
for b in range(2000):                 # sequential draws, one vector per replicate
    idx = rng.integers(0, n, size=n)  # same idx used for every model in replicate b
    # AUC of every model on (y_test[idx], proba[idx]); ΔAUC_b for every pair
ci = np.percentile(delta_b, [2.5, 97.5])   # numpy default (linear) interpolation
```

  One resample vector per replicate is shared by all models and pairs, so every comparison is paired. A replicate containing only one class is discarded and the count reported (expected to be zero at n = 10,281 with prevalence ≈ 0.49). The same 2,000 replicates give the per-model AUC CIs used in 3A(a).
- **Also reported:**
  - the DeLong-based 95% CI, ΔAUC ± 1.959964 × √(DeLong variance of the difference);
  - DeLong z and two-sided p;
  - the existing within-family Benjamini–Hochberg q.
- **Interpretation:** a CI that includes 0 is not evidence of equivalence. No equivalence margin is prespecified, so no equivalence claim will be made.

### 3A(c) Omnibus ROC test

- **H₀:** all 15 AUCs are equal.
- **Estimates:** θ is the 15-vector of AUCs, and S is the 15 × 15 DeLong covariance matrix. Both come from a single joint call to `fast_delong` on all 15 probability vectors (rows in `MODELS` order, records ordered by `np.argsort(-y_test)`, as in `delong_test`).
- **Contrast:** L is 14 × 15, with row i = e_i − e_15 (each model minus the last model in `MODELS` order). The χ² statistic does not depend on which full-rank contrast is used.
- **Statistic:** χ² = (Lθ)ᵀ (L S Lᵀ)⁻¹ (Lθ), with df = rank(L S Lᵀ) and p = `scipy.stats.chi2.sf(chi2, df)`.
- **Rank-deficient fallback:** the rank is computed with `np.linalg.matrix_rank` at its default tolerance. If it is below 14, the Moore–Penrose pseudo-inverse `np.linalg.pinv` replaces the inverse and df = that rank. The rank, the smallest eigenvalue and the condition number of L S Lᵀ are reported either way.
- **Secondary:** the same test within the top-six family (5 × 6 contrast, df ≤ 5).
- **Prespecified interpretation:** the 15-model omnibus is expected to reject. The low-tier models (e.g. k-NN, AUC 0.739) are far from the top tier, so rejection says nothing about the top tier. The top-six omnibus is the test relevant to the non-resolvability claim. Non-rejection there is not evidence of equivalence.

### 3A(d) Decision-curve analysis

- **Threshold grid:** p_t from 0.05 to 0.95 in steps of 0.01 (91 thresholds):

```python
pt_grid = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
```

- **Classification:** a record is treated as positive when `proba >= p_t`.
- **Net benefit** (Vickers & Elkin, 2006), with n = 10,281:

  NB(p_t) = TP/n − (FP/n) × p_t / (1 − p_t)

- **Reference strategies:**
  - treat-all: NB = π_s − (1 − π_s) × p_t / (1 − p_t);
  - treat-none: NB = 0.
- **Models:** all 15, using the deposited probabilities at the observed test prevalence π_s. Net benefit at other prevalences is covered in 3A(e).
- **Reported:**
  - the full NB table;
  - for each model, the threshold ranges where NB exceeds both treat-all and treat-none;
  - the maximum NB difference between any two top-six models across the grid.
- **No CIs:** net benefit is reported as point estimates, and this is stated in the text.
- **Figure:** the top six, treat-all and treat-none.

### 3A(e) Prevalence transport

- **Targets:** π_t ∈ {0.05, 0.10, 0.20, 0.4944}. The 0.4944 target is a near-identity check.
- **Method:** prior-probability (logit-intercept) adjustment under the label-shift assumption, i.e. class-conditional feature distributions are unchanged. This is **not resampling and not case deletion**. Every one of the 10,281 test records is retained.

```python
p   = np.clip(proba, 1e-6, 1 - 1e-6)
p_adj = expit(logit(p) + logit(pi_t) - logit(pi_s))
```

- **Evaluation:** the test set is analytically reweighted to the target prevalence, with weights w = π_t/π_s for y = 1 and w = (1 − π_t)/(1 − π_s) for y = 0. Reported per model and target, for all 15 models:
  - AUC;
  - weighted Brier score;
  - weighted calibration-in-the-large (weighted mean of p_adj vs π_t);
  - weighted calibration intercept and slope (weighted logistic regression of y on logit(p_adj), `C=np.inf`);
  - sensitivity, specificity, PPV and NPV, classifying with p_adj at threshold 0.5 and at threshold = π_t (PPV and NPV weighted to π_t);
  - transported net benefit on the 3A(d) grid, classifying with p_adj. Here NB = sens × π_t − (1 − spec) × (1 − π_t) × p_t/(1 − p_t), and treat-all NB = π_t − (1 − π_t) × p_t/(1 − p_t).
- **AUC is unchanged by design.** The adjustment is a monotone transform of p, and class-level weights cancel in the AUC. Equality with the primary AUC (|Δ| < 1e-12) is checked and reported. It is not a finding.
- **Interpretation:** these results describe behaviour under a change in prevalence alone. They are not external validation, and they assume the original probabilities were calibrated at π_s.

### 3A(f) Exact-profile collision bound

- **Data:** the full 68,534-record analytical cohort from Section 1.1 (primary) and the seed-42 test set of 10,281 records (secondary).
- **Profile:** the exact tuple of the 13 feature values in the Section 1.1 column order, with no rounding or binning. PP, MAP and `risk_count` are deterministic functions of the other ten features, so this grouping is identical to grouping on the 10 recorded predictors.
- For each profile g, n₁_g and n₀_g are the counts of positive and negative records.
- **Reported:**
  - the number of records;
  - the number of distinct profiles;
  - the number of profiles with ≥ 2 records ("shared");
  - the number of profiles containing both labels ("conflicting");
  - the number of records in conflicting profiles.
  - **The collision bound:** B = Σ_g min(n₁_g, n₀_g) / N. This is the minimum fraction of records that **any** classifier depending only on these 13 features must misclassify on this sample. So 1 − B is the maximum achievable accuracy on this sample.
  - **The AUC ceiling:** 1 − 0.5 × Σ_g n₁_g n₀_g / (N₁ N₀). Positive–negative pairs that share a profile can at best tie.
- **This is NOT the Bayes error,** and the text will say so explicitly:
  - it is computed in-sample and depends on the sample;
  - it counts only exact duplicates and ignores irreducible uncertainty among records with distinct but similar profiles;
  - it depends on how coarsely the features were recorded (for example, ordinal cholesterol and glucose).

  It is a floor for this dataset's recorded representation, not an estimate of population-level irreducible error.
- **Check against the manuscript:** the manuscript reports 3,333 shared profiles and 1,574 with both outcome labels. Whether these counts are reproduced is recorded. If they are not, both sets of counts are reported with the grouping difference explained.

### 3A(g) Grouped haemodynamic permutation importance

- **Model:** the retrained Random Forest from Section 3 (the variant selected by the gate).
- **Data:** the full seed-42 test set `X_test_sc` (10,281 records).
- **Baseline:** the AUC on the unpermuted data.
- **Block:** {SBP, DBP, PP, MAP}, i.e. column indices 3, 4, 10 and 11 in the Section 1.1 order. The block is permuted **as one unit**: the same row permutation is applied to all four columns, so their within-record relationships are kept.

```python
rng = np.random.default_rng(42)
for r in range(10):
    perm = rng.permutation(n)
    Xp = X_test_sc.copy()
    Xp[:, [3, 4, 10, 11]] = X_test_sc[perm][:, [3, 4, 10, 11]]
    drop[r] = auc_baseline - roc_auc_score(y_test, rf.predict_proba(Xp)[:, 1])
```

- **Reported:** the mean and SD (ddof = 1) of the AUC drop over the 10 repeats.
- **Comparison:** single-feature permutation importance for all 13 features on the same model and data, with 10 repeats and a fresh `np.random.default_rng(42)`. Features are processed in column order 0–12, one permutation per feature per repeat.
- **Also reported:**
  - the block importance against the sum of the four single-feature importances;
  - the rank of the block among the block plus the nine non-haemodynamic features. This shows whether, for example, Cholesterol moves up once the four correlated pressure variables are treated as one unit.

### 3A(h) PP/MAP-removed importance check (Random Forest only)

Answers Reviewer 2, comment 4 ("a quick check without PP and MAP"). **Random Forest only.** It is not extended to MLP or XGBoost.

- **Feature set (11):** the full 13-feature matrix of Section 1.1 minus **PP and MAP only**, keeping `risk_count`, in this order:

```python
FEATURES_11 = [
    'Gender (0: female, 1: male)', 'age(years)', 'BMI',
    'ap_hi(Systolic blood pressure)', 'ap_lo(Diastolic blood pressure)',
    'cholesterol(1: normal, 2: above normal, 3: well above normal)',
    'Glucose(1: normal, 2: above normal, 3: well above normal)',
    'smoking', 'alcohol', 'active', 'risk_count'
]
```

- **Relation to Table S7:** this subset deliberately differs from the "Original 10 variables" row of manuscript Table S7 (AUC 0.799843), which also drops `risk_count`. No row of Table S7 matches this feature set, so the 11-feature AUC is a **new** value. That is expected, not an inconsistency. Table S7 reports AUC only; this analysis adds the feature ranking.
- **Split and fit:** the seed-42 split of Section 1.2, made with the identical two `train_test_split` calls on the 11-column matrix. The row assignment depends only on n, `y` and the seed, so it is the same as the primary split. This is checked: `y_test` must equal `artifacts/predictions/y_test.npy`, and the 11 shared columns of the test rows must equal those of the 13-feature `X_test`. A `StandardScaler` is fitted on the 11-feature training set only. The Random Forest uses the Section 2.1 configuration with `random_state=42`, fitted on the scaled 11-feature training set.
- **Discrimination:** test-set AUC on all 10,281 records, reported next to the Table S7 values for context (10 variables 0.799843; 12 variables 0.799302; 13 variables 0.799534). No pass/fail threshold applies.
- **Importance:** run on the Section 4.1 common sample (`revision-results/xai/xai_common_sample_idx.npy`, read from file, never redrawn), for both RF models:

```python
permutation_importance(rf_11, X11_test_sc[xai_idx], y_test[xai_idx],
                       scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1)
permutation_importance(rf_13, X_test_sc[xai_idx], y_test[xai_idx],      # Section 3 retrained RF
                       scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1)
```

  Reported per feature: mean AUC drop and SD over the 10 repeats (`importances_std`, as returned by scikit-learn).
- **Ranks:** `scipy.stats.rankdata(-mean, method='average')`, reported three ways:
  - (i) the rank in the 11-feature model (of 11);
  - (ii) the rank in the 13-feature model (of 13);
  - (iii) the rank in the 13-feature model among the 11 shared features only.
- **The prespecified question: does Cholesterol rise in rank once PP and MAP are removed?** The comparison is (iii) against (i), so that removing two features does not by itself count as a rise. "Rises" means a strictly smaller rank number in the 11-feature model. The answer (yes / no / unchanged) is stated explicitly. The same comparison is reported for every feature.
- **Order:** this runs after Section 3, which provides the 13-feature RF. It also runs after the Section 4.1 common sample has been drawn and its SHA-256 recorded. The sample is drawn independently of any result, so drawing it before this analysis does not break Section 4's rule that the sample is fixed before any explainer runs.

## 4. Common-sample explainability (XAI) analysis

Addresses Reviewer 2, comment 3 (unequal sample sizes across pipelines), and partly Reviewer 3 (separating method effects from model effects).

### 4.1 The fixed sample

- **n = 1,000** records, drawn once from the 10,281-record primary test set, stratified on `y_test`.
- **Sampling seed: 42.** This is the base seed, chosen so no new seed is introduced.
- Exact draw:

```python
idx_all = np.arange(len(y_test))           # 0..10280, positions in the seed-42 test set
xai_idx, _ = train_test_split(idx_all, train_size=1000,
                              stratify=y_test, random_state=42)
xai_idx = np.sort(xai_idx)
np.save('revision-results/xai/xai_common_sample_idx.npy', xai_idx)
```

- The SHA-256 of `xai_common_sample_idx.npy` and the sample's class balance are written to `xai_common_sample_manifest.json`. This happens **before** any explainer runs. All five pipelines read this one file. No pipeline draws its own sample.
- Rows are always passed in ascending `xai_idx` order. This matters for SAINT (Section 4.2, pipeline 5).

### 4.2 The five pipelines on the common sample

Method settings follow the executed notebooks. Anything that cannot be kept the same is stated.

| # | Pipeline | Model source | Input | Settings |
| --- | --- | --- | --- | --- |
| 1 | MLP permutation importance | Retrained MLP (Section 2.2) | `X_test_sc[xai_idx]` | `sklearn.inspection.permutation_importance`, `scoring='roc_auc'`, `n_repeats=10`, `random_state=42` (as in `Classical_Models_CVD.ipynb` cell 35) |
| 2 | XGBoost TreeSHAP | Retrained XGBoost (Section 2.3) | `X_test[xai_idx]` (raw) | `shap.TreeExplainer(xgb_model).shap_values(...)`; importance = mean \|SHAP\| (as in `Stacking_Ensemble_CVD_GPU.ipynb` cells 29 and 31) |
| 3 | TabNet attention | `artifacts/models/tabnet_cvd_model.zip` | `X_test_sc[xai_idx]` | `explain_matrix, _ = tabnet.explain(X)`; importance = `explain_matrix.sum(axis=0)`, normalised to sum to 1. This is the pytorch-tabnet `feature_importances_` formula applied to the sample. |
| 4 | FT-Transformer KernelSHAP | `artifacts/models/ft_transformer_cvd.pt` | `X_test_sc[xai_idx]` | `shap.KernelExplainer` on class-1 probability. Background is 200 training rows drawn by `np.random.seed(42); np.random.choice(len(X_train_sc), 200, replace=False)`, with `nsamples=100` (as in `FT_Transformer_CVD.ipynb` cell 21). Importance = mean \|SHAP\| for class 1. |
| 5 | SAINT-inspired permutation | `artifacts/models/saint_cvd_model_metadata_corrected.pt`, canonical feature order as in `src/run_saint_sensitivity.py` | `X_test_sc[xai_idx]` | Change in AUC when one feature is shuffled. `n_repeats=10`. Permutations drawn from `np.random.default_rng(42)`. Inference `batch_size=256` in fixed row order (as in `SAINT_CVD_Complete.ipynb` cell 25) |

### 4.3 Rankings and concordance

- Features are ranked 1 (most important) to 13 within each pipeline. Ties get average ranks (`scipy.stats.rankdata(-importance, method='average')`).
- Concordance measures:
  - pairwise Spearman ρ for all 10 pipeline pairs
  - Kendall's W across the 5 pipelines, with tie correction
  - each pipeline's top-3 feature set
- **The common-sample analysis is the PRIMARY reported concordance result.** The original unequal-sample analysis (the submitted Table 7 / Figure 6) becomes a **sensitivity analysis**. The revised Abstract, Table 7 and Figure 6 will report the common-sample values: rankings, pairwise Spearman ρ and Kendall's W. The original values move to the supplement, labelled as the unequal-sample sensitivity analysis.
- The supplement reports the two sets of rankings side by side, with the change in each feature's rank. The finding that SBP and Age are the most consistently prioritised features is restated only if the primary (common-sample) result supports it.
- Permutation pipelines (1 and 5) also report the standard deviation across the 10 repeats. Reviewer 3 asked for this.

### 4.4 Gaps between the executed code and the manuscript text (found while writing this protocol)

These gaps will be corrected in the revised manuscript text. They are recorded here so the correction is visibly not driven by results.

| Item | Manuscript (submitted text) | Executed code |
| --- | --- | --- |
| KernelSHAP background (3.8.3) | "200 background samples … selected by k-means summarisation" | 200 rows sampled uniformly at random (`np.random.choice`, seed 42); no k-means |
| KernelSHAP explained rows | "300 test records" | The **first** 300 rows of `X_test_sc` (`X_test_sc[:300]`), not a random sample; `nsamples=100` not stated in text |
| SAINT permutation sample (3.8.5) | "a 1,000-record test-set sample" | The **first** 1,000 rows of `X_test_scaled`; shuffles use the global NumPy RNG with no dedicated seed |
| TabNet attention (3.8.4) | Mean attention "across … all 10,281 test records" | Ranking taken from `tabnet.feature_importances_`, which pytorch-tabnet computes from the **training** matrix passed to `fit()`. Verify against the installed pytorch-tabnet version. |
| Figure 4 caption | "Ties were tied with midranks" | Reviewer 2, comment 7: change to "Ties were handled using midranks" |

## 5. Five-seed stability analysis

Addresses Reviewer 2, comment 6. It extends the published split-sensitivity analysis (manuscript Section 4.5 / Table S8). That analysis covered CatBoost, XGBoost, LightGBM, RF and MLP; its seeds and per-run outputs were not retained.

### 5.1 Design

- **Models (6):** Stacking Ensemble (corrected), XGBoost, CatBoost, FT-Transformer, SAINT-inspired, Random Forest.
- **Seeds (5):** 42, 101, 202, 303, 404, reduced to three (42, 101, 202) only under the compute-budget rule below. No other seeds will be run.
- **What a seed controls:** seed `s` replaces `RANDOM_STATE` everywhere:
  - both `train_test_split` steps in Section 1.2, so each seed gives a new stratified 70/15/15 split and the scaler is refit on that seed's training set;
  - every model's random state;
  - for Stacking, the `StratifiedKFold(random_state=s)` and fold-model seeds `s + fold`, mirroring `make_models(R + fold)` in `src/run_corrected_stacking.py`;
  - for PyTorch models, `np.random.seed(s)`, `torch.manual_seed(s)` and `torch.cuda.manual_seed_all(s)`.
- **Hyperparameters unchanged. No retuning for any seed.** Each model keeps its published training procedure. Early stopping is part of that procedure, not tuning, and is applied on the seed's own validation split:
  - XGBoost: Section 2.3, early stopping 30 rounds on `X_val`.
  - Random Forest: Section 2.1.
  - CatBoost: `Stacking_Ensemble_CVD_GPU.ipynb` cell 17, verbatim: `iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3, bagging_temperature=0.5, eval_metric='AUC', early_stopping_rounds=30, random_state=RANDOM_STATE, verbose=0, task_type='GPU' if GPU_AVAILABLE else 'CPU'`, fit with `eval_set=(X_val, y_val)`.
  - LightGBM (standalone; needed only to set the stacking iteration count, not ranked): `Stacking_Ensemble_CVD_GPU.ipynb` cell 16, verbatim: `n_estimators=500, num_leaves=63, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1, device='gpu' if GPU_AVAILABLE else 'cpu'`, fit with `eval_set=[(X_val, y_val)]` and `callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]`. Its test AUC is recorded but not ranked.
  - **Stacking: iteration counts derived once per seed, then fixed across all folds.** The counts are **not** carried over from seed 42 (186 / 103 / 217) to the other seeds. Those values were chosen by early stopping on the seed-42 validation set, so reusing them at other seeds would bring in information from a different split. It would also be inconsistent with the standalone models, which early-stop on their own seed's validation set. At each seed `s`, including 42:
    1. **Derive the counts once.** Fit the standalone XGBoost (Section 2.3), LightGBM (above) and CatBoost (above) with early stopping on seed `s`'s own validation split. These XGBoost and CatBoost fits are the same ones used as ranked models at that seed.
    2. **Apply the deposited convention unchanged:** tree count = `best_iteration + 1` for all three base learners. That is XGBoost `best_iteration + 1`, LightGBM `best_iteration_ + 1` and CatBoost `best_iteration_ + 1`, exactly as `src/run_corrected_stacking.py` did. At seed 42 this gives 186 / 103 / 217 if the original early-stopping results are reproduced. The five-seed analysis changes only the seed, not the counting convention.
    3. **Hold the three counts fixed** in all five out-of-fold folds and in the final refit for that seed. **No early stopping is done inside any OOF fold.** Fold-wise early stopping would give each fold a different count and bring back the OOF / final-refit specification mismatch that the corrected stacking removed.
    4. Build the stacking ensemble exactly as `src/run_corrected_stacking.py` does, apart from those three counts: same `params` otherwise, RF 300 trees, `StratifiedKFold(n_splits=5, shuffle=True, random_state=s)`, fold-model seeds `s + fold`, final refit at seed `s`, and meta-learner `LogisticRegression(C=1, max_iter=2000, random_state=s)`.
  - **Known, immaterial deviation, kept unchanged:** LightGBM's `best_iteration_` counts from 1 (`lightgbm.train` stores the 0-based index + 1), so the +1 convention gives LightGBM one tree more than its standalone model uses (103 vs 102 at seed 42). This is kept so that seed 42 reproduces the deposited pipeline. It will be listed in the manuscript's implementation-fidelity table (Appendix Table S10) as a known, immaterial deviation and will not be "fixed" in this analysis.
  - The per-seed counts (best iteration as reported, and trees used) for all three base learners go in `five_seed_stacking_iterations.csv`, alongside the deposited seed-42 values (186 / 103 / 217).
  - FT-Transformer: manuscript Section 3.7.4 and `FT_Transformer_CVD.ipynb`. d_token 64, 8 heads, 3 layers, FFN 256, dropout 0.1; AdamW with lr 1e-3 and weight decay 1e-4; cosine schedule (T_max 100, eta_min 1e-5); gradient clip 1.0; at most 150 epochs; patience 20 on validation AUC; batch size 1,024.
  - SAINT-inspired: manuscript Section 3.7.5 and `SAINT_CVD_Complete.ipynb`. d_model 64, 8 heads, 4 layers, dropout 0.15; class-weighted cross-entropy; AdamW with lr 1e-3 and weight decay 1e-4; cosine schedule; gradient clipping; at most 50 epochs; patience 10; batch size 256. Test inference uses batch size 256 in the seed's test-set row order.
- **Seed-42 anchor check (runs before any other seed).** Derive the seed-42 tree counts (stacking step 1 above), build the seed-42 stacking with them, then apply these rules:
  1. **Fidelity gate (pass/fail).** The retrained seed-42 stacking probabilities must match the deposited `proba_stack_corrected.npy`, with |ΔAUC| ≤ 0.001 against its AUC of 0.800367 **and** Pearson correlation ≥ 0.999. Failing this gate stops the run.
  2. **Tree counts are reported, not used to pass or fail the run.** For each base learner, `seed42_anchor_check.json` records the derived best iteration and the trees used (best_iteration + 1), beside the published 186 / 103 / 217. It also records the device used and the installed xgboost, lightgbm and catboost versions. A small difference from the published counts does not stop the run.
  3. **Structural guard.** If any derived tree count differs from its published value by **more than 20%** (|derived − published| / published > 0.20), the run stops and is reported, **regardless of the AUC result**. In trees, the run continues only within these ranges: XGBoost 149–223 (published 186), LightGBM 83–123 (published 103), CatBoost 174–260 (published 217). A shift that large points to a pipeline difference, not to hardware nondeterminism.
  4. **Stopping and proceeding.** Exactly two conditions stop the run automatically: a fidelity-gate failure and a structural-guard breach. If either occurs, the derived counts, AUC, correlation, device and versions are reported to the author before any other seed runs. Whether and how to proceed is the author's decision, recorded as an amendment in Section 8. If neither occurs, the remaining seeds proceed.
  5. **Why the gate is on predictions, not counts.** `best_iteration` depends on the hardware: GPU and CPU histogram construction and floating-point summation order differ, so early stopping can settle a few rounds apart. The published counts came from a GPU run (GTX 1650), so a small shift on another device is expected. Fidelity is therefore judged on the predictions, not on an intermediate count.
- **Seed 42 is retrained too.** All six ranked models are retrained at every seed that runs, in one environment, so every seed is treated the same way. Seed-42 results are compared with the deposited arrays as a check but never substituted for them.
- **Metric:** test-set ROC-AUC on each seed's own test set, with a 1,000-resample percentile bootstrap 95% CI (bootstrap seed = `s`).
- **Ranking:** within each seed, every model in the ranked set (the "Models" bullet above) is ranked by test AUC (1 = highest). Exact ties share the average rank. The standalone LightGBM is never ranked.
- **Descriptive outputs:**
  - the AUC and rank of every ranked model at every seed run
  - per-model mean, SD, minimum and maximum AUC across those seeds
  - how often each model ranks first
  - Kendall's W of the model rankings across the seeds run
  - the AUC spread among the ranked models within each seed, compared with the between-seed SDs
- **Device and environment: consistency within Section 5.**
  - **One device, one environment.** The seed-42 anchor check and every seed run (five, or three under the budget rule) use one device in one software environment. The device is never switched between seeds.
  - **What "device" means here.** It is the setting for every GPU-capable component: XGBoost `device`, LightGBM `device`, CatBoost `task_type`, and the PyTorch device for FT-Transformer and SAINT-inspired. The scikit-learn Random Forest and the logistic-regression meta-learner always run on CPU, identically at every seed.
  - **Preference.** The GPU is preferred: the published results came from a GTX 1650, and it is faster. CPU is acceptable, provided **every** Section 5 run uses it.
  - **No silent fallback.** The device is set explicitly at the start of Section 5. The notebook pattern `'cuda' if GPU_AVAILABLE else 'cpu'` is not used, because it can fall back to CPU without warning. On a GPU run, the presence of the GPU is asserted before each model fit. A failed assertion counts as the GPU becoming unavailable.
  - **GPU lost part way through: STOP.** The remaining seeds are **not** finished on CPU. Either Section 5 is restarted on CPU from scratch, discarding all GPU results including the anchor check, or the run waits for the GPU and resumes on it. Whichever happened is logged in Section 8 and `RUN_LOG.md`.
  - **Why.** The five-seed analysis compares seeds with each other. Mixing devices would confound hardware variation with split variation, and split variation is the quantity being measured.
  - **Recorded per run in `five_seed_run_manifest.json`:** device, GPU model, driver and CUDA versions (if GPU), Python and library versions (numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, torch), and wall time.
  - **Reruns.** A run that fails is rerun with an identical configuration on the same device. A crash is not a reason to change hyperparameters, and every rerun is logged.
- **Compute budget and run order.**
  1. **The seed-42 anchor check runs first** (above). It needs only the three standalone boosters and the stacking ensemble at seed 42.
  2. **Seed 101 then runs alone, in full,** with all six ranked models plus the standalone LightGBM. Wall time is recorded per model and in total in `five_seed_run_manifest.json`, from the start of the first model fit to the end of the last test prediction for that seed, excluding environment setup and data loading.
  3. **If seed 101's total wall time exceeds 8 hours (28,800 s),** the number of **seeds** is reduced from five to three (**42, 101, 202**) and **all six models are kept**. The analysis exists to answer Reviewer 2's objection that the deep models and the stacking ensemble were left out, so models are not dropped to save time. The reduction is logged as an amendment in Section 8 and committed **before** any further seed starts. The remaining seeds then run in the order 42, 202.
  4. **If seed 101 finishes within 8 hours,** the remaining seeds run in the order 42, 202, 303, 404.

  The seed-42 boosters and stacking fitted for the anchor check are reused in the full seed-42 run, not refitted.

### 5.2 Prespecified interpretations

The decision rule depends only on which of the six models ranks first at each seed. It is applied across all five seeds, or across three seeds (42, 101, 202) if the Section 5.1 compute-budget reduction is triggered.

- **Outcome A (unstable ranking):** two or more different models rank first across the seeds run.
- **Outcome B (stable leader):** the same model ranks first at every seed run.

Under the three-seed reduction, the wording below changes only in the seed count and seed list ("three independent stratified splits (seeds 42, 101, 202)"). The text also states that the analysis was reduced from five seeds under the prespecified compute-budget rule.

**Wording if Outcome A occurs (supports the non-resolvability claim):**

> Across five independent stratified splits (seeds 42, 101, 202, 303, 404) with fixed hyperparameters, the top-ranked model changed between splits ([k] different models ranked first). Between-seed AUC standard deviations ([range]) exceeded the within-split spread among the leading models ([range]). This extends the original split-sensitivity analysis to the stacking ensemble and the deep tabular architectures. It supports our conclusion that the ordering of the leading models is not resolvable at this sample size and should not be read as evidence that any single architecture is superior.

**Wording if Outcome B occurs (weakens the non-resolvability claim):**

> Across five independent stratified splits with fixed hyperparameters, [model] ranked first at every seed. Although pairwise DeLong tests on the primary split were non-significant and the absolute AUC differences were small (mean advantage [x], range [y]), this consistent leader suggests a small but reproducible advantage that a single-split significance test was not powered to detect. We therefore revise the manuscript's claim: the leading models are not statistically distinguishable on any single split, but the repeated-split evidence favours [model]. Whether this advantage is clinically meaningful, and whether it holds on external data, is untested.

**Whichever outcome occurs will be reported** in the revised manuscript (Section 4.5), with the full per-seed table in the supplement. The Abstract, Discussion and Conclusions will be revised to match. Under Outcome B, the sentence "the observed ranking is not resolvable at this sample size" in the Abstract and Conclusions will be qualified as shown above, not kept unchanged.

## 6. Software and reporting

- The revision analyses are run from scripts committed under `src/revision/` **after** this protocol is committed. Each script quotes the Section number of this protocol it implements.
- Every output file is listed in `revision-results/MANIFEST_SHA256.txt` with its SHA-256.
- `revision-results/RUN_LOG.md` records the date, commit hash, environment, and any deviation or rerun.
- **Device policy (general).**
  - **Section 5** requires one device for all its runs (Section 5.1, "Device and environment").
  - **Sections 3 and 3A** prefer, for each model, the device the original used for that model, and record it per model in that section's output: XGBoost on GPU; Random Forest and MLP on CPU (scikit-learn). The same rule applies to the Section 4 pipelines: the deposited TabNet, FT-Transformer and SAINT-inspired checkpoints were trained on GPU and are run there if available, with the inference device recorded.
  - **No cross-device consistency outside Section 5.** The analyses in Sections 3, 3A and 4 are not compared against each other, so they need not share a device. Any device actually used that differs from the original is stated in the corresponding results text.

## 7. Planned output files

All paths are relative to the repository root.

| Section | File | Contents |
| --- | --- | --- |
| 3 | `artifacts/models-revision/rf_seed42.joblib` | Retrained Random Forest |
| 3 | `artifacts/models-revision/mlp_seed42.joblib` | Retrained MLP |
| 3 | `artifacts/models-revision/xgb_seed42.json` | Retrained XGBoost (native format) |
| 3 | `artifacts/models-revision/scaler_seed42.joblib` | StandardScaler fitted on seed-42 train |
| 3 | `revision-results/fidelity/proba_rf_retrained_seed42.npy` | Retrained RF test probabilities |
| 3 | `revision-results/fidelity/proba_mlp_retrained_seed42.npy` | Retrained MLP test probabilities |
| 3 | `revision-results/fidelity/proba_xgb_retrained_seed42.npy` | Retrained XGBoost test probabilities |
| 3 | `revision-results/fidelity/retrain_fidelity_seed42.csv` | Max \|Δp\|, correlation, deposited vs retrained AUC, ΔAUC, pass/fail, XGB best_iteration, device and library versions per model |
| 3 | `revision-results/fidelity/rf_variant_check.csv` | RF variant C vs S against the deposited array: max \|Δp\|, correlation, ΔAUC, match flag, selected variant (variant S fitted only if C fails) |
| 3 | `revision-results/fidelity/proba_rf_variantS_seed42.npy` | Variant S test probabilities (only if variant C fails) |
| 3A(a) | `revision-results/lowcompute/stacking_before_after.csv` | Per version: AUC (bootstrap CI), Acc, F1, Precision, Recall, Brier, calibration intercept and slope, rank among 15; plus ΔAUC (bootstrap CI), DeLong z and p, mean/max \|Δp\|, correlation, n class flips, top-six change, identity check of `proba_stack_primary_original.npy` |
| 3A(b) | `revision-results/lowcompute/bootstrap_auc_by_model.csv` | Per model: AUC, bootstrap 95% CI, number of discarded replicates |
| 3A(b) | `revision-results/lowcompute/delong_effect_sizes_top6.csv` | 15 pairs: ΔAUC, bootstrap 95% CI, DeLong 95% CI, z, p, BH q |
| 3A(b) | `revision-results/lowcompute/delong_effect_sizes_vs_mlp.csv` | 9 pairs: ΔAUC, bootstrap 95% CI, DeLong 95% CI, z, p, BH q |
| 3A(c) | `revision-results/lowcompute/omnibus_roc_test.json` | 15-model and top-six tests: χ², df, p, rank, smallest eigenvalue, condition number, pseudo-inverse used (yes/no) |
| 3A(d) | `revision-results/lowcompute/decision_curve_net_benefit.csv` | Long table: model (15 + treat-all + treat-none) × 91 thresholds, net benefit |
| 3A(d) | `revision-results/lowcompute/decision_curve_summary.csv` | Per model: threshold ranges where NB exceeds both reference strategies; max top-six NB difference |
| 3A(d) | `revision-results/lowcompute/figure_decision_curve_top6.png` | Top six + treat-all + treat-none, 300 dpi |
| 3A(e) | `revision-results/lowcompute/prevalence_transport_metrics.csv` | Model × target prevalence: AUC (check), weighted Brier, calibration-in-the-large, intercept, slope, sens/spec/PPV/NPV at 0.5 and at π_t |
| 3A(e) | `revision-results/lowcompute/prevalence_transport_net_benefit.csv` | Model × target prevalence × 91 thresholds, transported net benefit (+ treat-all) |
| 3A(f) | `revision-results/lowcompute/profile_collision_bound.csv` | Cohort and test set: N, distinct profiles, shared, conflicting, records in conflicting profiles, bound B, 1 − B, AUC ceiling, manuscript-count reproduction flag |
| 3A(g) | `revision-results/lowcompute/grouped_haemodynamic_permutation.csv` | Block and 13 single features: mean AUC drop, SD over 10 repeats, rank among block + 9 non-haemodynamic features, sum of the four singles |
| 3A(h) | `artifacts/models-revision/rf_11feat_seed42.joblib` | Random Forest fitted on the 11-feature set (PP and MAP removed) |
| 3A(h) | `artifacts/models-revision/scaler_11feat_seed42.joblib` | StandardScaler fitted on the 11-feature seed-42 training set |
| 3A(h) | `revision-results/lowcompute/pp_map_removed_rf_importance.csv` | Per feature: mean AUC drop and SD (10 repeats, common sample) for the 11- and 13-feature RF; ranks (i) of 11, (ii) of 13, (iii) of 11 shared in the 13-feature model; rank change |
| 3A(h) | `revision-results/lowcompute/pp_map_removed_rf_summary.json` | Split and column identity checks; 11- and 13-feature test AUC; Table S7 context values; explicit answer: does Cholesterol rise in rank (yes / no / unchanged) |
| 4.1 | `revision-results/xai/xai_common_sample_idx.npy` | The 1,000 test-set positions (sorted) |
| 4.1 | `revision-results/xai/xai_common_sample_manifest.json` | Sampling seed, n, class balance, SHA-256 of index file |
| 4.2 | `revision-results/xai/importance_mlp_permutation.csv` | Feature, mean ΔAUC, SD over 10 repeats |
| 4.2 | `revision-results/xai/importance_xgb_treeshap.csv` | Feature, mean \|SHAP\| |
| 4.2 | `revision-results/xai/importance_tabnet_attention.csv` | Feature, normalised attention importance |
| 4.2 | `revision-results/xai/importance_ft_kernelshap.csv` | Feature, mean \|SHAP\| (class 1) |
| 4.2 | `revision-results/xai/importance_saint_permutation.csv` | Feature, mean ΔAUC, SD over 10 repeats |
| 4.3 | `revision-results/xai/rank_matrix_common_sample.csv` | 13 features × 5 pipelines, ranks |
| 4.3 | `revision-results/xai/spearman_matrix_common_sample.csv` | 5 × 5 Spearman ρ |
| 4.3 | `revision-results/xai/kendall_w_common_sample.json` | Kendall's W (tie-corrected), χ², df, p |
| 4.3 | `revision-results/xai/rank_comparison_original_vs_common.csv` | Original rank, common-sample rank, change, per feature × pipeline |
| 4.3 | `revision-results/xai/figure_rank_heatmap_common_sample.png` | Rank heatmap, 300 dpi, column order fixed as pipelines 1–5 |
| 5 | `revision-results/five_seed/seed42_anchor_check.json` | Fidelity gate: stacking AUC vs 0.800367, \|ΔAUC\|, Pearson correlation with `proba_stack_corrected.npy`, pass/fail. Tree counts (reported, not gating): derived best iteration and trees used vs 186 / 103 / 217, % difference, structural-guard (> 20%) flag. Device; xgboost / lightgbm / catboost versions. Proceed / stop |
| 5 | `revision-results/five_seed/split_indices_seed{S}.npz` | Train/val/test row indices per seed run (5 files, or 3 under the budget rule) |
| 5 | `revision-results/five_seed/proba_{stack,xgb,cat,ft,saint,rf}_seed{S}.npy` | Test probabilities, 6 models × seeds run (30 files, or 18) |
| 5 | `revision-results/five_seed/y_test_seed{S}.npy` | Test labels per seed run |
| 5 | `revision-results/five_seed/five_seed_auc_long.csv` | Model, seed, AUC, 95% CI low, 95% CI high, rank, best iteration/epoch |
| 5 | `revision-results/five_seed/five_seed_rank_table.csv` | 6 models × seeds run, ranks |
| 5 | `revision-results/five_seed/five_seed_summary.csv` | Per-model mean, SD, min, max AUC; times ranked first |
| 5 | `revision-results/five_seed/five_seed_concordance.json` | Kendall's W across seeds; within-seed AUC spread among ranked models |
| 5 | `revision-results/five_seed/five_seed_outcome.json` | Seeds run; which prespecified outcome (A or B) occurred; the leader per seed |
| 5 | `revision-results/five_seed/five_seed_run_manifest.json` | Per run: device, GPU model, driver and CUDA versions, Python and library versions, wall time; run order (anchor, then seed 101); whether the 8-hour seed reduction was triggered; any device interruption and restart; reruns |
| 5 | `revision-results/five_seed/five_seed_stacking_iterations.csv` | Per seed: XGBoost, LightGBM and CatBoost best iteration as reported and trees used in stacking (best_iteration + 1); seed-42 deposited values (186 / 103 / 217) for comparison |
| 5 | `revision-results/five_seed/proba_lgbm_standalone_seed{S}.npy` | Standalone LightGBM test probabilities (not ranked; one per seed run) |
| 5 | `revision-results/five_seed/figure_five_seed_auc.png` | Per-model AUC by seed with CIs, 300 dpi |
| 5 | `revision-results/five_seed/seed42_vs_deposited.csv` | Seed-42 retrained vs deposited AUC per model (descriptive check) |
| 6 | `revision-results/MANIFEST_SHA256.txt` | SHA-256 for every file above |
| 6 | `revision-results/RUN_LOG.md` | Run dates, commit hashes, deviations |

## 8. Amendment log

| # | Date | Section | Change | Reason | Committed before analysis? |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-24 (UTC+6) | Header; repository files outside the protocol | **Editorial only.** Repository URL updated after the GitHub account was renamed from `mdzaideng` to `md-zaid-hossain`: the `git clone` URL in `README.md`. The protocol text itself contains no repository URL. In the same commit, a "Lock timestamp" row was added to the header table, giving the UTC equivalent of the header date. **No analytical choice changed.** | GitHub account rename; record-keeping | Yes. Made before any analysis was run. |
| 2 | 2026-11-24 (UTC+6) | 3A(g) | Amendment 2. Two non-analytical controls added to §3A(g), to establish what block-to-sum ratio arises from grouped permutation at different levels of redundancy, since grouped permutation always exceeds the sum of singles when the grouped features are correlated. Control A (independent reference): Age, Cholesterol, BMI, Active - all original recorded variables, none derived from another, largest pairwise \|Spearman rho\| = 0.17. Control B (partly redundant reference): Age, Cholesterol, BMI, RiskCount - RiskCount is partly a deterministic function of Cholesterol by protocol §1.1, Spearman rho = 0.66 on the test set. Both use four features, matching the haemodynamic block, so the ratios are comparable. Committed before either control was run. No existing analysis or interpretation is changed. | Reference points for the §3A(g) block-to-sum ratio (stated in the change text) | Yes. Committed before either control was run. |
| 3 | 2026-11-24 (UTC+6) | 4.3 | Amendment 3. Decomposition analysis added to §4.3. While running §4.2 it was found that the submitted SAINT-inspired importance ranking was mislabelled: the notebook computed importances on the correctly ordered matrix but printed them under an older feature list, so positions 6-12 carry the wrong names. The model and its predictions are unaffected. The common-sample concordance (Kendall's W 0.836) therefore differs from the submitted value (0.657) for two reasons at once: the common sample and the label correction. To separate them, the concordance of the ORIGINAL unequal-sample rankings is recomputed with the SAINT labels corrected, giving a third reference point. No existing output is modified or deleted. Committed and pushed before the analysis ran. | Separate the effect of the SAINT label correction from the effect of the common sample (stated in the change text) | Yes. Committed and pushed before the analysis ran. |
| 4 | 2026-11-24 (UTC+6) | 5.1 | Amendment 4. Two items, recorded before any seed beyond the anchor was run. (a) SAINT weight restoration in §5 uses the procedure AS EXECUTED in the original notebook: model.state_dict().copy() is a shallow copy, so the snapshot tracks training and the restored weights are those of the final epoch, not the best epoch. At seed 42 the original run reached its best validation AUC at epoch 19 (0.8027) and stopped at epoch 29 (0.8020); the deposited checkpoint holds the epoch-29 weights. §5 keeps this behaviour so that only the seed varies and the seed-42 retrain remains comparable with the deposited arrays. The defect is recorded as discrepancy item 5 and belongs in the implementation fidelity table. (b) As a diagnostic only, the seed-42 run additionally retains the true best-epoch weights using a deepcopy and reports their held-out test AUC alongside the final-epoch value, to quantify how much the defect affects performance. This diagnostic does not feed any §5 result and no other seed records it. | Resolve the SAINT restore ambiguity in favour of the executed procedure; quantify the defect at seed 42 (stated in the change text) | Yes. Committed and pushed before any seed beyond the anchor was run. |
| 5 | 2026-11-24 (UTC+6) | 5.2 | **Amendment 5 — POST HOC. Replaces the Outcome A supporting sentence.** **Status: post hoc.** These quantities were specified after the five-seed results were produced. They are NOT prespecified and must never be described as such in the manuscript or the response letter. **Reason:** the locked Outcome A text reads "between-seed AUC standard deviations ([range]) exceeded the within-split spread among the leading models ([range])". That sentence compares a standard deviation to a range and is therefore ill-posed as written; a range over six values is roughly 2.5x its own standard deviation, so the comparison cannot be evaluated. The defect was present at lock time and was discovered on execution. It is recorded, not silently corrected. This is the third instance in which a locked prediction or statement failed on execution; the others are the Naive Bayes clipping outcome in 3A(e) and the justification for section 4. **Replacement quantities, over the six ranked models and five seeds:** (i) between-seed SD per model, and its mean over models; (ii) within-seed SD across models per seed, and its mean over seeds; (iii) the ratio of (i) to (ii); (iv) the same comparison using ranges instead of SDs; (v) a two-way main-effect decomposition: seed-effect SD, model-effect SD, residual SD, and each term's share of total sum of squares; (vi) per-model rank vector, mean rank and rank range across seeds. No stop condition. Descriptive only. No interval estimates: five seeds are not an independent sample from a defined population. | Locked Outcome A comparison ill-posed (stated in the change text) | **No — POST HOC.** Specified after the five-seed results were produced; committed and pushed alone before quantities (i)–(vi) were computed. |
| 6 | 2026-11-24 (UTC+6) | 5 | **Amendment 6 — POST HOC. Retraining as a variance stabiliser (manuscript §5.6).** **Status: post hoc.** Specified after the five-seed results, prompted by the observation that the observed between-seed AUC SD was about three times smaller than the per-seed bootstrap SE implied. NOT prespecified; must never be described as such. **Quantities:** (i) Reference: the analytic (Hanley–McNeil) standard error of test-set AUC at n = 10,281, AUC approx 0.80, positive rate 0.49441. This is the primary reference because it requires no simulation and no model. (ii) Corroboration: SD of a fixed model's AUC over 200 test sets drawn by the protocol's own first split step. Report for the seed-42 XGBoost and Random Forest models. State the leakage caveat: roughly 70% of each drawn test set was in that model's training data, so the LEVEL is inflated; only the SD is interpretable. (iii) Contrast: the observed between-seed SD per model from §5, retrained at each split. (iv) The ratio of (iii) to (i), reported with the explicit caveat that (iii) rests on five values and so carries roughly 35% relative uncertainty; the ratio is therefore indicative, not precise. No mechanism is attributed. The five-seed capacity table (early-stopping counts against per-seed mean AUC) is reported as data only; with five seeds no relationship is estimable and none is fitted. The effect is attributed to retraining as a whole, not to early stopping. No stop condition. No interval estimates, for the reason given in Amendment 5. | Between-seed SD about three times smaller than the bootstrap SE implied (stated in the change text) | **No — POST HOC.** Specified after the five-seed results; committed and pushed alone before quantities (i)–(iv) were written to the repository. |
| 7 | 2026-11-24 (UTC+6) | 4.5 / Appendix Table S8 | **Amendment 7 — re-execution of two robustness analyses whose outputs were not retained.** **Status: prespecified.** Written and pushed before these analyses are run. **Rationale:** Table S8 of the submitted manuscript records "Not retained; reported descriptively" for the Random Forest learning curve and the feature-profile overlap analysis. Both are cheap to re-execute, and depositing their outputs removes two undeposited claims from the paper. **(a) Random Forest learning curve.** Section 3 Random Forest configuration, primary seed-42 split, training-set fractions 10, 25, 50, 75 and 100 per cent, five stratified subsamples at each fraction, test AUC on the fixed primary test set. Report mean and SD at each fraction. **(b) Feature-profile overlap.** Random Forest, MLP and XGBoost on a profile-disjoint split in which no exact feature profile appears in both training and test partitions. Report test AUC for each model and the resulting partition sizes and positive rates. Both are reported as re-executions. The original values (learning curve 0.7942 to 0.7995; overlap AUC 0.8100 / 0.8052 / 0.8116) are quoted in the outputs for comparison only. If the new values differ, the new ones stand and the difference is recorded; the original values are not reproducible and are not treated as a target. No stop condition. | Table S8 records two undeposited, non-reproducible claims (stated in the change text) | Yes. Written and pushed before these analyses are run. |
| 8 | 2026-11-24 (UTC+6) | 4.5 / 5 | **Amendment 8 — bounded attempt to reproduce the withdrawn between-split standard deviations.** **Status: prespecified.** Written and pushed before the runs. **Background:** the submitted manuscript reports between-split SDs of 0.0038-0.0042 and mean AUCs of 0.7958-0.7988 for CatBoost, XGBoost, LightGBM, Random Forest and MLP across five stratified splits. No code producing these values exists in any deposited or local file, and the original seeds are unknown, so exact reproduction is impossible in principle. This amendment tests only whether the reported MAGNITUDE arises from a plausible configuration. Exactly two configurations are run, both on the five models named above, both using seeds 42, 101, 202, 303, 404, both evaluated by test AUC on each seed's own test set. **Configuration A — per-split refitting.** Hyperparameters as published; early stopping and tree counts re-derived at each split from that split's own validation set. This matches the §5 procedure; for XGBoost, LightGBM (standalone), CatBoost and Random Forest, Configuration A reuses the already-deposited §5 arrays rather than refitting, since §5 used this exact procedure, these exact seeds and this exact split function. Only MLP is newly fit for Configuration A, as it is not part of §5's model set. **Configuration B — fixed capacity.** Hyperparameters as published AND tree counts / epoch counts fixed at the published seed-42 values (186 / 103 / 217 trees for XGBoost / LightGBM / CatBoost, the values already established as both the manuscript's published counts and the §5 seed-42 derived counts). Models are refitted on each split's training data but no per-split early stopping is performed. Random Forest has no early-stopping or capacity-selection step in either configuration (n_estimators is fixed at 300 throughout the protocol); Configuration A and B therefore give identical Random Forest results by construction, and the single §5 array is reused for both. MLP has no published epoch count in the manuscript; Configuration B fixes its iteration count at Configuration A's own seed-42 value (`n_iter_`), consistent with the boosters, where the published count already equals the seed-42-derived count. **Prespecified expectation:** if the manuscript's figure arose from a procedure without per-split capacity selection, Configuration B will give between-split SDs near the analytic standard error (0.0044) and Configuration A will give SDs near those observed in §5 (0.0008-0.0017). **Reporting rule, binding:** report whatever both configurations give. Do not adjust either configuration to approach 0.0038-0.0042. If neither reproduces the reported magnitude, the result is recorded as "not reproduced under either prespecified configuration" and no third configuration is attempted under this amendment. **Secondary use:** A versus B is a matched test of the refitting-stabiliser effect reported in Amendment 6, holding model set and splits constant. Stop condition: none. | No code or seeds exist for the submitted 0.0038-0.0042 figure; test whether its magnitude is plausible under two bounded configurations (stated in the change text) | Yes. Written and pushed before the runs. |
| 9 | 2026-11-25 (UTC+6) | 4.3 (XGBoost pipeline) / 3A(g) | **Amendment 9 — correlation-aware attribution for the XGBoost pipeline.** **Status: prespecified.** Written and pushed before the runs. **Rationale:** Reviewer 3 noted that correlated predictors destabilise SHAP and permutation importance and that the manuscript cited this literature without acting on it. Two predictors (PP and MAP) are deterministic functions of SBP and DBP, and RiskCount is a deterministic function of five recorded variables, so methods requiring an estimated conditional distribution are ill-posed here. Two correlation-aware approaches that do not require one are specified instead. All runs use the XGBoost model and the common 1,000-record sample of Section 4.1. **(a)** TreeSHAP under both feature_perturbation settings, "interventional" and "tree_path_dependent". Report the mean absolute SHAP value and rank for all 13 features under each, the Spearman correlation between the two rankings, and each feature's rank change. **(b)** Partition (Owen value) attribution, with the feature hierarchy built by hierarchical clustering on the absolute Spearman correlation matrix of the 13 features. Report the ranking, its Spearman correlation with each ranking from (a), and the dendrogram's cluster structure. **(c)** For each of the three attributions, report the summed absolute attribution of the haemodynamic set {SBP, DBP, PP, MAP} and each member's share of that sum. **Run specification:** Model: the retrained seed-42 XGBoost (`artifacts/models-revision/xgb_seed42.json`), raw (unscaled) features, as in Section 4.2. Explained rows: the Section 4.1 common sample (n = 1,000). Output scale: raw log-odds margin for all three attributions. Background (interventional TreeSHAP and the Partition masker): 200 training rows, `np.random.seed(42); np.random.choice(len(X_train), 200, replace=False)`, as for Section 4.2 pipeline 4. Hierarchy: \|Spearman rho\| on the training set; distance 1 − \|rho\|; average linkage (scipy `linkage` on the condensed matrix). Partition explainer: max_evals = 2,000; report whether the hierarchy was fully expanded. Ranks: `rankdata(-mean\|attribution\|, method='average')`. **Prespecified expectation:** the haemodynamic set's total attribution will be similar across the three methods, while its distribution among the four members will differ, consistent with the grouped permutation result of Section 4.5. **Reporting rule, binding:** report whatever is obtained. Do not tune any setting toward the expectation. If the results contradict the grouped permutation finding, that contradiction is reported and the manuscript text is revised accordingly. Stop condition: none. | Reviewer 3: correlated predictors destabilise SHAP and permutation importance (stated in the change text) | Yes. Written and pushed before the runs. |
| 10 | 2026-11-25 (UTC+6) | New analysis (external validation, NHANES) | **Amendment 10 — external validation on NHANES.** **Status: prespecified.** Written and pushed before any model is scored on NHANES data. Every mapping below is fixed now and is not revised after any performance result is seen. **Rationale:** Reviewer 3 asked for external validation. A feasibility count (no scoring) established that NHANES carries every required variable and that sufficient records survive harmonisation. **COHORT.** Primary: NHANES 2013-2014 and 2015-2016 only. These use auscultatory blood pressure (BPX), consistent with one another. Secondary, reported SEPARATELY and never pooled with the primary: NHANES 2017-March 2020, which uses oscillometric blood pressure (BPXO). The two methods differ systematically and SBP is the dominant predictor, so pooling would confound measurement method with transportability. Age 40-65 inclusive. The primary cohort contains 0.6% of records aged 30-39 while NHANES 30-65 is 26.1% that band, so the model has effectively not seen it. Survey weights are NOT applied. The target is individual-level transportability, not a population estimate. Stated as a limitation. **VARIABLE MAPPING (fixed).** Blood pressure: the mean of non-missing readings 1-3 (BPXSY1-3 / BPXDI1-3). Records with DBP = 0 are excluded. Sex: RIAGENDR 1 (male) -> 1, 2 (female) -> 0, matching the primary coding. BMI: the measured BMXBMI. Cleaning, exactly as the primary cohort: stage 1 SBP in [80, 200] and DBP in [40, 130]; stage 2 BMI in [10, 60] and SBP - DBP > 0. Cholesterol category from LBXTC (mg/dL): <200 -> 1, 200-239 -> 2, >=240 -> 3. These are standard clinical cut-offs. The source dataset does not document its own, so this is an assumption, stated as such. Glucose category from serum LBXSGL (mg/dL): <100 -> 1, 100-125 -> 2, >=126 -> 3. Serum is primary because requiring fasting glucose costs 51% of the sample. Smoking: current smoker = SMQ020 = 1 AND SMQ040 in {1, 2} -> 1, otherwise 0. Alcohol: any drinking in the past 12 months -> 1, otherwise 0, derived from ALQ120Q with ALQ120U (2013-16). Active: meeting 150 minutes per week of moderate-equivalent activity -> 1, otherwise 0. Minutes are summed over PAQ620/PAD630 (moderate work), PAQ665/PAD675 (moderate recreation) and PAQ635/PAD645 (walking or cycling), with vigorous minutes PAQ605/PAD615 and PAQ650/PAD660 counted double. Derived, exactly as the primary cohort: PP = SBP - DBP; MAP = (SBP + 2 x DBP) / 3; risk_count = (cholesterol>1) + (glucose>1) + smoking + alcohol + (1 - active). Feature column order identical to protocol §1.1 FEATURES. Outcome: positive if any of MCQ160B, C, D, E, F = 1; negative if all five = 2; otherwise the record is excluded. **SCALING — critical.** The deposited seed-42 StandardScaler (artifacts/models-revision/scaler_seed42.joblib) is APPLIED to the NHANES matrix. It is never refitted. Refitting would silently re-centre the external data and invalidate every result. **MODELS.** The six leading models plus the MLP baseline, from the deposited and Section 3 retrained artefacts. No model is retrained or tuned on NHANES. **REPORTED, per model and per cohort (primary and secondary separately):** (i) N, event count and prevalence. (ii) AUC with 1,000-resample percentile bootstrap 95% CI, seed 42. (iii) Calibration intercept and slope, as submitted. (iv) The same after intercept-only recalibration to the NHANES prevalence; the slope is NOT refitted. (v) Brier score before and after (iv). (vi) The AUC drop against each model's primary-split AUC. (vii) The marginal distributions of all 13 features beside the primary cohort's, so case-mix differences are visible. **PRESPECIFIED SENSITIVITY ANALYSES — exactly these three, no others:** (A) Fasting glucose (LBXGLU) in place of serum, same cut-offs. (B) Ever-smoker (SMQ020 = 1) in place of current smoker. (C) Cholesterol cut-offs at <190 / 190-229 / >=230 in place of the primary set. **BINDING RULES.** Whatever is obtained is reported, including a large AUC drop. No mapping is changed after any performance result is seen. No sensitivity analysis beyond (A)-(C) is run under this amendment. The outcome is self-reported diagnosed CVD history, a different construct from the primary label, whose provenance is undocumented. A poor result cannot distinguish model failure from label mismatch, and the manuscript will say so rather than attribute it either way. **Run specification (added before the push, at the author's approval):** Missing data: complete case. A record is excluded if any input is missing, refused (7, 77, 777, 7777) or don't know (9, 99, 999, 9999), except the structural skips below. DBP = 0: a record is excluded if ANY of readings 1-3 has DBP = 0, before averaging. Alcohol: 2013-16: ALQ120Q 1-365 -> 1; ALQ120Q = 0 -> 0; ALQ101 = 2 and ALQ110 = 2 (skipped) -> 0. 2017-20 (secondary): ALQ121 1-10 -> 1; ALQ121 = 0 -> 0; ALQ111 = 2 (never drank, skipped) -> 0. Active: weekly minutes = PAQ625 x PAD630 + PAQ670 x PAD675 + PAQ640 x PAD645 + 2 x (PAQ610 x PAD615 + PAQ655 x PAD660); an item is 0 when its gate (PAQ620 / PAQ665 / PAQ635 / PAQ605 / PAQ650) = 2; >= 150 -> 1, otherwise 0. Model inputs: Random Forest, MLP and FT-Transformer get scaler_seed42.transform(X); SAINT-inspired gets the scaler stored in its deposited checkpoint, inference batch 256, rows ordered by cycle then SEQN; XGBoost, CatBoost and the Stacking Ensemble get raw X, as they were trained. No scaler is ever refitted. CatBoost and Stacking Ensemble: no saved model object exists. Each is refitted at seed 42 on the PRIMARY training split only (CatBoost as in the Section 5 anchor; stacking as in src/run_corrected_stacking.py, tree counts 186 / 103 / 217) and must pass the fidelity gate (\|dAUC\| <= 0.001 and r >= 0.999 against the deposited test-set arrays) before any NHANES scoring. Recalibration (iv): the intercept is re-estimated on the NHANES cohort by logistic regression with logit(p) as offset (slope fixed at 1); this is in-sample and is stated as such. | Reviewer 3 asked for external validation; feasibility count (no scoring) confirmed the variables and N (stated in the change text) | Yes. Written and pushed before any model is scored on NHANES data. |
| 11 | 2026-11-25 (UTC+6) | New secondary analysis (gender-stratified performance) | **Amendment 11 — gender-stratified performance.** **Status: prespecified.** Written and pushed before the run. **Rationale:** the submitted manuscript states that no fairness analysis was performed, while a deposited notebook computes per-gender metrics for a model and test set outside the primary benchmark. The statement is inaccurate and the existing output is not usable for the primary models. **Run:** for each of the six leading models, on the primary seed-42 test set, report accuracy, precision, recall, false positive rate, positive prediction rate and AUC with bootstrap CI, separately for the two recorded gender categories, plus the group sizes and outcome prevalence in each. Report the between-group AUC difference per model. **Run specification:** predictions are the deposited test-set probability arrays (`artifacts/predictions/proba_{stack_corrected,xgb,cat,ft,saint,rf}.npy`, `y_test.npy`); no model is refitted. Gender categories come from the Gender column of the primary seed-42 test set (0 = female, 1 = male). Classification metrics use the manuscript's 0.5 threshold. AUC 95% CIs use the Section 3A(b) convention (percentile bootstrap, 2,000 resamples, seed 42), resampling within each gender group. The between-group AUC difference is male minus female, reported as a point estimate. No stop condition. Descriptive only. No claim of fairness or its absence is prespecified; whatever is obtained is reported. | Submitted manuscript states no fairness analysis was performed; the existing per-gender output is for a model outside the primary benchmark (stated in the change text) | Yes. Written and pushed before the run. |
