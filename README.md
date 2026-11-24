<h1 align="center">CVD XAI Cross-Pipeline Benchmark</h1>

<p align="center">
  <strong>A reproducible 15-model benchmark for cardiovascular disease status classification</strong><br>
  <em>with paired DeLong testing, calibration analysis, and cross-pipeline XAI feature-ranking comparison</em>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.22114346"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22114346-1682D4?style=flat-square"></a>
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-success?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY%204.0-green?style=flat-square">
  <img alt="Validation" src="https://img.shields.io/badge/release%20validation-PASS-brightgreen?style=flat-square">
  <img alt="Models" src="https://img.shields.io/badge/models-15-informational?style=flat-square">
</p>

<p align="center">
  <strong>Version DOI</strong> <a href="https://doi.org/10.5281/zenodo.22114347">10.5281/zenodo.22114347</a>
  &nbsp;·&nbsp;
  <strong>All-versions DOI</strong> <a href="https://doi.org/10.5281/zenodo.22114346">10.5281/zenodo.22114346</a>
</p>

---

> [!WARNING]
> **Research use only.** This is a retrospective, cross-sectional CVD-**status** classification benchmark — not a prospective risk score, diagnostic device, or clinical decision-support system. The outcome label is disease status at examination, not an incident cardiovascular event. External validation on independent cohorts is required before any clinical interpretation.

---

## Contents

[Overview](#overview) · [At a glance](#at-a-glance) · [Headline results](#headline-results) · [Repository structure](#repository-structure) · [Quick start](#quick-start) · [Reproduction levels](#reproduction-levels) · [Notebook guide](#notebook-guide) · [Results index](#results-index) · [Data](#data-access-and-cohort-construction) · [Integrity](#integrity-and-provenance) · [Scope](#scope-and-limitations) · [Licensing](#licensing) · [Citation](#citation) · [Authors](#authors)

---

## Overview

This repository is the complete computational record for a multi-paradigm benchmark of cardiovascular disease status classification. Fifteen supervised classifiers spanning four modelling paradigms were trained and evaluated on **one common stratified held-out test set** (`n = 10,281`), then compared with paired statistical testing rather than by leaderboard ordering alone.

The four paradigms evaluated:

| Paradigm | Models |
|---|---|
| **Classical ML** | Logistic Regression, Decision Tree, k-NN, Naive Bayes, SVM (RBF), Random Forest |
| **Gradient boosting** | XGBoost, LightGBM, CatBoost |
| **Neural / tabular deep learning** | MLP, FT-Transformer, TabNet, VIME-inspired SSL, SAINT-inspired |
| **Ensemble** | Corrected Stacking Ensemble (XGB + LGBM + CatBoost + RF, logistic meta-learner) |

Three analytical layers sit on top of the benchmark:

- **Discrimination testing** — all 105 pairwise AUC comparisons via the paired midrank DeLong test, with Benjamini–Hochberg and Holm multiplicity control.
- **Calibration** — Brier score, calibration intercept and slope computed from exact stored test probabilities for every model.
- **Cross-pipeline explainability** — Spearman rank concordance between feature rankings produced by five independent XAI pipelines (permutation importance, SHAP, and native attention), testing whether different explanation methods agree on what drives the prediction.

Everything needed to re-derive the statistics is deposited: exact test-set probability vectors, binary predictions, trained model artifacts, and portable path-independent scripts.

---

## At a glance

| | |
|---|---|
| **Models benchmarked** | 15, on one common test split |
| **Held-out test set** | 10,281 records (stratified) |
| **Analytical cohort** | 68,534 records after two-stage cleaning (from 70,000) |
| **Pairwise DeLong comparisons** | 105, all reported with BH and Holm adjustment |
| **XAI pipelines compared** | 5 (MLP permutation, XGBoost SHAP, TabNet attention, FT-Transformer SHAP, SAINT-inspired permutation) |
| **Executed notebooks** | 10 authoritative, with stored outputs and no stored exceptions |
| **Prediction arrays deposited** | 35 `.npy` files (17 probability, 17 binary, 1 label vector) |
| **Model artifacts** | 5 (FT-Transformer, TabNet, VIME-inspired, SAINT-inspired ×2) |
| **Verification** | `python tools/verify_release.py` → `RELEASE VALIDATION: PASS` |

---

## Headline results

### Discrimination — all 15 models

Ranked by ROC-AUC on the common held-out test set. **Bold** rows form the statistically indistinguishable top tier.

| # | Model | ROC-AUC | Accuracy | F1 | Precision | Recall | Brier |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** | **Stacking Ensemble** | **0.800367** | 0.73349 | 0.71903 | 0.75091 | 0.68975 | 0.18137 |
| **2** | **XGBoost** | **0.800238** | 0.73291 | 0.71790 | 0.75124 | 0.68739 | 0.18118 |
| **3** | **CatBoost** | **0.800092** | 0.73291 | 0.71644 | 0.75397 | 0.68247 | 0.18108 |
| **4** | **FT-Transformer** | **0.799770** | 0.73359 | 0.71951 | 0.75032 | 0.69113 | 0.18150 |
| **5** | **SAINT-inspired** | **0.799590** | 0.73252 | 0.72401 | 0.73899 | 0.70962 | 0.18173 |
| **6** | **Random Forest** | **0.799533** | 0.73222 | 0.71121 | 0.76180 | 0.66693 | 0.18169 |
| 7 | LightGBM | 0.799532 | 0.73378 | 0.71740 | 0.75489 | 0.68345 | 0.18143 |
| 8 | VIME-inspired SSL | 0.799047 | 0.73300 | 0.71581 | 0.75546 | 0.68011 | 0.18171 |
| 9 | TabNet | 0.798654 | 0.73398 | 0.71555 | 0.75905 | 0.67677 | 0.18212 |
| 10 | MLP | 0.798016 | 0.73300 | 0.71756 | 0.75216 | 0.68601 | 0.18231 |
| 11 | Logistic Regression | 0.792856 | 0.73115 | 0.71070 | 0.75934 | 0.66791 | 0.18637 |
| 12 | Decision Tree | 0.791536 | 0.72717 | 0.71445 | 0.74030 | 0.69034 | 0.18512 |
| 13 | SVM (RBF) | 0.790108 | 0.73106 | 0.70713 | 0.76595 | 0.65670 | 0.18649 |
| 14 | Naive Bayes | 0.784925 | 0.71919 | 0.68229 | 0.77423 | 0.60988 | 0.23239 |
| 15 | k-NN | 0.738954 | 0.69731 | 0.69109 | 0.69746 | 0.68483 | 0.22196 |

Best model: **Stacking Ensemble**, ROC-AUC `0.800367` (bootstrap 95% CI `0.791538 – 0.809095`).

### The central finding — rank order is not statistical separation

> The six highest-ranked models span only **≈ 0.0008 AUC**. All **15** pairwise DeLong comparisons among them are non-significant, both unadjusted and after BH correction.

| Comparison set | Comparisons | Significant after BH (α = 0.05) |
|---|---:|:---:|
| Top-six models, pairwise | 15 | **0 / 15** |
| Advanced models vs. MLP baseline | 9 | **0 / 9** |
| All models, pairwise | 105 | 60 / 105 |

The 60 significant results across the full 105-pair matrix are driven almost entirely by the clearly weaker models (k-NN, Naive Bayes, SVM, Decision Tree, Logistic Regression) — not by separation within the upper tier.

<details>
<summary><strong>Top-six pairwise DeLong detail (all non-significant)</strong></summary>

| Model A | Model B | ΔAUC | z | p | BH q |
|---|---|---:|---:|---:|---:|
| Stacking Ensemble | Random Forest | 0.000834 | 1.1995 | 0.2303 | 0.8865 |
| Stacking Ensemble | SAINT-inspired | 0.000777 | 1.0312 | 0.3025 | 0.8865 |
| XGBoost | Random Forest | 0.000704 | 1.0178 | 0.3088 | 0.8865 |
| CatBoost | Random Forest | 0.000559 | 0.7845 | 0.4328 | 0.8865 |
| XGBoost | SAINT-inspired | 0.000647 | 0.7790 | 0.4360 | 0.8865 |
| Stacking Ensemble | FT-Transformer | 0.000598 | 0.7696 | 0.4415 | 0.8865 |
| Stacking Ensemble | CatBoost | 0.000275 | 0.6213 | 0.5344 | 0.8865 |
| CatBoost | SAINT-inspired | 0.000502 | 0.5689 | 0.5694 | 0.8865 |
| XGBoost | FT-Transformer | 0.000468 | 0.5454 | 0.5855 | 0.8865 |
| CatBoost | FT-Transformer | 0.000322 | 0.3777 | 0.7057 | 0.8865 |
| Stacking Ensemble | XGBoost | 0.000130 | 0.3373 | 0.7359 | 0.8865 |
| FT-Transformer | Random Forest | 0.000236 | 0.2694 | 0.7876 | 0.8865 |
| XGBoost | CatBoost | 0.000146 | 0.2510 | 0.8018 | 0.8865 |
| FT-Transformer | SAINT-inspired | 0.000179 | 0.2180 | 0.8274 | 0.8865 |
| SAINT-inspired | Random Forest | 0.000057 | 0.0595 | 0.9525 | 0.9525 |

</details>

<details>
<summary><strong>Advanced models vs. MLP baseline (all non-significant after BH)</strong></summary>

| Model vs MLP | ΔAUC | z | p (unadj.) | BH q |
|---|---:|---:|---:|---:|
| Stacking Ensemble | 0.002351 | 2.6117 | 0.00901 | 0.07829 |
| CatBoost | 0.002076 | 2.3090 | 0.02095 | 0.07829 |
| XGBoost | 0.002222 | 2.2248 | 0.02610 | 0.07829 |
| FT-Transformer | 0.001753 | 1.8349 | 0.06652 | 0.14966 |
| SAINT-inspired | 0.001574 | 1.6161 | 0.10608 | 0.17211 |
| LightGBM | 0.001516 | 1.5260 | 0.12701 | 0.17211 |
| Random Forest | 0.001517 | 1.4991 | 0.13386 | 0.17211 |
| VIME-inspired SSL | 0.001031 | 1.1924 | 0.23309 | 0.26223 |
| TabNet | 0.000637 | 0.6207 | 0.53480 | 0.53480 |

Three models reach nominal significance against the MLP baseline; none survives BH correction across the nine-comparison family.

</details>

### Calibration — top six

Computed from exact stored probabilities. A well-calibrated model has intercept ≈ 0 and slope ≈ 1.

| Model | Brier | Calibration intercept | Calibration slope |
|---|---:|---:|---:|
| Stacking Ensemble | 0.181368 | +0.0056 | 0.9918 |
| XGBoost | 0.181185 | +0.0046 | 0.9853 |
| CatBoost | 0.181082 | +0.0047 | 1.0011 |
| FT-Transformer | 0.181496 | −0.0138 | 1.0108 |
| SAINT-inspired | 0.181730 | −0.0731 | 1.0422 |
| Random Forest | 0.181691 | +0.0068 | 1.0324 |

All six are close to ideal calibration. By contrast, Naive Bayes (slope 0.2405) and k-NN (slope 0.1218) are severely miscalibrated despite non-trivial AUC — reinforcing that discrimination and calibration must be reported together.

### Cross-pipeline XAI concordance

Spearman rank correlation between the feature rankings produced by five independent explanation pipelines:

| | MLP Perm. | XGBoost SHAP | TabNet Attn. | FT-Trans. SHAP | SAINT Perm. |
|---|:---:|:---:|:---:|:---:|:---:|
| **MLP Perm.** | 1.000 | 0.780 | 0.659 | 0.692 | 0.319 |
| **XGBoost SHAP** | 0.780 | 1.000 | 0.819 | **0.896** | 0.330 |
| **TabNet Attn.** | 0.659 | 0.819 | 1.000 | 0.819 | 0.187 |
| **FT-Trans. SHAP** | 0.692 | **0.896** | 0.819 | 1.000 | 0.214 |
| **SAINT Perm.** | 0.319 | 0.330 | 0.187 | 0.214 | 1.000 |

Four of the five pipelines show substantial agreement (ρ = 0.66–0.90), while the SAINT-inspired permutation ranking is a clear outlier (ρ = 0.19–0.33 against all others). Models that are statistically indistinguishable in discrimination can therefore still disagree markedly about *why* they predict what they predict — the motivating observation for the cross-pipeline analysis.

### Robustness and sensitivity

<details>
<summary><strong>Guideline-defined feature ablation</strong></summary>

| Feature subset | n | Test AUC |
|---|:---:|---:|
| Core clinical 3 (Age, SBP, Cholesterol) | 3 | 0.792891 |
| Guideline-aligned 6 (+ Sex, Glucose, Smoking) | 6 | 0.794442 |
| **Original 10 variables** | **10** | **0.799843** |
| Original + PP + MAP | 12 | 0.799302 |
| Full 13 variables (+ risk_count) | 13 | 0.799534 |

A three-variable clinical core recovers ~99% of the full-model AUC. Engineered features (pulse pressure, mean arterial pressure, risk count) add nothing beyond the original ten.

</details>

<details>
<summary><strong>SAINT-inspired batch-size and ordering sensitivity</strong></summary>

| Condition | AUC | Mean abs. diff. vs. batch-256 | Prediction corr. |
|---|---:|---:|---:|
| batch_size 16 | 0.803179 | 0.011367 | 0.99793 |
| batch_size 64 | 0.803898 | 0.005142 | 0.99961 |
| **batch_size 256 (reference)** | **0.804481** | — | — |
| batch_size 1024 | 0.804531 | 0.002012 | 0.99994 |
| shuffled order 1 (batch 256) | 0.804500 | 0.003111 | 0.99986 |
| shuffled order 2 (batch 256) | 0.804470 | 0.003083 | 0.99987 |
| shuffled order 3 (batch 256) | 0.804633 | 0.003213 | 0.99985 |

Predictions are stable to batch construction and sample ordering (correlation ≥ 0.998 in every condition).

</details>

<details>
<summary><strong>Corrected stacking ensemble — out-of-fold detail</strong></summary>

Five-fold out-of-fold AUC for each base learner:

| Fold | XGBoost | LightGBM | CatBoost | Random Forest |
|:---:|---:|---:|---:|---:|
| 1 | 0.803535 | 0.802650 | 0.803932 | 0.801756 |
| 2 | 0.804926 | 0.804122 | 0.805533 | 0.803607 |
| 3 | 0.801954 | 0.801725 | 0.802353 | 0.800542 |
| 4 | 0.795221 | 0.795150 | 0.796102 | 0.793554 |
| 5 | 0.797903 | 0.797070 | 0.798669 | 0.796743 |

Logistic meta-learner: intercept −2.3713; weights XGB 1.6917, LGBM 0.4979, CatBoost 3.0566, RF −0.4670. Average precision 0.779327.

</details>

---

## Repository structure

```text
cvd-xai-cross-pipeline-benchmark/
│
├── notebooks/
│   ├── core/                     10 authoritative executed notebooks (.ipynb)
│   └── executed-pdf/             9 human-readable PDF exports
│
├── src/                          Portable, path-independent analysis scripts
│   ├── prepare_data.py               two-stage cohort construction
│   ├── run_validated_stats.py        metrics, DeLong, BH/Holm, calibration
│   ├── run_corrected_stacking.py     corrected stacking ensemble
│   ├── run_guideline_ablation.py     feature-subset ablation
│   └── run_saint_sensitivity.py      batch/order sensitivity
│
├── results/
│   ├── figures/
│   │   ├── main/                 Figures 1–4 (PNG + PDF)
│   │   ├── alternates/           alternate DeLong heatmap renderings
│   │   └── graphical-abstract/   PNG + editable SVG
│   └── tables/                   10 validated result tables (CSV) + stats log
│
├── artifacts/
│   ├── predictions/              35 .npy arrays — exact test probabilities,
│   │                             binary predictions, and the label vector
│   └── models/                   5 trained model artifacts (.pt / .zip)
│
├── data/
│   └── README.md                 source, schema, cleaning rules, checksums
│                                 (patient-level CSVs are not redistributed)
│
├── docs/
│   ├── REPRODUCIBILITY.md        four-level reproduction guide
│   ├── AUDIT_REPORT.md           package audit and corrections applied
│   ├── DATA_DICTIONARY.md        11-column schema definition
│   ├── RESULTS_INDEX.md          artifact → generator mapping
│   └── RELEASE_CHECKLIST.md      release procedure
│
├── environment/
│   ├── requirements-verify.txt   minimal, verification only
│   ├── requirements-corrections.txt  full reproduction stack
│   ├── environment.yml           conda equivalent
│   └── ORIGINAL_ENVIRONMENT.md   original Windows/CUDA training environment
│
├── provenance/
│   ├── SHA256SUMS.txt            deterministic release manifest
│   ├── SOURCE_FILES.csv          per-file provenance record
│   └── ORIGINAL_ARCHIVES_SHA256.txt
│
├── tools/
│   ├── verify_release.py         programmatic release validator
│   └── generate_manifest.py      regenerates SHA256SUMS.txt
│
├── archive/                      superseded notebooks and the original
│                                 hard-coded script — provenance only
│
├── .github/workflows/validate.yml    CI: runs verify_release.py
├── CITATION.cff · .zenodo.json · CHANGELOG.md
└── LICENSE.md · LICENSES/
```

---

## Quick start

### 1 · Verify the deposited evidence (~1 minute, no dataset needed)

```bash
git clone https://github.com/md-zaid-hossain/cvd-xai-cross-pipeline-benchmark.git
cd cvd-xai-cross-pipeline-benchmark

python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r environment/requirements-verify.txt

python tools/verify_release.py
```

Expected output:

```text
RELEASE VALIDATION: PASS
```

This confirms that every required file is present, all ten core notebooks parse with stored outputs and no stored exceptions, the label vector is 10,281 binary records, all 15 model prediction arrays reproduce the published metrics table **to within 1 × 10⁻¹²**, and every file matches the SHA-256 manifest.

### 2 · Recompute the statistics from saved probabilities

No retraining required — this is the fastest fully deterministic route to the headline numbers.

```bash
pip install -r environment/requirements-corrections.txt
python src/run_validated_stats.py
```

Outputs are written to `reproduced-results/` for comparison against `results/tables/` and `results/figures/main/`.

### 3 · Conda alternative

```bash
conda env create -f environment/environment.yml
conda activate cvd-xai
```

---

## Reproduction levels

| Level | What it reproduces | Requirements | Dataset needed |
|:---:|---|---|:---:|
| **1** | File integrity, notebook validity, metric agreement | `requirements-verify.txt` | No |
| **2** | All metrics, 105 DeLong comparisons, BH/Holm, calibration, heatmap | `requirements-corrections.txt` | No |
| **3** | Corrected stacking, feature ablation, SAINT sensitivity | + XGBoost, LightGBM, CatBoost | Yes |
| **4** | Full notebook re-execution from scratch | Original Windows/CUDA environment | Yes |

Levels 1 and 2 run entirely from the deposited artifacts and need no access to the source data. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for level-by-level acceptance criteria.

---

## Notebook guide

The ten core notebooks preserve their original filenames and stored outputs. Recommended reading order:

| # | Notebook | Role |
|---:|---|---|
| 1 | [`Remove all of these bad rows.ipynb`](notebooks/core/) | Stage-1 blood-pressure plausibility filtering |
| 2 | [`Classical_Models_CVD.ipynb`](notebooks/core/) | Six classical models plus the MLP baseline |
| 3 | [`Stacking_Ensemble_CVD_GPU.ipynb`](notebooks/core/) | Gradient boosting and the original stacking workflow |
| 4 | [`TabNet_CVD_Clean.ipynb`](notebooks/core/) | TabNet with native attention-based importance |
| 5 | [`FT_Transformer_CVD.ipynb`](notebooks/core/) | FT-Transformer with KernelSHAP attribution |
| 6 | [`VIME_Pretraining_CVD_FIXED.ipynb`](notebooks/core/) | VIME-inspired self-supervised pretraining |
| 7 | [`SAINT_CVD_Complete.ipynb`](notebooks/core/) | Supervised SAINT-inspired pipeline |
| 8 | [`Heart_Disease_With_Optuna_GPU_Optimized.ipynb`](notebooks/core/) | Extended MLP / XAI / Optuna experiments |
| 9 | [`Statistical_Significance_Tests.ipynb`](notebooks/core/) | Original statistical notebook — superseded for final numbers |
| 10 | [`Versions.ipynb`](notebooks/core/) | Original environment capture |

> [!IMPORTANT]
> For the **final** DeLong, FDR, and calibration results, `src/run_validated_stats.py` is authoritative — not notebook 9. The notebooks are preserved as the evidentiary record of original execution; the portable scripts are the reproducible path.

PDF exports of nine of these runs are in [`notebooks/executed-pdf/`](notebooks/executed-pdf/) for reading without a Jupyter environment.

---

## Results index

| Artifact | Path | Generated by |
|---|---|---|
| Methodology workflow | `results/figures/main/Figure_1_Methodology_Workflow.png` | Study design source |
| AUC with 95% CI | `results/figures/main/Figure_2_Corrected_AUC_95CI.png` | Saved predictions + bootstrap |
| Top-six calibration curves | `results/figures/main/Figure_3_Top6_Calibration.png` | `src/run_validated_stats.py` |
| Paired DeLong heatmap | `results/figures/main/Figure_4_DeLong_Heatmap.png` / `.pdf` | `src/run_validated_stats.py` |
| 15-model performance table | `results/tables/validated_primary_metrics.csv` | `src/run_validated_stats.py` |
| All 105 DeLong comparisons | `results/tables/validated_delong_all_pairs.csv` | `src/run_validated_stats.py` |
| Top-six DeLong + BH | `results/tables/validated_delong_top6_bh.csv` | `src/run_validated_stats.py` |
| Advanced models vs. MLP | `results/tables/validated_delong_vs_mlp_bh.csv` | `src/run_validated_stats.py` |
| Exact calibration metrics | `results/tables/exact_primary_calibration.csv` | `src/run_validated_stats.py` |
| Cross-pipeline Spearman matrix | `results/tables/cross_pipeline_spearman_matrix.csv` | Executed XAI notebooks |
| Corrected stacking metrics | `results/tables/corrected_stacking_metrics.csv` | `src/run_corrected_stacking.py` |
| Stacking out-of-fold AUCs | `results/tables/corrected_stacking_oof_folds.csv` | `src/run_corrected_stacking.py` |
| Feature ablation | `results/tables/guideline_defined_feature_ablation.csv` | `src/run_guideline_ablation.py` |
| SAINT batch/order sensitivity | `results/tables/saint_batch_order_sensitivity.csv` | `src/run_saint_sensitivity.py` |

Full mapping: [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md).

---

## Data access and cohort construction

> [!NOTE]
> **Patient-level CSV files are intentionally not redistributed here.** The source record did not expose a redistribution licence that could be verified during packaging. To avoid implying rights the authors do not hold, this repository ships the processing code, schema, expected row counts, and checksums — but not the data itself.

**Source.** Cardiovascular Disease Dataset (Ulianova, 2019), Kaggle — 70,000 records × 11 columns.
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

**Two-stage cleaning.**

| Stage | Rule | Records retained |
|:---:|---|---:|
| Input | Converted 11-column study file | 70,000 |
| 1 | SBP ∈ [80, 200] mmHg and DBP ∈ [40, 130] mmHg | **68,653** |
| 2 | BMI ∈ [10, 60] kg/m² and pulse pressure (SBP − DBP) > 0 | **68,534** |

**To recreate the cohort:** download the source from Kaggle, convert it to the 11-column schema in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md), then run:

```bash
python src/prepare_data.py path/to/Heart_disease.csv
```

The script asserts each row count above and fails loudly on any mismatch. Place local files under `data/local/` — that path is gitignored.

> **On hash checking.** `prepare_data.py` prints the SHA-256 of the files it writes. These will **not** byte-match the reference hashes in [`data/README.md`](data/README.md), and that is expected — pandas re-serializes floating-point and quoting detail on write, so identical data produces different bytes (verified row-for-row with `pandas.testing.assert_frame_equal`). **Verify by row count, not by hash.** The published hashes exist only to confirm that a copy of a *supplied* file is the same file the authors used.

---

## Integrity and provenance

Every file in the release is covered by a deterministic SHA-256 manifest.

```bash
# Full release validation (recommended — checks far more than hashes)
python tools/verify_release.py

# Hash manifest only, Linux/macOS
sha256sum -c provenance/SHA256SUMS.txt
```

```powershell
# Windows PowerShell, single file
Get-FileHash -Algorithm SHA256 <path-to-file>
```

`tools/verify_release.py` performs six independent classes of check:

1. All required files and directories are present.
2. All ten core notebooks parse as valid JSON.
3. No core notebook contains a stored execution error.
4. Every core notebook contains executed cells with stored outputs.
5. The label vector is exactly 10,281 binary records, and all 15 model probability/prediction arrays reproduce `validated_primary_metrics.csv` to within 1 × 10⁻¹².
6. Every file hash matches `provenance/SHA256SUMS.txt`.

This runs automatically on every push and pull request via [`.github/workflows/validate.yml`](.github/workflows/validate.yml).

Per-file provenance — original source archive, size, and hash — is recorded in [`provenance/SOURCE_FILES.csv`](provenance/SOURCE_FILES.csv). The corrections applied during packaging are documented in [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md).

---

## Scope and limitations

Stated plainly, so results are not over-read:

- **Cross-sectional, not prospective.** The outcome is CVD status at examination, not an incident event. This is not a risk score.
- **One dataset, one fixed split.** All comparisons are internal. No external cohort validation was performed.
- **Upper-tier differences are not resolvable.** Differences among the top six models are smaller than between-split variation and are not statistically significant. Do not read the ranking as a recommendation.
- **Implementation-inspired, not canonical.** The VIME and SAINT pipelines are labelled *VIME-inspired SSL* and *SAINT-inspired*. They are variants, not canonical reimplementations of the published methods, and should not be cited as such.
- **No full retraining in the correction environment.** Deep and ensemble models were not retrained on CPU during correction; the repository therefore claims no formal equivalence among upper-tier models.
- **Explanations are not causal.** Attention weights and post-hoc attributions describe model behaviour, not disease mechanism. Cross-pipeline concordance measures agreement between explainers — not correctness of either.
- **`archive/` is provenance only.** It contains incomplete notebooks with stored errors and must never be used as an evidentiary source.

---

## Licensing

This repository uses **mixed licensing**:

| Component | Licence |
|---|---|
| Source code in `src/` and `tools/` | **MIT** |
| Documentation, figures, tables, prediction arrays, model artifacts | **CC BY 4.0** |
| Third-party datasets | Not included; remain subject to their source terms |

Full text: [`LICENSE.md`](LICENSE.md) and [`LICENSES/`](LICENSES/).

---

## Citation

If you use this repository, please cite the archived release.

**APA**

> Hossain, M. Z., Islam, M. S., Ratul, S. H., Ullah, S. M. A., & Rahman, M. A. (2026). *CVD XAI multi-paradigm benchmark reproducibility package* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22114347

**BibTeX**

```bibtex
@software{hossain_cvd_xai_2026,
  author    = {Hossain, Md. Zaid and
               Islam, Md. Shahriar and
               Ratul, Shahriar Hasan and
               Ullah, Shah Muhammad Azmat and
               Rahman, Md. Ashikur},
  title     = {{CVD XAI Cross-Pipeline Benchmark: A Reproducibility
                Package for Multi-Paradigm Cardiovascular Disease
                Status Classification}},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22114347},
  url       = {https://doi.org/10.5281/zenodo.22114347}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

Use the **version DOI** (`10.5281/zenodo.22114347`) to cite this exact release, or the **all-versions DOI** (`10.5281/zenodo.22114346`) to cite the project across all future releases.

**Dataset.** Please also cite the source data: Ulianova, S. (2019). *Cardiovascular disease dataset* [Data set]. Kaggle.

---

## Authors

All authors are at **Khulna University of Engineering and Technology (KUET)**, Khulna-9203, Bangladesh.

| Author | Department | Contact |
|---|---|---|
| **Md. Zaid Hossain** | Industrial Engineering and Management | [mdzaideng@gmail.com](mailto:mdzaideng@gmail.com) |
| **Md. Shahriar Islam** | Industrial Engineering and Management | [shahriarsanto06@gmail.com](mailto:shahriarsanto06@gmail.com) |
| **Shahriar Hasan Ratul** ✉ <br><sub>Lecturer</sub> | Industrial Engineering and Management | [shahriar@iem.kuet.ac.bd](mailto:shahriar@iem.kuet.ac.bd) |
| **Shah Muhammad Azmat Ullah** <br><sub>Assistant Professor</sub> | Electronics and Communication Engineering | [azmat@ece.kuet.ac.bd](mailto:azmat@ece.kuet.ac.bd) |
| **Md. Ashikur Rahman** | Industrial Engineering and Management | [ashikur.iem.kuet@gmail.com](mailto:ashikur.iem.kuet@gmail.com) |

<sub>✉ Corresponding author</sub>

---

## Reproducibility checklist

Before comparing a rerun against any value reported here:

- [ ] `python tools/verify_release.py` returns `RELEASE VALIDATION: PASS`
- [ ] The test-set label vector is 10,281 binary records
- [ ] Cohort counts are `70,000 → 68,653 → 68,534` (verify by row count, **not** by file hash)
- [ ] Statistics were computed with `src/run_validated_stats.py`, not the legacy notebook
- [ ] The corrected stacking array (`proba_stack_corrected.npy`) is in use — not `proba_stack_original.npy`
- [ ] Multiplicity correction is applied within the correct comparison family (top-six, vs-MLP, or all-105)
- [ ] Non-significance is not read as evidence of equivalence
- [ ] Rank order among the top six is not interpreted as a performance difference
- [ ] VIME and SAINT results are described as *inspired* variants, not canonical implementations
- [ ] Nothing from `archive/` is used as an evidentiary source

---

<p align="center">
  <sub><strong>Release v1.0.0</strong> · 26 August 2026 · <a href="CHANGELOG.md">Changelog</a> · <a href="docs/RELEASE_CHECKLIST.md">Release checklist</a></sub>
</p>
