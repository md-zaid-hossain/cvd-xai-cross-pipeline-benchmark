# CVD XAI benchmark: reproducibility package

This repository accompanies **“Explainable Machine Learning for Cardiovascular Disease Status Classification: A Multi-Paradigm Benchmark with Cross-Pipeline Feature-Ranking Analysis.”** It consolidates the executed notebooks, portable correction scripts, validated prediction arrays, saved model artifacts, final figures, final tables, and the manuscript/supplementary files used for the 26 August 2026 submission package.

## Headline result

Fifteen supervised classifiers were evaluated on the same stratified held-out test set (`n = 10,281`). The corrected Stacking Ensemble had the highest ROC-AUC (`0.800367`; bootstrap 95% CI `0.791538-0.809095`), but the six highest-ranked models spanned only about `0.0008` AUC and none of their 15 pairwise DeLong comparisons was statistically significant. Exact test probabilities, binary predictions, calibration results, and the complete paired-test tables are included.

> **Research-use warning:** This is a retrospective CVD-status classification benchmark, not a prospective clinical risk score, diagnostic device, or clinical decision-support system. External validation is required before clinical interpretation or deployment.

## Repository map

| Path | Contents |
| --- | --- |
| `manuscript/` | Final manuscript and supplementary material dated 2026-08-26 |
| `notebooks/core/` | Ten authoritative notebooks with stored outputs and no stored execution errors |
| `notebooks/executed-pdf/` | Human-readable PDF exports of the principal notebook runs |
| `src/` | Portable data-preparation and corrected-analysis scripts |
| `results/figures/` | Main figures, graphical abstract, editable SVG, and alternate heatmaps |
| `results/tables/` | Validated metrics, DeLong/FDR results, calibration, ablation, and sensitivity tables |
| `artifacts/predictions/` | Common-test-set probabilities and predictions for all 15 models |
| `artifacts/models/` | Saved FT-Transformer, TabNet, VIME-inspired, and SAINT-inspired artifacts |
| `data/` | Source, licence, schema, cleaning, and checksum instructions; patient-level CSVs are not redistributed |
| `archive/` | Older, non-authoritative notebooks and the original hard-coded validation script retained for provenance |
| `docs/` | Reproduction guide, audit report, results index, and release checklist |
| `provenance/` | Source archive hashes and deterministic SHA-256 release manifest |

## Quick verification

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r environment/requirements-verify.txt
python tools/verify_release.py
```

Expected terminal message: `RELEASE VALIDATION: PASS`.

Conda users can instead create the full environment once with
`conda env create -f environment/environment.yml` followed by `conda activate cvd-xai`.
`docs/REPRODUCIBILITY.md` lists which environment covers which reproduction level.

## Recompute the validated statistics

The primary statistical analysis uses the included prediction arrays and does not require retraining:

```bash
python -m pip install -r environment/requirements-corrections.txt
python src/run_validated_stats.py
```

New files are written to `reproduced-results/`. Compare them with `results/tables/` and `results/figures/main/`.

## Recreate the analytical cohort

Patient-level CSV files are intentionally excluded because the Kaggle record does not expose a redistribution licence that could be verified during packaging. See `data/README.md`, download the source from Kaggle, convert it to the 11-column schema in `docs/DATA_DICTIONARY.md`, and run:

```bash
python src/prepare_data.py path/to/Heart_disease.csv
```

This must produce 68,653 rows after the blood-pressure bounds and 68,534 rows after the BMI and positive-pulse-pressure checks.

## Reproducibility scope

- The notebooks preserve the original executed outputs and original Windows/CUDA environment.
- The correction scripts are path-portable and reproduce the corrected stacking, DeLong/FDR, calibration, Random Forest ablation, and SAINT-inspired sensitivity analyses.
- Full retraining of every deep/ensemble model was not repeated in the CPU correction environment. The repository therefore does not claim formal equivalence among the upper-tier models.
- The implementations are accurately labelled **VIME-inspired SSL** and **SAINT-inspired**; they are not claimed to be canonical reimplementations of the original methods.
- `archive/legacy-notebooks/` is provenance only. It contains incomplete notebooks and stored errors and must not be used as the evidentiary source for manuscript results.

## Citation and DOI workflow

The root `CITATION.cff` enables GitHub's citation panel. After creating the GitHub repository, add its URL to `CITATION.cff` and `.zenodo.json`. Connect the repository to Zenodo, create the GitHub release `v1.0.0`, and then add the assigned Zenodo DOI to the manuscript, supplementary file, repository description, and citation metadata.

## Licences

- Code in `src/` and `tools/`: MIT License.
- Original documentation, figures, tables, manuscript material, prediction arrays, and model artifacts: CC BY 4.0.
- Third-party datasets are not included and remain subject to their source terms.

See `LICENSE.md` and `LICENSES/` for details.

## Authors

Md. Zaid Hossain; Md. Shahriar Islam; Shahriar Hasan Ratul; Shah Muhammad Azmat Ullah; Md. Ashikur Rahman.

Affiliation: Khulna University of Engineering and Technology, Khulna-9203, Bangladesh.
