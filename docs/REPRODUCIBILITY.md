# Reproducibility guide

## Environment options

Two equivalent routes are provided. Use whichever matches your setup:

| Route | Command | Covers |
| --- | --- | --- |
| pip, verification only | `pip install -r environment/requirements-verify.txt` | Level 1 |
| pip, full reproduction | `pip install -r environment/requirements-corrections.txt` | Levels 2-3 |
| conda, full reproduction | `conda env create -f environment/environment.yml` then `conda activate cvd-xai` | Levels 2-3 |

The conda file installs PyTorch CPU-only by default; see the comments inside it for the GPU variant.

## Level 1: verify the deposited evidence

Install `environment/requirements-verify.txt` and run `python tools/verify_release.py`. This validates file integrity, core notebook JSON, stored-error absence, prediction-array dimensions and ranges, and exact agreement between the included arrays and `validated_primary_metrics.csv`.

## Level 2: reproduce statistics from saved test probabilities

Install `environment/requirements-corrections.txt` and run `python src/run_validated_stats.py`. This regenerates:

- metrics for all 15 models;
- all 105 paired DeLong comparisons using midranks and paired covariance;
- Benjamini-Hochberg and Holm adjustments;
- the nine advanced/specialised-model comparisons against MLP;
- the focused 15 top-six comparisons;
- exact-probability calibration metrics and calibration curves;
- the validated DeLong heatmap.

This is the fastest and most deterministic route because it does not retrain models.

## Level 3: reproduce corrected secondary analyses

After preparing `data/local/Heart_disease_clean_stage1.csv`:

```bash
python src/run_guideline_ablation.py
python src/run_corrected_stacking.py
python src/run_saint_sensitivity.py
```

The stacking run is more computationally expensive and requires XGBoost, LightGBM, and CatBoost. The SAINT-inspired sensitivity run uses the included checkpoint and is sensitive to PyTorch version and batch implementation.

## Level 4: rerun the original notebooks

Copy the local stage-1 CSV into the notebook working directory as `Heart_disease_clean.csv`. Some notebooks also consume or write `.npy` and model files in the current directory. Use the original Windows/CUDA environment described in `environment/ORIGINAL_ENVIRONMENT.md` for the closest match.

The stored notebook outputs are the evidence of original execution. Empty trailing cells and setup cells with `execution_count: null` are retained as originally supplied; no authoritative core notebook contains a stored exception.

## Methodological caveats

- The outcome is cross-sectional CVD status, not a prospective event endpoint.
- Model comparisons are internal to one dataset and fixed split.
- Upper-tier differences are smaller than the reported between-split standard deviations.
- The VIME and SAINT pipelines are implementation-inspired variants, not canonical implementations.
- Attention and post-hoc feature attribution should not be interpreted as causality.
