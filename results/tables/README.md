# Validated result tables

| File | Description |
| --- | --- |
| `validated_primary_metrics.csv` | AUC, accuracy, F1, precision, recall, and Brier score for 15 models |
| `validated_delong_all_pairs.csv` | All 105 paired DeLong comparisons with global BH and Holm adjustments |
| `validated_delong_vs_mlp_bh.csv` | Nine focused advanced/specialised comparisons against MLP |
| `validated_delong_top6_bh.csv` | Fifteen focused comparisons among the six highest-AUC models |
| `exact_primary_calibration.csv` | AUC, Brier score, calibration intercept, and calibration slope |
| `corrected_stacking_metrics.csv` | Corrected stacking test metrics and meta-learner coefficients |
| `corrected_stacking_oof_folds.csv` | Fold-specific out-of-fold base-model AUCs |
| `guideline_defined_feature_ablation.csv` | Random Forest results across five clinically defined feature sets |
| `saint_batch_order_sensitivity.csv` | SAINT-inspired sensitivity to batch size and row order |
| `cross_pipeline_spearman_matrix.csv` | Pairwise Spearman correlations across five feature-ranking pipelines |
| `validated_stats_corrected_stack.txt` | Console summary retained from the correction run |

The authoritative Stacking Ensemble values use `proba_stack_corrected.npy` and `pred_stack_corrected.npy`. Original stacking arrays are retained in `artifacts/predictions/` only for audit comparison.
