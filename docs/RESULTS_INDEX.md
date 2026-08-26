# Manuscript-to-artifact results index

| Manuscript item | Repository artifact | Reproducible from |
| --- | --- | --- |
| Figure 1, methodology workflow | `results/figures/main/Figure_1_Methodology_Workflow.png` | Design source / study workflow |
| Table 3, cohort characteristics | Manuscript table | Prepared analytical cohort |
| Table 4, 15-model performance | `results/tables/validated_primary_metrics.csv` | `src/run_validated_stats.py` |
| Figure 2, AUC with 95% CI | `results/figures/main/Figure_2_Corrected_AUC_95CI.png` | Saved predictions + bootstrap analysis |
| Table 5, top-six calibration | `results/tables/exact_primary_calibration.csv` | `src/run_validated_stats.py` |
| Figure 3, calibration curves | `results/figures/main/Figure_3_Top6_Calibration.png` | `src/run_validated_stats.py` |
| Figure 4, paired DeLong heatmap | `results/figures/main/Figure_4_DeLong_Heatmap.png` | `src/run_validated_stats.py` |
| Table 6, robustness/sensitivity | Ablation, stacking, and SAINT CSV files | Portable scripts in `src/` |
| Table 7, feature ranks | Manuscript table and cross-pipeline matrix | Executed XAI notebooks and `cross_pipeline_spearman_matrix.csv` |
| Supplement Table S1 | `corrected_stacking_metrics.csv` | `src/run_corrected_stacking.py` |
| Supplement Table S1a | `corrected_stacking_oof_folds.csv` | `src/run_corrected_stacking.py` |
| Supplement Tables S2-S3 | `validated_delong_vs_mlp_bh.csv`, `validated_delong_top6_bh.csv` | `src/run_validated_stats.py` |
| Supplement Table S4 | `exact_primary_calibration.csv` | `src/run_validated_stats.py` |
| Supplement Table S5 | `guideline_defined_feature_ablation.csv` | `src/run_guideline_ablation.py` |
| Supplement Table S6 | `saint_batch_order_sensitivity.csv` | `src/run_saint_sensitivity.py` |

The learning-curve and five-seed sensitivity numbers described in the manuscript are retained in the manuscript/supplementary material but were not supplied as standalone result files in the source package.
