# Authoritative executed notebooks

These ten notebooks are the authoritative stored execution record supplied with the final analysis package. Each notebook parses correctly, contains stored outputs, and contains no stored exception.

Recommended order:

1. `Remove all of these bad rows.ipynb` - stage-1 blood-pressure filtering.
2. `Classical_Models_CVD.ipynb` - classical models and MLP.
3. `Stacking_Ensemble_CVD_GPU.ipynb` - boosting and original stacking workflow.
4. `TabNet_CVD_Clean.ipynb` - TabNet and native attention.
5. `FT_Transformer_CVD.ipynb` - FT-Transformer and KernelSHAP.
6. `VIME_Pretraining_CVD_FIXED.ipynb` - VIME-inspired self-supervised pipeline.
7. `SAINT_CVD_Complete.ipynb` - supervised SAINT-inspired pipeline.
8. `Heart_Disease_With_Optuna_GPU_Optimized.ipynb` - extended MLP/XAI/Optuna experiments.
9. `Statistical_Significance_Tests.ipynb` - original statistical notebook; the portable corrected script in `src/` is authoritative for the final DeLong/FDR results.
10. `Versions.ipynb` - original environment capture.

The notebooks retain original filenames to preserve provenance. They write files to the current working directory and may include package-install cells. For deterministic final statistics, use `src/run_validated_stats.py` with the deposited arrays.
