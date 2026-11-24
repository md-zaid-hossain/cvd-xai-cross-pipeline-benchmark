# Original execution environment

The executed core notebooks recorded the following environment:

- Windows 11
- Python 3.13.9 (Anaconda)
- NumPy 2.3.5
- pandas 2.3.3
- scikit-learn 1.7.2
- PyTorch 2.6.0+cu124
- SHAP 0.51.0
- NVIDIA GeForce GTX 1650

Additional packages installed or imported by individual notebooks include XGBoost, LightGBM, CatBoost, pytorch-tabnet, Optuna, Captum, LIME, DiCE-ML, Gradio, seaborn, matplotlib, and einops. GPU/CUDA builds should be installed using the package vendor's instructions for the target system rather than assuming the CPU correction environment is equivalent.

The later correction environment recorded in the manuscript used Linux, Python 3.13.5, pandas 2.2.3, scikit-learn 1.8.0, PyTorch 2.10.0+cpu, XGBoost 3.1.3, LightGBM 4.6.0, CatBoost 1.2.8, and SHAP 0.50.0.

## SHAP version note

The SHAP values reported in the manuscript were computed with **SHAP 0.51.0**, in the original notebooks (`Stacking_Ensemble_CVD_GPU.ipynb` cell 29: `shap.TreeExplainer(xgb_model)`, version recorded in `Versions.ipynb`). The revision analyses that compute SHAP values (Section 4.2 and Amendment 9) also ran with 0.51.0, in the original Anaconda environment.

The correction environment (`environment/requirements-corrections.txt` and its conda mirror `environment/environment.yml`) pins **SHAP 0.50.0**, matching the correction environment recorded in the manuscript. That environment computes no SHAP values, and the pins are deliberately left unchanged.

In both versions `shap.TreeExplainer(model)` called without background data resolves its default `feature_perturbation="auto"` to **`tree_path_dependent`**. This was checked against the v0.50.0 source (`shap/explainers/_tree.py`, `"auto"` added in 0.47) and the installed 0.51.0 code. The SHAP rankings are therefore unaffected by the version difference.
