#!/usr/bin/env python3
"""Protocol Section 3: retraining fidelity gate.

Implements docs/ANALYSIS_PROTOCOL.md Sections 1 (cohort/split), 2
(published model configurations) and 3 (fidelity gate, including the
Random Forest variant C / variant S rule). Retrains Random Forest, MLP
and XGBoost at seed 42 and compares their test-set probabilities with the
deposited artifacts/predictions/proba_{rf,mlp,xgb}.npy arrays.

Run from the repository root:
    python src/revision/section3_retraining_fidelity.py --data-file <path to stage-1 CSV>

Writes exactly the Section 7 outputs listed for Section 3.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

SBP = "ap_hi(Systolic blood pressure)"
DBP = "ap_lo(Diastolic blood pressure)"
CH = "cholesterol(1: normal, 2: above normal, 3: well above normal)"
GL = "Glucose(1: normal, 2: above normal, 3: well above normal)"

FEATURES = [
    "Gender (0: female, 1: male)", "age(years)", "BMI",
    SBP, DBP, CH, GL,
    "smoking", "alcohol", "active", "PP", "MAP", "risk_count",
]


def build_cohort(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Protocol Section 1.1."""
    df = pd.read_csv(data_file)
    if df.shape != (68653, 11):
        raise ValueError(f"stage-1 file shape {df.shape} != expected (68653, 11)")
    df["target"] = (df["cardiovascular disease"] == "has cardiovascular disease").astype(int)
    before = len(df)
    df = df[df["BMI"].between(10, 60) & ((df[SBP] - df[DBP]) > 0)].reset_index(drop=True)
    if len(df) != 68534:
        raise ValueError(f"cohort after cleaning has {len(df)} rows, expected 68,534 (removed {before - len(df)})")
    df["PP"] = df[SBP] - df[DBP]
    df["MAP"] = (df[SBP] + 2 * df[DBP]) / 3
    df["risk_count"] = (
        (df[CH] > 1).astype(int) + (df[GL] > 1).astype(int)
        + df["smoking"] + df["alcohol"] + (1 - df["active"])
    )
    X = df[FEATURES].values
    y = df["target"].values
    return X, y


def primary_split(X: np.ndarray, y: np.ndarray):
    """Protocol Section 1.2."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_tmp
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    return (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc,
            y_train, y_val, y_test, scaler)


def fit_rf(X_train_sc: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Protocol Section 2.1 (variant C and variant S share this configuration)."""
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)
    return model


def fit_mlp(X_train_sc: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """Protocol Section 2.2."""
    model = MLPClassifier(
        hidden_layer_sizes=(16, 16), activation="relu", alpha=1e-5,
        learning_rate_init=0.001, solver="adam", max_iter=500,
        early_stopping=True, validation_fraction=0.1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_sc, y_train)
    return model


def fit_xgb(X_train, y_train, X_val, y_val, device: str) -> xgb.XGBClassifier:
    """Protocol Section 2.3."""
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5, eval_metric="auc",
        early_stopping_rounds=30, random_state=RANDOM_STATE, verbosity=0,
        tree_method="hist", device=device,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fidelity_row(name: str, deposited: np.ndarray, retrained: np.ndarray, y_test: np.ndarray,
                  device: str, library: str, library_version: str, extra: dict) -> dict:
    max_abs_diff = float(np.max(np.abs(deposited - retrained)))
    corr = float(pearsonr(deposited, retrained)[0])
    auc_deposited = float(roc_auc_score(y_test, deposited))
    auc_retrained = float(roc_auc_score(y_test, retrained))
    delta_auc = auc_retrained - auc_deposited
    passed = (abs(delta_auc) <= 0.001) and (corr >= 0.999)
    row = {
        "model": name, "max_abs_diff": max_abs_diff, "pearson_correlation": corr,
        "deposited_auc": auc_deposited, "retrained_auc": auc_retrained,
        "delta_auc": delta_auc, "pass": passed,
        "device": device, "library": library, "library_version": library_version,
    }
    row.update(extra)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True,
                         help="Local stage-1 CSV (68,653 rows, 11 columns)")
    parser.add_argument("--predictions-dir", type=Path,
                         default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--models-dir", type=Path,
                         default=ROOT / "artifacts" / "models-revision")
    parser.add_argument("--output-dir", type=Path,
                         default=ROOT / "revision-results" / "fidelity")
    parser.add_argument("--xgb-device", default="cuda",
                         help="XGBoost device (Section 6 device policy: GPU preferred)")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Python {sys.version}")
    print(f"numpy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | xgboost {xgb.__version__}")
    print(f"Platform: {platform.platform()}")

    X, y = build_cohort(args.data_file)
    (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc,
     y_train, y_val, y_test, scaler) = primary_split(X, y)

    # Mandatory split identity check (Section 1.2).
    deposited_y_test = np.load(args.predictions_dir / "y_test.npy").astype(int)
    if not np.array_equal(y_test, deposited_y_test):
        raise SystemExit(
            "SPLIT IDENTITY CHECK FAILED: regenerated y_test does not equal "
            "artifacts/predictions/y_test.npy element-wise. Stopping per Section 1.2."
        )
    print(f"Split identity check: PASS ({len(y_test):,} test records match deposited y_test.npy)")

    joblib.dump(scaler, args.models_dir / "scaler_seed42.joblib")

    fidelity_rows = []

    # --- Random Forest: variant C first (Section 3, step 4) ---
    print("\nFitting Random Forest (variant C, Classical_Models_CVD.ipynb cell 17) ...")
    t0 = time.time()
    rf_c = fit_rf(X_train_sc, y_train)
    proba_rf_c = rf_c.predict_proba(X_test_sc)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s")
    deposited_rf = np.load(args.predictions_dir / "proba_rf.npy").astype(float)

    rf_c_row = fidelity_row(
        "Random Forest (variant C)", deposited_rf, proba_rf_c, y_test,
        device="cpu", library="scikit-learn", library_version=sklearn.__version__, extra={},
    )
    rf_variant_rows = [{**rf_c_row, "variant": "C", "n_estimators_match": True}]

    rf_selected = rf_c
    rf_selected_proba = proba_rf_c
    rf_selected_variant = "C"
    np.save(args.output_dir / "proba_rf_retrained_seed42.npy", proba_rf_c)

    if not rf_c_row["pass"]:
        print("  variant C FAILED the fidelity criteria -> fitting variant S "
              "(Stacking_Ensemble_CVD_GPU.ipynb cell 18) ...")
        t0 = time.time()
        rf_s = fit_rf(X_train_sc, y_train)  # identical config/split/scaler code
        proba_rf_s = rf_s.predict_proba(X_test_sc)[:, 1]
        print(f"  done in {time.time() - t0:.1f}s")
        rf_s_row = fidelity_row(
            "Random Forest (variant S)", deposited_rf, proba_rf_s, y_test,
            device="cpu", library="scikit-learn", library_version=sklearn.__version__, extra={},
        )
        rf_variant_rows.append({**rf_s_row, "variant": "S", "n_estimators_match": True})
        np.save(args.output_dir / "proba_rf_variantS_seed42.npy", proba_rf_s)
        if rf_s_row["pass"]:
            rf_selected, rf_selected_proba, rf_selected_variant = rf_s, proba_rf_s, "S"
            np.save(args.output_dir / "proba_rf_retrained_seed42.npy", proba_rf_s)
        # Discrepancy only if BOTH variants fail (Section 3, step 4).
        rf_discrepancy = (not rf_c_row["pass"]) and (not rf_s_row["pass"])
    else:
        rf_discrepancy = False

    # lineterminator="\n": LF on every OS, matching .gitattributes (*.csv eol=lf).
    pd.DataFrame(rf_variant_rows).to_csv(
        args.output_dir / "rf_variant_check.csv", index=False, lineterminator="\n"
    )
    joblib.dump(rf_selected, args.models_dir / "rf_seed42.joblib")

    final_rf_row = fidelity_row(
        "Random Forest", deposited_rf, rf_selected_proba, y_test,
        device="cpu", library="scikit-learn", library_version=sklearn.__version__,
        extra={"selected_variant": rf_selected_variant, "both_variants_failed": rf_discrepancy},
    )
    fidelity_rows.append(final_rf_row)

    # --- MLP ---
    print("\nFitting MLP (Classical_Models_CVD.ipynb cell 19) ...")
    t0 = time.time()
    mlp = fit_mlp(X_train_sc, y_train)
    proba_mlp = mlp.predict_proba(X_test_sc)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s")
    deposited_mlp = np.load(args.predictions_dir / "proba_mlp.npy").astype(float)
    np.save(args.output_dir / "proba_mlp_retrained_seed42.npy", proba_mlp)
    joblib.dump(mlp, args.models_dir / "mlp_seed42.joblib")
    fidelity_rows.append(fidelity_row(
        "MLP", deposited_mlp, proba_mlp, y_test,
        device="cpu", library="scikit-learn", library_version=sklearn.__version__, extra={},
    ))

    # --- XGBoost ---
    print(f"\nFitting XGBoost (Stacking_Ensemble_CVD_GPU.ipynb cell 15), device={args.xgb_device} ...")
    t0 = time.time()
    xgb_model = fit_xgb(X_train, y_train, X_val, y_val, device=args.xgb_device)
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s, best_iteration={xgb_model.best_iteration}")
    deposited_xgb = np.load(args.predictions_dir / "proba_xgb.npy").astype(float)
    np.save(args.output_dir / "proba_xgb_retrained_seed42.npy", proba_xgb)
    xgb_model.save_model(str(args.models_dir / "xgb_seed42.json"))
    fidelity_rows.append(fidelity_row(
        "XGBoost", deposited_xgb, proba_xgb, y_test,
        device=args.xgb_device, library="xgboost", library_version=xgb.__version__,
        extra={
            "best_iteration": int(xgb_model.best_iteration),
            "published_best_iteration": 185,
            "best_iteration_matches_published": int(xgb_model.best_iteration) == 185,
        },
    ))

    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_df.to_csv(
        args.output_dir / "retrain_fidelity_seed42.csv", index=False, lineterminator="\n"
    )

    print("\n" + "=" * 78)
    print("SECTION 3 RETRAINING FIDELITY GATE -- SUMMARY")
    print("=" * 78)
    print(fidelity_df.to_string(index=False))
    print(f"\nRandom Forest: variant C pass = {rf_c_row['pass']}, "
          f"selected variant = {rf_selected_variant}, "
          f"both variants failed (discrepancy) = {rf_discrepancy}")


if __name__ == "__main__":
    main()
