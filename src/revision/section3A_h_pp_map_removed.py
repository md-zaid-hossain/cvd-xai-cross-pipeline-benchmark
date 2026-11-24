#!/usr/bin/env python3
"""Protocol Section 3A(h): PP/MAP-removed importance check (Random Forest only).

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(h).
- 11 features = Section 1.1 matrix minus PP and MAP (risk_count kept).
- Seed-42 split with the identical two train_test_split calls on the
  11-column matrix; checks: y_test equals artifacts/predictions/y_test.npy and
  the 11 shared test columns equal those of the 13-feature X_test.
- StandardScaler fitted on the 11-feature training set; Random Forest with the
  Section 2.1 configuration, random_state=42.
- Permutation importance on the Section 4.1 common sample (read from file,
  hash-checked), for the 11-feature RF and the Section 3 13-feature RF:
      permutation_importance(model, X_sc[xai_idx], y_test[xai_idx],
          scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1)
- Ranks: scipy.stats.rankdata(-mean, method='average'):
  (i) of 11 in the 11-feature model; (ii) of 13 in the 13-feature model;
  (iii) of the 11 shared features in the 13-feature model.
- Prespecified question: does Cholesterol rise in rank once PP and MAP are
  removed? Compare (iii) with (i); "rises" = strictly smaller rank in (i).

Run from the repository root:
    python src/revision/section3A_h_pp_map_removed.py --data-file <stage-1 CSV>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "revision"))
from common_sample import load_common_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
              "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
KEEP = [i for i, f in enumerate(FEAT_NAMES) if f not in ("PP", "MAP")]  # 11 columns, Section 1.1 order
NAMES_11 = [FEAT_NAMES[i] for i in KEEP]
TABLE_S7 = {"10 variables (no PP, MAP, risk_count)": 0.799843,
            "12 variables (original + PP + MAP)": 0.799302,
            "13 variables (full)": 0.799534}


def answer(r_iii: float, r_i: float) -> str:
    return "yes" if r_i < r_iii else ("unchanged" if r_i == r_iii else "no")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--rf13-file", type=Path, default=ROOT / "artifacts" / "models-revision" / "rf_seed42.joblib")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "artifacts" / "models-revision")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()

    xai_idx, sample_sha = load_common_sample()
    X, y = s3.build_cohort(args.data_file)
    deposited_y = np.load(args.predictions_dir / "y_test.npy").astype(int)

    # 13-feature split (Section 1.2) for the reference RF and the column check.
    parts13 = s3.primary_split(X, y)
    X_test13, X_test13_sc, y_test13 = parts13[2], parts13[5], parts13[8]

    # 11-feature split: identical calls on the 11-column matrix.
    X11 = X[:, KEEP]
    X_tmp, X11_test, y_tmp, y_test = train_test_split(X11, y, test_size=0.15, random_state=42, stratify=y)
    X11_train, X11_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp)
    checks = {
        "y_test_equals_deposited": bool(np.array_equal(y_test, deposited_y)),
        "y_test_equals_13_feature_split": bool(np.array_equal(y_test, y_test13)),
        "shared_11_test_columns_equal_13_feature_X_test": bool(np.array_equal(X11_test, X_test13[:, KEEP])),
    }
    if not all(checks.values()):
        raise SystemExit(f"IDENTITY CHECK FAILED: {checks}")

    scaler11 = StandardScaler()
    X11_train_sc = scaler11.fit_transform(X11_train)
    X11_test_sc = scaler11.transform(X11_test)
    rf11 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                  max_features="sqrt", random_state=42, n_jobs=-1)
    rf11.fit(X11_train_sc, y_train)
    joblib.dump(rf11, args.models_dir / "rf_11feat_seed42.joblib")
    joblib.dump(scaler11, args.models_dir / "scaler_11feat_seed42.joblib")

    rf13 = joblib.load(args.rf13_file)
    auc11 = roc_auc_score(y_test, rf11.predict_proba(X11_test_sc)[:, 1])
    auc13 = roc_auc_score(y_test, rf13.predict_proba(X_test13_sc)[:, 1])

    pi11 = permutation_importance(rf11, X11_test_sc[xai_idx], y_test[xai_idx],
                                  scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
    pi13 = permutation_importance(rf13, X_test13_sc[xai_idx], y_test[xai_idx],
                                  scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)

    r_i = rankdata(-pi11.importances_mean, method="average")              # of 11
    r_ii = rankdata(-pi13.importances_mean, method="average")             # of 13
    r_iii = rankdata(-pi13.importances_mean[KEEP], method="average")      # of 11 shared, 13-feature model

    rows = []
    for f in FEAT_NAMES:
        j13 = FEAT_NAMES.index(f)
        row = {"feature": f,
               "rf13_mean_auc_drop": float(pi13.importances_mean[j13]),
               "rf13_sd_auc_drop": float(pi13.importances_std[j13]),
               "rank_ii_of_13_in_13_feature_model": float(r_ii[j13])}
        if f in NAMES_11:
            k = NAMES_11.index(f)
            row.update({"rf11_mean_auc_drop": float(pi11.importances_mean[k]),
                        "rf11_sd_auc_drop": float(pi11.importances_std[k]),
                        "rank_i_of_11_in_11_feature_model": float(r_i[k]),
                        "rank_iii_of_11_shared_in_13_feature_model": float(r_iii[k]),
                        "rank_change_iii_minus_i": float(r_iii[k] - r_i[k]),
                        "rises_in_rank": answer(r_iii[k], r_i[k])})
        else:
            row["rises_in_rank"] = "removed (not in 11-feature model)"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "pp_map_removed_rf_importance.csv", index=False, lineterminator="\n")

    chol = table.set_index("feature").loc["Cholesterol"]
    summary = {
        "protocol_section": "3A(h)",
        "common_sample_sha256": sample_sha, "common_sample_n": int(len(xai_idx)),
        "identity_checks": checks,
        "features_11": NAMES_11,
        "test_auc_full_10281": {"rf_11_features_pp_map_removed": auc11, "rf_13_features_section3": auc13,
                                "difference_11_minus_13": auc11 - auc13},
        "table_s7_context_values": TABLE_S7,
        "table_s7_note": "No Table S7 row matches the 11-feature set (Table S7's 10-variable row also drops risk_count); the 11-feature AUC is a new value.",
        "permutation_importance": "sklearn permutation_importance, scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1, on the Section 4.1 common sample",
        "rank_method": "scipy.stats.rankdata(-mean, method='average')",
        "cholesterol": {
            "rank_i_of_11_in_11_feature_model": float(chol["rank_i_of_11_in_11_feature_model"]),
            "rank_iii_of_11_shared_in_13_feature_model": float(chol["rank_iii_of_11_shared_in_13_feature_model"]),
            "rank_ii_of_13_in_13_feature_model": float(chol["rank_ii_of_13_in_13_feature_model"]),
            "does_cholesterol_rise_in_rank": chol["rises_in_rank"],
            "rule": "compare (iii) with (i); 'rises' = strictly smaller rank number in the 11-feature model",
        },
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "device": "cpu (scikit-learn)",
    }
    with open(args.output_dir / "pp_map_removed_rf_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    pd.set_option("display.width", 220)
    print(json.dumps({k: summary[k] for k in ("identity_checks", "test_auc_full_10281", "cholesterol")}, indent=2))
    print(table[["feature", "rf13_mean_auc_drop", "rf13_sd_auc_drop", "rf11_mean_auc_drop", "rf11_sd_auc_drop",
                 "rank_ii_of_13_in_13_feature_model", "rank_iii_of_11_shared_in_13_feature_model",
                 "rank_i_of_11_in_11_feature_model", "rises_in_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
