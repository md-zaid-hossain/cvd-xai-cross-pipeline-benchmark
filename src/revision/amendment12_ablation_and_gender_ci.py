#!/usr/bin/env python3
"""Protocol Amendment 12 (prespecified, pushed alone in c5f2041): ablation and gender-difference CIs.

(i)   Refit the five guideline-defined ablation Random Forests (3, 6, 10, 12, 13 features) exactly as
      src/run_guideline_ablation.py fits them, deposit their test-set probability arrays, and check each
      refit AUC against results/tables/guideline_defined_feature_ablation.csv at six decimal places.
(ii)  Score the deposited 11-feature model (artifacts/models-revision/rf_11feat_seed42.joblib with its own
      scaler, applied, never refitted) on the rebuilt 11-column primary test set; inference, not a refit.
      Match requirement against 0.799994 at six decimal places.
(iii) (a) Percentile bootstrap AUC CIs for each matched feature set (Section 3A(b) convention: 2,000
      resamples, seed 42, one shared index vector per replicate).
      (b) Male-minus-female AUC-difference CIs for the six leading models (Amendment 11 convention:
      2,000 resamples, seed 42, within-group resampling, female replicates drawn first), with a check
      that the per-group CIs reproduce revision-results/lowcompute/gender_stratified_performance.csv.

A failed identity check stops the run. A failed match requirement stops the interval for that set only.
The E1/E2 verdicts are computed mechanically from the prespecified refutation criteria.

Run from the repository root:
    python src/revision/amendment12_ablation_and_gender_ci.py --data-file <stage-1 CSV>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "artifacts" / "predictions"
MODELS_REV = ROOT / "artifacts" / "models-revision"
OUT_ABL = ROOT / "revision-results" / "ablation"
OUT_LC = ROOT / "revision-results" / "lowcompute"
_spec = importlib.util.spec_from_file_location("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)

AMENDMENT_COMMIT = "c5f2041e91798cf730008cd0e577d932ceaf646d"
N_BOOT, BOOT_SEED = 2000, 42
TARGET_11 = 0.799994  # pp_map_removed_rf_summary.json: 0.7999943341455275

# Exactly as src/run_guideline_ablation.py
SBP = "ap_hi(Systolic blood pressure)"; DBP = "ap_lo(Diastolic blood pressure)"
CH = "cholesterol(1: normal, 2: above normal, 3: well above normal)"
GL = "Glucose(1: normal, 2: above normal, 3: well above normal)"
G = "Gender (0: female, 1: male)"; AGE = "age(years)"
SUBSETS = {
    "Core clinical 3 (Age, SBP, Cholesterol)": [AGE, SBP, CH],
    "Guideline-aligned 6 (Age, Sex, SBP, Cholesterol, Glucose, Smoking)": [AGE, G, SBP, CH, GL, "smoking"],
    "Original 10 variables": [G, AGE, "BMI", SBP, DBP, CH, GL, "smoking", "alcohol", "active"],
    "Original + PP + MAP (12 variables)": [G, AGE, "BMI", SBP, DBP, CH, GL, "smoking", "alcohol", "active", "PP", "MAP"],
    "Full 13 variables": [G, AGE, "BMI", SBP, DBP, CH, GL, "smoking", "alcohol", "active", "PP", "MAP", "risk_count"],
}
FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
              "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
KEEP = [i for i, f in enumerate(FEAT_NAMES) if f not in ("PP", "MAP")]
LEADING = {"Stacking Ensemble": "stack_corrected", "XGBoost": "xgb", "CatBoost": "cat",
           "FT-Transformer": "ft", "SAINT-inspired": "saint", "Random Forest": "rf"}
GROUPS = {"female": 0, "male": 1}


def r6(x: float) -> float:
    return round(float(x), 6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    a = ap.parse_args()
    OUT_ABL.mkdir(parents=True, exist_ok=True)

    y_dep = np.load(PRED / "y_test.npy").astype(int)
    table = pd.read_csv(ROOT / "results" / "tables" / "guideline_defined_feature_ablation.csv")
    table_auc = dict(zip(table["Subset"], table["Test AUC"]))

    # ---- (i) refit, exactly as src/run_guideline_ablation.py ----
    df = pd.read_csv(a.data_file)
    df = df[df.BMI.between(10, 60) & ((df[SBP] - df[DBP]) > 0)].reset_index(drop=True)
    y = (df["cardiovascular disease"] == "has cardiovascular disease").astype(int).values
    df["PP"] = df[SBP] - df[DBP]; df["MAP"] = (df[SBP] + 2 * df[DBP]) / 3
    df["risk_count"] = (df[CH] > 1).astype(int) + (df[GL] > 1).astype(int) + df.smoking + df.alcohol + (1 - df.active)

    identity, sets, proba = {}, [], {}
    for name, fs in SUBSETS.items():
        X = df[fs].values
        Xt, Xte, yt, yte = train_test_split(X, y, test_size=.15, random_state=42, stratify=y)
        Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=.1765, random_state=42, stratify=yt)
        identity[f"y_test_equals_deposited__{len(fs)}feat"] = bool(np.array_equal(yte, y_dep))
        if not identity[f"y_test_equals_deposited__{len(fs)}feat"]:
            raise SystemExit(f"IDENTITY CHECK FAILED for {name}")
        m = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, max_features="sqrt",
                                   random_state=42, n_jobs=-1).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        key = f"{len(fs)}"
        np.save(OUT_ABL / f"proba_ablation_{key}feat_seed42.npy", p)
        auc, target = roc_auc_score(yte, p), float(table_auc[name])
        proba[key] = p
        sets.append({"feature_set": key, "subset": name, "source": "refit (src/run_guideline_ablation.py procedure)",
                     "auc": float(auc), "table_auc": target, "difference_full_precision": float(auc - target),
                     "auc_6dp": r6(auc), "table_auc_6dp": r6(target), "matched_6dp": r6(auc) == r6(target)})

    # ---- (ii) score the deposited 11-feature model ----
    Xs3, ys3 = s3.build_cohort(a.data_file)
    parts = s3.primary_split(Xs3, ys3)
    X_test13, y_test13 = parts[2], parts[8]
    identity["y_test_equals_deposited__11feat"] = bool(np.array_equal(y_test13, y_dep))
    if not identity["y_test_equals_deposited__11feat"]:
        raise SystemExit("IDENTITY CHECK FAILED for the 11-feature test set")
    rf11 = joblib.load(MODELS_REV / "rf_11feat_seed42.joblib")
    sc11 = joblib.load(MODELS_REV / "scaler_11feat_seed42.joblib")
    p11 = rf11.predict_proba(sc11.transform(X_test13[:, KEEP]))[:, 1]
    np.save(OUT_ABL / "proba_rf_11feat_seed42.npy", p11)
    auc11 = roc_auc_score(y_dep, p11)
    proba["11"] = p11
    sets.append({"feature_set": "11", "subset": "PP and MAP removed (11 variables)",
                 "source": "inference from deposited rf_11feat_seed42.joblib + scaler_11feat_seed42.joblib",
                 "auc": float(auc11), "table_auc": TARGET_11, "difference_full_precision": float(auc11 - TARGET_11),
                 "auc_6dp": r6(auc11), "table_auc_6dp": TARGET_11, "matched_6dp": r6(auc11) == TARGET_11})
    order = ["3", "6", "10", "11", "12", "13"]
    sets.sort(key=lambda r: order.index(r["feature_set"]))

    # ---- (iii)(a) feature-set AUC CIs, Section 3A(b) convention ----
    matched = [r["feature_set"] for r in sets if r["matched_6dp"]]
    n = len(y_dep)
    rng = np.random.default_rng(BOOT_SEED)
    boot = {k: [] for k in matched}
    discarded_a = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if y_dep[idx].min() == y_dep[idx].max():
            discarded_a += 1
            continue
        for k in matched:
            boot[k].append(roc_auc_score(y_dep[idx], proba[k][idx]))
    for r in sets:
        k = r["feature_set"]
        if k in boot:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            r.update(ci_low=float(lo), ci_high=float(hi), bootstrap_replicates_used=len(boot[k]), interval="computed")
        else:
            r.update(ci_low=None, ci_high=None, bootstrap_replicates_used=0,
                     interval="NOT COMPUTED: match requirement failed")
    pd.DataFrame(sets).to_csv(OUT_ABL / "feature_set_auc_ci.csv", index=False, lineterminator="\n")

    # ---- (iii)(b) male-minus-female AUC difference, Amendment 11 convention ----
    gender = X_test13[:, 0].astype(int)
    assert set(np.unique(gender)) == {0, 1}
    P = {m: np.load(PRED / f"proba_{f}.npy").astype(float) for m, f in LEADING.items()}
    gidx = {g: np.flatnonzero(gender == v) for g, v in GROUPS.items()}
    rng = np.random.default_rng(BOOT_SEED)
    rep = {g: [] for g in GROUPS}  # per replicate: dict model -> AUC, or None if single-class
    for g in GROUPS:  # female first, then male: Amendment 11's draw order
        ig = gidx[g]
        for _ in range(N_BOOT):
            s = rng.choice(ig, size=len(ig), replace=True)
            rep[g].append(None if y_dep[s].min() == y_dep[s].max()
                          else {m: roc_auc_score(y_dep[s], P[m][s]) for m in LEADING})
    pairs = [(f, mm) for f, mm in zip(rep["female"], rep["male"]) if f is not None and mm is not None]
    discarded_b = N_BOOT - len(pairs)

    a11 = pd.read_csv(OUT_LC / "gender_stratified_performance.csv")
    rows, reproduce = [], {}
    for m in LEADING:
        pt = {g: roc_auc_score(y_dep[gidx[g]], P[m][gidx[g]]) for g in GROUPS}
        for g in GROUPS:  # per-group CIs must reproduce Amendment 11
            vals = [r[m] for r in rep[g] if r is not None]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ref = a11.query("model == @m and gender == @g").iloc[0]
            reproduce[f"{m}|{g}"] = bool(lo == ref.auc_ci_low and hi == ref.auc_ci_high)
        d = np.array([mm[m] - f[m] for f, mm in pairs])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"model": m, "auc_male": float(pt["male"]), "auc_female": float(pt["female"]),
                     "difference_male_minus_female": float(pt["male"] - pt["female"]),
                     "ci_low": float(lo), "ci_high": float(hi), "ci_includes_zero": bool(lo <= 0 <= hi),
                     "bootstrap_replicates_used": int(len(d))})
    gdf = pd.DataFrame(rows)
    gdf.to_csv(OUT_LC / "gender_auc_difference_ci.csv", index=False, lineterminator="\n")

    # ---- expectations, mechanically from the prespecified refutation criteria ----
    by = {r["feature_set"]: r for r in sets}
    if "6" in boot and "13" in boot:
        s6, s13 = by["6"], by["13"]
        disjoint = s6["ci_high"] < s13["ci_low"] or s13["ci_high"] < s6["ci_low"]
        e1 = {"verdict": "REFUTED" if disjoint else "HELD", "six_ci": [s6["ci_low"], s6["ci_high"]],
              "thirteen_ci": [s13["ci_low"], s13["ci_high"]],
              "direction_contradicting_saturation": bool(s6["ci_high"] < s13["ci_low"])}
    else:
        e1 = {"verdict": "NOT TESTED: a required set failed its match requirement"}
    excl = [r["model"] for r in rows if not r["ci_includes_zero"]]
    e2 = {"verdict": "REFUTED" if excl else "HELD", "models_whose_interval_excludes_zero": excl}

    out = {"protocol_section": "Amendment 12 (prespecified)", "amendment_commit": AMENDMENT_COMMIT,
           "match_rule": "AUC rounded to six decimal places equals the recorded value rounded to six decimal places",
           "identity_checks": identity, "feature_sets": sets,
           "bootstrap_feature_sets": {"resamples": N_BOOT, "seed": BOOT_SEED,
                                      "method": "percentile; Section 3A(b) convention; one shared index vector per replicate",
                                      "discarded_single_class": discarded_a},
           "bootstrap_gender_difference": {"resamples": N_BOOT, "seed": BOOT_SEED,
                                           "method": "percentile; Amendment 11 convention; within-group, female drawn first; replicate b = male_b - female_b",
                                           "discarded_single_class_pairs": discarded_b,
                                           "per_group_ci_reproduces_amendment11": reproduce,
                                           "groups": {g: int(len(gidx[g])) for g in GROUPS}},
           "gender_difference": rows,
           "expectations": {"E1_six_vs_thirteen_overlap": e1, "E2_gender_difference_includes_zero_all_six": e2},
           "reference_not_used": {"proba_rf_npy_auc": float(roc_auc_score(y_dep, np.load(PRED / "proba_rf.npy")))},
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
           "device": "cpu (scikit-learn)"}
    with open(OUT_ABL / "amendment12_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    pd.set_option("display.width", 230)
    print(pd.DataFrame(sets)[["feature_set", "auc", "table_auc", "difference_full_precision", "matched_6dp",
                              "ci_low", "ci_high"]].to_string(index=False))
    print(gdf.round(5).to_string(index=False))
    print("identity:", identity); print("A11 reproduced:", all(reproduce.values()), "| discarded:", discarded_a, discarded_b)
    print("E1:", e1); print("E2:", e2)


if __name__ == "__main__":
    main()
