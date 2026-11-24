#!/usr/bin/env python3
"""Protocol Amendment 11 (prespecified, pushed alone in 3e2ef42): gender-stratified performance.

For each of the six leading models on the primary seed-42 test set, per recorded gender category:
accuracy, precision, recall, false positive rate, positive prediction rate (threshold 0.5) and AUC with a
95% percentile bootstrap CI (Section 3A(b) convention: 2,000 resamples, seed 42; resampled within each
gender group, one shared index vector per replicate across the six models), plus group size and outcome
prevalence; and the between-group AUC difference (male minus female) as a point estimate.

Inputs: the deposited probability arrays artifacts/predictions/proba_{stack_corrected,xgb,cat,ft,saint,rf}.npy
and y_test.npy; no model is refitted. Gender comes from the Gender column of the primary seed-42 test set
(0 = female, 1 = male), rebuilt with the Section 3 cohort/split functions and checked against y_test.npy.
Descriptive only; no claim of fairness or its absence is made.

Run from the repository root:
    python src/revision/amendment11_gender_stratified.py --data-file <stage-1 CSV>
Writes: revision-results/lowcompute/gender_stratified_performance.csv
        revision-results/lowcompute/gender_stratified_performance.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "artifacts" / "predictions"
OUT = ROOT / "revision-results" / "lowcompute"
_spec = importlib.util.spec_from_file_location("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)

MODELS = {"Stacking Ensemble": "stack_corrected", "XGBoost": "xgb", "CatBoost": "cat",
          "FT-Transformer": "ft", "SAINT-inspired": "saint", "Random Forest": "rf"}
GROUPS = {"female": 0, "male": 1}
N_BOOT, BOOT_SEED, THRESH = 2000, 42, 0.5


def point_metrics(y, p):
    pred = (p >= THRESH).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
            "positive_prediction_rate": pred.mean(), "auc": roc_auc_score(y, p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    a = ap.parse_args()

    X, y = s3.build_cohort(a.data_file)
    parts = s3.primary_split(X, y)
    X_test, y_test = parts[2], parts[8]
    y_dep = np.load(PRED / "y_test.npy").astype(int)
    if not np.array_equal(y_test, y_dep):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    gender = X_test[:, 0].astype(int)
    assert set(np.unique(gender)) == {0, 1}
    P = {m: np.load(PRED / f"proba_{f}.npy").astype(float) for m, f in MODELS.items()}

    checks = {}
    for m, f in MODELS.items():  # 0.5-threshold predictions agree with the deposited prediction arrays
        dep = PRED / f"pred_{f}.npy"
        if dep.exists():
            checks[f"pred_matches_deposited_{f}"] = bool(np.array_equal((P[m] >= THRESH).astype(int), np.load(dep).astype(int)))
        checks[f"overall_auc_{f}"] = float(roc_auc_score(y_dep, P[m]))

    rng = np.random.default_rng(BOOT_SEED)
    boot = {(m, g): [] for m in MODELS for g in GROUPS}
    discarded = {g: 0 for g in GROUPS}
    idx = {g: np.flatnonzero(gender == v) for g, v in GROUPS.items()}
    for g in GROUPS:  # within-group resampling, one shared index vector per replicate across models
        ig = idx[g]
        for _ in range(N_BOOT):
            s = rng.choice(ig, size=len(ig), replace=True)
            if y_dep[s].min() == y_dep[s].max():
                discarded[g] += 1
                continue
            for m in MODELS:
                boot[(m, g)].append(roc_auc_score(y_dep[s], P[m][s]))

    rows = []
    for m in MODELS:
        for g, v in GROUPS.items():
            ig = idx[g]
            pm = point_metrics(y_dep[ig], P[m][ig])
            lo, hi = np.percentile(boot[(m, g)], [2.5, 97.5])
            rows.append({"model": m, "gender": g, "gender_code": v, "n": int(len(ig)),
                         "outcome_prevalence": float(y_dep[ig].mean()), **{k: float(x) for k, x in pm.items()},
                         "auc_ci_low": float(lo), "auc_ci_high": float(hi), "bootstrap_replicates_used": len(boot[(m, g)])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gender_stratified_performance.csv", index=False, lineterminator="\n")

    diff = {m: float(df.query("model == @m and gender == 'male'")["auc"].iloc[0] -
                     df.query("model == @m and gender == 'female'")["auc"].iloc[0]) for m in MODELS}
    out = {"protocol_section": "Amendment 11 (prespecified)", "amendment_commit": "3e2ef42e0c6fc055a80a29f1c4ba355eaacc234e",
           "threshold": THRESH, "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "method": "percentile, within gender group",
                                              "discarded_single_class": discarded},
           "groups": {g: {"code": v, "n": int(len(idx[g])), "outcome_prevalence": float(y_dep[idx[g]].mean())} for g, v in GROUPS.items()},
           "auc_difference_male_minus_female": diff, "checks": checks,
           "note": "Descriptive only; no claim of fairness or its absence is made."}
    with open(OUT / "gender_stratified_performance.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    pd.set_option("display.width", 230)
    print(df.drop(columns=["gender_code"]).round(4).to_string(index=False))
    print("\nAUC difference (male - female):", {k: round(v, 4) for k, v in diff.items()})
    print("groups:", out["groups"]); print("checks:", checks); print("discarded:", discarded)


if __name__ == "__main__":
    main()
