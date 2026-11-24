#!/usr/bin/env python3
"""Protocol Section 3A(e): prevalence transport.

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(e). Prior-probability
(logit-intercept) adjustment under the label-shift assumption -- NOT
resampling and NOT case deletion; all 10,281 test records are retained.

    p     = clip(proba, 1e-6, 1 - 1e-6)
    p_adj = expit(logit(p) + logit(pi_t) - logit(pi_s))

Evaluation reweights the test set analytically to pi_t:
w = pi_t / pi_s for y = 1, (1 - pi_t) / (1 - pi_s) for y = 0.

Per model (all 15) and target pi_t in {0.05, 0.10, 0.20, 0.4944}:
AUC (checked against the primary AUC, |diff| < 1e-12, both unweighted on
p_adj and with the class weights); weighted Brier; weighted
calibration-in-the-large; weighted calibration intercept and slope
(weighted logistic regression of y on logit(p_adj), C=np.inf); sens,
spec, PPV, NPV with p_adj at thresholds 0.5 and pi_t (PPV/NPV weighted);
transported net benefit on the 3A(d) grid:
    NB = sens * pi_t - (1 - spec) * (1 - pi_t) * p_t / (1 - p_t)
    treat-all = pi_t - (1 - pi_t) * p_t / (1 - p_t)

Run from the repository root:
    python src/revision/section3A_e_prevalence_transport.py
Writes:
    revision-results/lowcompute/prevalence_transport_metrics.csv
    revision-results/lowcompute/prevalence_transport_net_benefit.csv
"""
from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import MODELS  # noqa: E402

TARGETS = [0.05, 0.10, 0.20, 0.4944]
PT_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)  # identical to 3A(d)
AUC_TOL = 1e-12


def classify(y, p_adj, w, thr):
    pos = p_adj >= thr
    tp, fn = np.sum(pos & (y == 1)), np.sum(~pos & (y == 1))
    tn, fp = np.sum(~pos & (y == 0)), np.sum(pos & (y == 0))
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    wtp, wfp = np.sum(w[pos & (y == 1)]), np.sum(w[pos & (y == 0)])
    wtn, wfn = np.sum(w[~pos & (y == 0)]), np.sum(w[~pos & (y == 1)])
    ppv = wtp / (wtp + wfp) if (wtp + wfp) > 0 else np.nan
    npv = wtn / (wtn + wfn) if (wtn + wfn) > 0 else np.nan
    return sens, spec, ppv, npv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(args.predictions_dir / "y_test.npy").astype(int)
    pi_s = float(y.mean())
    metrics, nb_rows = [], []

    for model, suffix in MODELS.items():
        proba = np.load(args.predictions_dir / f"proba_{suffix}.npy").astype(float).reshape(-1)
        auc_primary = roc_auc_score(y, proba)
        p = np.clip(proba, 1e-6, 1 - 1e-6)
        for pi_t in TARGETS:
            p_adj = expit(logit(p) + logit(pi_t) - logit(pi_s))
            w = np.where(y == 1, pi_t / pi_s, (1 - pi_t) / (1 - pi_s))
            auc_unweighted = roc_auc_score(y, p_adj)
            auc_weighted = roc_auc_score(y, p_adj, sample_weight=w)
            lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000).fit(
                logit(p_adj).reshape(-1, 1), y, sample_weight=w)
            row = {
                "model": model, "target_prevalence": pi_t, "source_prevalence": pi_s,
                "auc_primary": auc_primary, "auc_transported": auc_unweighted,
                "auc_transported_weighted": auc_weighted,
                "auc_abs_diff_max": max(abs(auc_unweighted - auc_primary), abs(auc_weighted - auc_primary)),
                "auc_unchanged_lt_1e-12": max(abs(auc_unweighted - auc_primary),
                                              abs(auc_weighted - auc_primary)) < AUC_TOL,
                "weighted_brier": brier_score_loss(y, p_adj, sample_weight=w),
                "weighted_mean_predicted": float(np.average(p_adj, weights=w)),
                "weighted_observed": float(np.average(y, weights=w)),
                "calibration_in_the_large": float(np.average(p_adj, weights=w) - pi_t),
                "weighted_calibration_intercept": float(lr.intercept_[0]),
                "weighted_calibration_slope": float(lr.coef_[0, 0]),
            }
            for label, thr in (("0.5", 0.5), ("pi_t", pi_t)):
                sens, spec, ppv, npv = classify(y, p_adj, w, thr)
                row.update({f"sens_at_{label}": sens, f"spec_at_{label}": spec,
                            f"ppv_at_{label}": ppv, f"npv_at_{label}": npv})
            metrics.append(row)

            for pt in PT_GRID:
                sens, spec, _, _ = classify(y, p_adj, w, pt)
                nb_rows.append({"model": model, "target_prevalence": pi_t, "threshold": pt,
                                "net_benefit": sens * pi_t - (1 - spec) * (1 - pi_t) * pt / (1 - pt)})

    for pi_t in TARGETS:
        for pt in PT_GRID:
            nb_rows.append({"model": "treat-all", "target_prevalence": pi_t, "threshold": pt,
                            "net_benefit": pi_t - (1 - pi_t) * pt / (1 - pt)})

    m = pd.DataFrame(metrics)
    m.to_csv(args.output_dir / "prevalence_transport_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(nb_rows).to_csv(args.output_dir / "prevalence_transport_net_benefit.csv",
                                 index=False, lineterminator="\n")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"pi_s = {pi_s:.6f}")
    print(f"AUC unchanged (<1e-12) in all {len(m)} model x target cells: {bool(m['auc_unchanged_lt_1e-12'].all())}; "
          f"max |diff| = {m['auc_abs_diff_max'].max():.3e}")
    cols = ["model", "target_prevalence", "weighted_brier", "calibration_in_the_large",
            "weighted_calibration_intercept", "weighted_calibration_slope",
            "sens_at_0.5", "spec_at_0.5", "ppv_at_0.5", "npv_at_0.5", "ppv_at_pi_t", "npv_at_pi_t"]
    top = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest", "MLP"]
    print(m[m["model"].isin(top)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
