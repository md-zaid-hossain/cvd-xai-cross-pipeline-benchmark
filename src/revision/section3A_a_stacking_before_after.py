#!/usr/bin/env python3
"""Protocol Section 3A(a): stacking correction impact (before vs after).

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(a), using the Section 3A
conventions and the Section 3A(b) paired bootstrap (2,000 resamples,
numpy default_rng(42), one shared index vector per replicate).

Compares the original notebook stacking (proba/pred_stack_original.npy,
mismatched configuration) with the corrected stacking
(proba/pred_stack_corrected.npy). Uses only deposited arrays.

Run from the repository root:
    python src/revision/section3A_a_stacking_before_after.py

Writes the Section 7 output for 3A(a):
    revision-results/lowcompute/stacking_before_after.csv
"""
from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True  # keep src/ free of __pycache__ (it would enter the manifest)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import MODELS, delong_test  # noqa: E402  (used unchanged, per Section 3A)

N_BOOT = 2000
BOOT_SEED = 42
VERSIONS = {"original": "stack_original", "corrected": "stack_corrected"}


def calibration(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Calibration intercept and slope exactly as in src/run_validated_stats.py."""
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def paired_bootstrap(y: np.ndarray, probas: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """Section 3A(b): sequential draws from default_rng(42); one idx per replicate, shared."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    aucs = {name: [] for name in probas}
    discarded = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():  # single-class replicate: discard and count
            discarded += 1
            continue
        for name, p in probas.items():
            aucs[name].append(roc_auc_score(yb, p[idx]))
    return {k: np.asarray(v) for k, v in aucs.items()}, discarded


def rank_with_stand_in(y: np.ndarray, pred_dir: Path, stand_in: str) -> tuple[pd.Series, list[str]]:
    """Primary AUC of all 15 models with `stand_in` as the Stacking Ensemble; rank 1 = highest."""
    aucs = {}
    for name, suffix in MODELS.items():
        if name == "Stacking Ensemble":
            suffix = stand_in
        aucs[name] = roc_auc_score(y, np.load(pred_dir / f"proba_{suffix}.npy").astype(float).reshape(-1))
    s = pd.Series(aucs)
    ranks = s.rank(ascending=False, method="average")
    top_six = list(s.sort_values(ascending=False).index[:6])
    return ranks, top_six


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = args.predictions_dir

    y = np.load(d / "y_test.npy").astype(int)
    proba = {v: np.load(d / f"proba_{s}.npy").astype(float).reshape(-1) for v, s in VERSIONS.items()}
    pred = {v: np.load(d / f"pred_{s}.npy").astype(int).reshape(-1) for v, s in VERSIONS.items()}

    # Identity check required by Section 3A(a).
    identity_proba = (d / "proba_stack_primary_original.npy").read_bytes() == (d / "proba_stack_original.npy").read_bytes()
    identity_pred = (d / "pred_stack_primary_original.npy").read_bytes() == (d / "pred_stack_original.npy").read_bytes()

    # Deposited pred arrays should be the 0.5-threshold classes; record whether they are.
    pred_is_threshold = {v: bool(np.array_equal(pred[v], (proba[v] >= 0.5).astype(int))) for v in VERSIONS}

    boot, discarded = paired_bootstrap(y, proba)
    delta_boot = boot["corrected"] - boot["original"]

    ranks, top_six = {}, {}
    for v, s in VERSIONS.items():
        r, t = rank_with_stand_in(y, d, s)
        ranks[v], top_six[v] = r["Stacking Ensemble"], t

    rows = []
    for v in VERSIONS:
        p, c = proba[v], pred[v]
        intercept, slope = calibration(y, p)
        ci = np.percentile(boot[v], [2.5, 97.5])
        rows.append({
            "row_type": "version", "version": v, "array": f"proba_{VERSIONS[v]}.npy",
            "auc": roc_auc_score(y, p), "auc_ci_low": ci[0], "auc_ci_high": ci[1],
            "accuracy": accuracy_score(y, c), "f1": f1_score(y, c),
            "precision": precision_score(y, c), "recall": recall_score(y, c),
            "brier": brier_score_loss(y, p),
            "calibration_intercept": intercept, "calibration_slope": slope,
            "rank_among_15": ranks[v], "top_six": "; ".join(top_six[v]),
            "pred_equals_proba_ge_0.5": pred_is_threshold[v],
        })

    delta_auc = rows[1]["auc"] - rows[0]["auc"]
    delta_ci = np.percentile(delta_boot, [2.5, 97.5])
    auc_c, auc_o, z, p_value, var = delong_test(y, proba["corrected"], proba["original"])
    absdiff = np.abs(proba["corrected"] - proba["original"])
    flips = int(np.sum((proba["corrected"] >= 0.5) != (proba["original"] >= 0.5)))
    flips_pred = int(np.sum(pred["corrected"] != pred["original"]))
    rows.append({
        "row_type": "comparison", "version": "corrected_minus_original",
        "delta_auc": delta_auc, "delta_auc_ci_low": delta_ci[0], "delta_auc_ci_high": delta_ci[1],
        "delong_auc_corrected": auc_c, "delong_auc_original": auc_o,
        "delong_z": z, "delong_p": p_value, "delong_variance_difference": var,
        "mean_abs_diff_p": float(absdiff.mean()), "max_abs_diff_p": float(absdiff.max()),
        "pearson_correlation": float(pearsonr(proba["corrected"], proba["original"])[0]),
        "n_class_flips_0.5": flips, "n_class_flips_pred_arrays": flips_pred,
        "top_six_membership_changes": set(top_six["corrected"]) != set(top_six["original"]),
        "bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED,
        "bootstrap_discarded_single_class": discarded,
        "identity_proba_stack_primary_original": identity_proba,
        "identity_pred_stack_primary_original": identity_pred,
    })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "stacking_before_after.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
