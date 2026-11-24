#!/usr/bin/env python3
"""Protocol Section 3A(g): grouped haemodynamic permutation importance.

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(g).
Model: the retrained Random Forest from Section 3
(artifacts/models-revision/rf_seed42.joblib; variant C, selected by the gate).
Data: full seed-42 test set X_test_sc (10,281 records), rebuilt with the
Section 3 script's functions (Sections 1.1-1.2) and checked against
artifacts/predictions/y_test.npy.

Block {SBP, DBP, PP, MAP} = column indices 3, 4, 10, 11, permuted as one
unit (same row permutation for all four), 10 repeats, rng = default_rng(42):
    for r in range(10):
        perm = rng.permutation(n)
        Xp[:, block] = X_test_sc[perm][:, block]
        drop[r] = auc_baseline - AUC(rf on Xp)

Comparison: single-feature permutation importance for all 13 features,
same model and data, 10 repeats, a FRESH default_rng(42). Loop order
(the protocol fixes the column order but not the loop nesting): features
in column order 0-12 as the outer loop, the 10 repeats as the inner loop,
one rng.permutation(n) per feature per repeat -- the same nesting as
sklearn.inspection.permutation_importance.

Mean and SD (ddof = 1) over the 10 repeats. Ranks use
scipy.stats.rankdata(-mean, method="average").

Run from the repository root:
    python src/revision/section3A_g_grouped_haemodynamic_permutation.py --data-file <stage-1 CSV>
Writes: revision-results/lowcompute/grouped_haemodynamic_permutation.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
              "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
BLOCK = [3, 4, 10, 11]
N_REPEATS = 10
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=ROOT / "artifacts" / "models-revision" / "rf_seed42.joblib")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = s3.build_cohort(args.data_file)
    parts = s3.primary_split(X, y)
    X_test_sc, y_test = parts[5], parts[8]
    if not np.array_equal(y_test, np.load(args.predictions_dir / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    rf = joblib.load(args.model_file)
    n = len(y_test)

    def auc_of(Xm: np.ndarray) -> float:
        return roc_auc_score(y_test, rf.predict_proba(Xm)[:, 1])

    auc_baseline = auc_of(X_test_sc)

    # Grouped block (protocol code block).
    rng = np.random.default_rng(SEED)
    block_drop = np.empty(N_REPEATS)
    for r in range(N_REPEATS):
        perm = rng.permutation(n)
        Xp = X_test_sc.copy()
        Xp[:, BLOCK] = X_test_sc[perm][:, BLOCK]
        block_drop[r] = auc_baseline - auc_of(Xp)

    # Single features: fresh rng, features 0-12 outer, repeats inner.
    rng = np.random.default_rng(SEED)
    single_drop = np.empty((13, N_REPEATS))
    for j in range(13):
        for r in range(N_REPEATS):
            perm = rng.permutation(n)
            Xp = X_test_sc.copy()
            Xp[:, j] = X_test_sc[perm, j]
            single_drop[j, r] = auc_baseline - auc_of(Xp)

    single_mean = single_drop.mean(axis=1)
    single_sd = single_drop.std(axis=1, ddof=1)
    sum_four = float(single_mean[BLOCK].sum())
    block_mean = float(block_drop.mean())

    non_haemo = [j for j in range(13) if j not in BLOCK]
    rank_pool_vals = np.concatenate([[block_mean], single_mean[non_haemo]])
    rank_pool = rankdata(-rank_pool_vals, method="average")
    rank_among_13 = rankdata(-single_mean, method="average")

    rows = [{
        "item": "Haemodynamic block (SBP+DBP+PP+MAP)", "type": "block",
        "columns_permuted": "3,4,10,11",
        "mean_auc_drop": block_mean, "sd_auc_drop": float(block_drop.std(ddof=1)),
        "rank_among_block_plus_9": float(rank_pool[0]), "rank_among_13_singles": np.nan,
        **{f"drop_r{r + 1}": block_drop[r] for r in range(N_REPEATS)},
    }]
    for j in range(13):
        rows.append({
            "item": FEAT_NAMES[j], "type": "single (haemodynamic)" if j in BLOCK else "single",
            "columns_permuted": str(j),
            "mean_auc_drop": float(single_mean[j]), "sd_auc_drop": float(single_sd[j]),
            "rank_among_block_plus_9": float(rank_pool[1 + non_haemo.index(j)]) if j in non_haemo else np.nan,
            "rank_among_13_singles": float(rank_among_13[j]),
            **{f"drop_r{r + 1}": single_drop[j, r] for r in range(N_REPEATS)},
        })
    out = pd.DataFrame(rows)
    out["baseline_auc"] = auc_baseline
    out["sum_of_four_haemodynamic_singles"] = sum_four
    out["block_over_sum_of_four_singles"] = block_mean / sum_four if sum_four != 0 else np.nan
    out["n_repeats"] = N_REPEATS
    out["rng"] = f"default_rng({SEED}); singles: fresh rng, features outer (0-12), repeats inner"
    out.to_csv(args.output_dir / "grouped_haemodynamic_permutation.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 200)
    print(f"baseline AUC = {auc_baseline:.10f}")
    print(out[["item", "type", "mean_auc_drop", "sd_auc_drop", "rank_among_block_plus_9",
               "rank_among_13_singles"]].to_string(index=False))
    print(f"\nblock mean drop = {block_mean:.6f}; sum of four singles = {sum_four:.6f}; "
          f"block / sum = {block_mean / sum_four:.3f}")


if __name__ == "__main__":
    main()
