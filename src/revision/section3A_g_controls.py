#!/usr/bin/env python3
"""Protocol Section 3A(g), Amendment 2: independent and partly-redundant controls.

Implements Amendment 2 in docs/ANALYSIS_PROTOCOL.md Section 8 (committed in
efb13a4 before this script was run). Two non-analytical controls give
reference points for the Section 3A(g) block-to-sum ratio:

  Control A (independent reference):     Age, Cholesterol, BMI, Active
  Control B (partly redundant reference): Age, Cholesterol, BMI, RiskCount
  Haemodynamic block (Section 3A(g)):     SBP, DBP, PP, MAP

Same Section 3 Random Forest (rf_seed42.joblib), same seed-42 X_test_sc,
10 repeats. Each block is permuted as one unit exactly as the Section 3A(g)
code block does, each with its OWN fresh default_rng(42) (so every block
receives the same ten row permutations). Single-feature importances use the
Section 3A(g) nesting (fresh default_rng(42); features 0-12 outer, repeats
inner) and are checked against
revision-results/lowcompute/grouped_haemodynamic_permutation.csv.

Redundancy is described by pairwise Spearman rho among the block's four
features on the seed-42 test set.

Run from the repository root:
    python src/revision/section3A_g_controls.py --data-file <stage-1 CSV>
Writes: revision-results/lowcompute/grouped_permutation_controls.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
              "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
N_REPEATS = 10
SEED = 42
BLOCKS = [  # ordered by redundancy level, lowest first
    ("Control A (independent reference)", ["Age", "Cholesterol", "BMI", "Active"],
     "independent: original recorded variables, none derived from another"),
    ("Control B (partly redundant reference)", ["Age", "Cholesterol", "BMI", "RiskCount"],
     "partly redundant: RiskCount contains a (cholesterol>1) term"),
    ("Haemodynamic block (Section 3A(g))", ["SBP", "DBP", "PP", "MAP"],
     "fully redundant: PP and MAP are deterministic functions of SBP and DBP"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=ROOT / "artifacts" / "models-revision" / "rf_seed42.joblib")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()

    X, y = s3.build_cohort(args.data_file)
    parts = s3.primary_split(X, y)
    X_test, X_test_sc, y_test = parts[2], parts[5], parts[8]
    if not np.array_equal(y_test, np.load(args.predictions_dir / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    rf = joblib.load(args.model_file)
    n = len(y_test)

    def auc_of(Xm: np.ndarray) -> float:
        return roc_auc_score(y_test, rf.predict_proba(Xm)[:, 1])

    auc_baseline = auc_of(X_test_sc)

    # Single features: Section 3A(g) nesting.
    rng = np.random.default_rng(SEED)
    single = np.empty((13, N_REPEATS))
    for j in range(13):
        for r in range(N_REPEATS):
            perm = rng.permutation(n)
            Xp = X_test_sc.copy()
            Xp[:, j] = X_test_sc[perm, j]
            single[j, r] = auc_baseline - auc_of(Xp)
    single_mean = single.mean(axis=1)

    g = pd.read_csv(args.output_dir / "grouped_haemodynamic_permutation.csv").set_index("item")
    ref = np.array([g.loc[name, "mean_auc_drop"] for name in FEAT_NAMES])
    max_single_diff = float(np.max(np.abs(ref - single_mean)))

    rows = []
    for label, feats, level in BLOCKS:
        idx = [FEAT_NAMES.index(f) for f in feats]
        rng = np.random.default_rng(SEED)  # fresh per block: identical permutations across blocks
        drop = np.empty(N_REPEATS)
        for r in range(N_REPEATS):
            perm = rng.permutation(n)
            Xp = X_test_sc.copy()
            Xp[:, idx] = X_test_sc[perm][:, idx]
            drop[r] = auc_baseline - auc_of(Xp)
        rhos = [abs(spearmanr(X_test[:, a], X_test[:, b])[0]) for a, b in itertools.combinations(idx, 2)]
        s = float(single_mean[idx].sum())
        rows.append({
            "block": label, "features": "+".join(feats), "redundancy_level": level,
            "max_pairwise_abs_spearman": max(rhos), "mean_pairwise_abs_spearman": float(np.mean(rhos)),
            "block_mean_auc_drop": float(drop.mean()), "block_sd_auc_drop": float(drop.std(ddof=1)),
            "sum_of_four_singles": s, "block_to_sum_ratio": float(drop.mean()) / s,
            "singles_share_of_block": s / float(drop.mean()),
            **{f"single_{f}": float(single_mean[FEAT_NAMES.index(f)]) for f in feats},
            **{f"drop_r{r + 1}": drop[r] for r in range(N_REPEATS)},
        })
    out = pd.DataFrame(rows)
    out["baseline_auc"] = auc_baseline
    out["n_repeats"] = N_REPEATS
    out["max_abs_diff_singles_vs_3Ag_output"] = max_single_diff
    out["rng"] = f"block: fresh default_rng({SEED}) per block; singles: fresh default_rng({SEED}), features outer 0-12, repeats inner"
    out.to_csv(args.output_dir / "grouped_permutation_controls.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 220)
    print(f"baseline AUC = {auc_baseline:.10f}; singles vs 3A(g) output: max |diff| = {max_single_diff:.3e}")
    print(out[["block", "max_pairwise_abs_spearman", "mean_pairwise_abs_spearman", "block_mean_auc_drop",
               "block_sd_auc_drop", "sum_of_four_singles", "block_to_sum_ratio",
               "singles_share_of_block"]].to_string(index=False))


if __name__ == "__main__":
    main()
