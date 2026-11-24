#!/usr/bin/env python3
"""Protocol Section 3A(f): exact-profile collision bound.

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(f). Cohort and seed-42
split are built by the Section 3 script's functions (Sections 1.1-1.2),
so the data path is identical to the fidelity gate.

Profile = exact tuple of the 13 feature values in Section 1.1 column
order, no rounding or binning. For each profile g: n1_g, n0_g.
  B           = sum_g min(n1_g, n0_g) / N     (collision bound)
  1 - B       = maximum achievable accuracy on this sample
  AUC ceiling = 1 - 0.5 * sum_g n1_g * n0_g / (N1 * N0)
This is NOT the Bayes error (in-sample; exact duplicates only; depends on
how coarsely the features were recorded).

Also checks (i) that grouping on the 10 recorded predictors gives the
same partition as the 13-feature grouping, as the protocol states, and
(ii) whether the manuscript's counts (3,333 shared profiles; 1,574 with
both outcome labels) are reproduced.

Run from the repository root:
    python src/revision/section3A_f_profile_collision.py --data-file <stage-1 CSV>
Writes: revision-results/lowcompute/profile_collision_bound.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

MANUSCRIPT_SHARED = 3333
MANUSCRIPT_CONFLICTING = 1574


def collision(X: np.ndarray, y: np.ndarray, columns: list[str]) -> dict:
    df = pd.DataFrame(X, columns=columns)
    df["_y"] = y
    g = df.groupby(columns, sort=False, dropna=False)["_y"].agg(n="size", n1="sum")
    g["n0"] = g["n"] - g["n1"]
    N, N1 = len(y), int(y.sum())
    N0 = N - N1
    shared = g["n"] >= 2
    conflicting = (g["n1"] > 0) & (g["n0"] > 0)
    B = float(np.minimum(g["n1"], g["n0"]).sum() / N)
    return {
        "n_records": N, "n_positive": N1, "n_negative": N0,
        "distinct_profiles": int(len(g)),
        "shared_profiles_ge2": int(shared.sum()),
        "records_in_shared_profiles": int(g.loc[shared, "n"].sum()),
        "conflicting_profiles_both_labels": int(conflicting.sum()),
        "records_in_conflicting_profiles": int(g.loc[conflicting, "n"].sum()),
        "collision_bound_B": B,
        "max_achievable_accuracy_1_minus_B": 1 - B,
        "auc_ceiling": float(1 - 0.5 * (g["n1"] * g["n0"]).sum() / (N1 * N0)),
        "_groups": g,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = s3.build_cohort(args.data_file)
    parts = s3.primary_split(X, y)
    X_test, y_test = parts[2], parts[8]
    if not np.array_equal(y_test, np.load(args.predictions_dir / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")

    cols13 = s3.FEATURES
    rows = []
    for label, XX, yy in (("cohort (primary)", X, y), ("test set seed 42 (secondary)", X_test, y_test)):
        r13 = collision(XX, yy, cols13)
        r10 = collision(XX[:, :10], yy, cols13[:10])  # the 10 recorded predictors
        same_partition = (r13["distinct_profiles"] == r10["distinct_profiles"]
                          and r13["collision_bound_B"] == r10["collision_bound_B"]
                          and r13["auc_ceiling"] == r10["auc_ceiling"])
        row = {"dataset": label, **{k: v for k, v in r13.items() if not k.startswith("_")},
               "grouping_10_predictors_identical": same_partition}
        if label.startswith("cohort"):
            row["manuscript_shared_profiles"] = MANUSCRIPT_SHARED
            row["manuscript_conflicting_profiles"] = MANUSCRIPT_CONFLICTING
            row["manuscript_shared_reproduced"] = r13["shared_profiles_ge2"] == MANUSCRIPT_SHARED
            row["manuscript_conflicting_reproduced"] = r13["conflicting_profiles_both_labels"] == MANUSCRIPT_CONFLICTING
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "profile_collision_bound.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
