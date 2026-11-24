#!/usr/bin/env python3
"""Protocol Amendment 7 (prespecified): re-execution of two undeposited §4.5 analyses.

Table S8 of the submitted manuscript records "Not retained; reported descriptively" for the
Random Forest learning curve and the feature-profile overlap analysis. This script re-executes
both and deposits their outputs. Reuses Section 3's cohort/split/model functions
(src/revision/section3_retraining_fidelity.py) so the data path, Random Forest config and
XGBoost config are identical to the rest of the revision.

(a) Random Forest learning curve. Section 3 Random Forest configuration, the primary seed-42
    split, training-set fractions 10/25/50/75/100%, five stratified subsamples per fraction
    (random_state 1..5, sklearn train_test_split(train_size=fraction, stratify=y_train)),
    evaluated on the fixed primary test set. At 100% a stratified subsample of the full
    training set is, by construction, the full training set itself for every draw: a single
    fit is reported and SD is 0 by construction, not from five independent fits.
(b) Feature-profile overlap. A profile-disjoint split of the full cohort: "profile" is the
    exact tuple of the 10 recorded predictors (Section 3A(f)'s definition; the 13-feature and
    10-predictor groupings give an identical partition there). Profile groups are shuffled with
    random_state 42 and assigned WHOLE to the test partition, greedily, until the cumulative
    test fraction is as close as possible to 0.15 (the primary split's test fraction) without a
    group being split across train and test. The remaining pool is split 0.1765 stratified
    (random_state 42, the protocol's own train/val ratio) into train and validation, so
    XGBoost's early stopping has a validation set; the amendment does not specify a validation
    carve-out, so this reuses the established §1.2 ratio for consistency. A fresh StandardScaler
    is fit on this split's training partition only. Random Forest, MLP and XGBoost (Section 3
    configurations) are each fit and scored on the profile-disjoint test partition.

Run from the repository root:
    python src/revision/section4_5_robustness_reexecution.py --data-file <stage-1 CSV>
Writes: revision-results/lowcompute/rf_learning_curve.csv
        revision-results/lowcompute/profile_disjoint_overlap.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
SUBSAMPLE_SEEDS = [1, 2, 3, 4, 5]
ORIGINAL_LEARNING_CURVE = {0.10: 0.7942, 1.00: 0.7995}  # manuscript 4.5: "0.7942 at 10%... 0.7995 at the full training size"
ORIGINAL_OVERLAP = {"Random Forest": 0.8100, "MLP": 0.8052, "XGBoost": 0.8116}


def assert_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("GPU lost: torch.cuda.is_available() is False. Stopping per the protocol's device policy "
                          "(no silent CPU fallback). Record the device state and resume on GPU.")


def learning_curve(X_train_sc: np.ndarray, y_train: np.ndarray, X_test_sc: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(y_train)
    for frac in FRACTIONS:
        aucs = []
        if frac >= 1.0:
            model = s3.fit_rf(X_train_sc, y_train)
            aucs = [float(roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1]))]
            degenerate = True
        else:
            for seed in SUBSAMPLE_SEEDS:
                idx_sub, _ = train_test_split(np.arange(n), train_size=frac, random_state=seed, stratify=y_train)
                model = s3.fit_rf(X_train_sc[idx_sub], y_train[idx_sub])
                aucs.append(float(roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1])))
            degenerate = False
        aucs = np.array(aucs)
        rows.append({
            "training_fraction": frac, "n_train_rows": int(round(frac * n)) if frac < 1.0 else n,
            "n_subsamples": len(aucs), "degenerate_single_fit": degenerate,
            "mean_auc": float(aucs.mean()), "sd_auc_ddof1": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "min_auc": float(aucs.min()), "max_auc": float(aucs.max()),
            "subsample_random_states": "n/a (full training set)" if degenerate else f"{SUBSAMPLE_SEEDS}",
            "original_mean_auc_quoted_for_comparison_only": ORIGINAL_LEARNING_CURVE.get(frac, ""),
        })
    return pd.DataFrame(rows)


def profile_groups(X10: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(X10, columns=s3.FEATURES[:10])
    df["_y"] = y
    df["_row"] = np.arange(len(df))
    return df


def profile_disjoint_split(X: np.ndarray, y: np.ndarray, target_test_frac: float = 0.15, seed: int = 42):
    df = profile_groups(X[:, :10], y)
    groups = df.groupby(list(s3.FEATURES[:10]), sort=False, dropna=False)["_row"].apply(list)
    keys = list(groups.index)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    N = len(y)
    target_n = target_test_frac * N
    test_rows: list[int] = []
    cum = 0
    for k in order:
        rows = groups.iloc[k]
        prev_diff = abs(cum - target_n)
        new_cum = cum + len(rows)
        new_diff = abs(new_cum - target_n)
        if cum >= target_n:
            break
        if new_diff <= prev_diff or cum == 0:
            test_rows.extend(rows)
            cum = new_cum
        else:
            # adding this group would overshoot more than stopping now undershoots; stop here.
            break

    test_idx = np.array(sorted(test_rows))
    train_pool_idx = np.array(sorted(set(range(N)) - set(test_rows)))

    # profile-disjointness check
    test_profiles = set(map(tuple, X[test_idx, :10].tolist()))
    train_profiles = set(map(tuple, X[train_pool_idx, :10].tolist()))
    overlap = test_profiles & train_profiles
    if overlap:
        raise SystemExit(f"PROFILE-DISJOINTNESS CHECK FAILED: {len(overlap)} profiles appear in both partitions.")
    n_profiles_in_test = len(test_profiles)

    X_tmp, y_tmp = X[train_pool_idx], y[train_pool_idx]
    i_train, i_val = train_test_split(np.arange(len(y_tmp)), test_size=0.1765, random_state=seed, stratify=y_tmp)
    X_train, y_train = X_tmp[i_train], y_tmp[i_train]
    X_val, y_val = X_tmp[i_val], y_tmp[i_val]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test,
        "X_train_sc": X_train_sc, "X_val_sc": X_val_sc, "X_test_sc": X_test_sc,
        "n_distinct_profiles": len(keys), "n_profiles_in_test": n_profiles_in_test,
        "target_test_fraction": target_test_frac, "achieved_test_fraction": len(test_idx) / N,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    parser.add_argument("--xgb-device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = s3.build_cohort(args.data_file)

    # --- (a) Random Forest learning curve, primary seed-42 split ---
    (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc,
     y_train, y_val, y_test, scaler) = s3.primary_split(X, y)
    deposited_y_test = np.load(ROOT / "artifacts" / "predictions" / "y_test.npy").astype(int)
    if not np.array_equal(y_test, deposited_y_test):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")

    lc = learning_curve(X_train_sc, y_train, X_test_sc, y_test)
    lc.to_csv(args.output_dir / "rf_learning_curve.csv", index=False, lineterminator="\n")

    # --- (b) Feature-profile overlap, profile-disjoint split ---
    assert_gpu()
    split = profile_disjoint_split(X, y, target_test_frac=0.15, seed=42)
    rf = s3.fit_rf(split["X_train_sc"], split["y_train"])
    mlp = s3.fit_mlp(split["X_train_sc"], split["y_train"])
    xg = s3.fit_xgb(split["X_train"], split["y_train"], split["X_val"], split["y_val"], device=args.xgb_device)

    def pos_rate(yy): return float(yy.mean())

    overlap_rows = []
    for name, auc in (
        ("Random Forest", roc_auc_score(split["y_test"], rf.predict_proba(split["X_test_sc"])[:, 1])),
        ("MLP", roc_auc_score(split["y_test"], mlp.predict_proba(split["X_test_sc"])[:, 1])),
        ("XGBoost", roc_auc_score(split["y_test"], xg.predict_proba(split["X_test"])[:, 1])),
    ):
        overlap_rows.append({
            "model": name, "test_auc": float(auc),
            "original_test_auc_quoted_for_comparison_only": ORIGINAL_OVERLAP[name],
            "delta_vs_original": float(auc) - ORIGINAL_OVERLAP[name],
            "n_train": len(split["y_train"]), "n_val": len(split["y_val"]), "n_test": len(split["y_test"]),
            "positive_rate_train": pos_rate(split["y_train"]), "positive_rate_val": pos_rate(split["y_val"]),
            "positive_rate_test": pos_rate(split["y_test"]),
            "n_distinct_profiles_cohort": split["n_distinct_profiles"],
            "target_test_fraction": split["target_test_fraction"], "achieved_test_fraction": split["achieved_test_fraction"],
            "xgboost_best_iteration": int(xg.best_iteration) if name == "XGBoost" else "",
            "device": args.xgb_device if name == "XGBoost" else "cpu",
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(args.output_dir / "profile_disjoint_overlap.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 200)
    print("(a) Random Forest learning curve\n" + lc.round(5).to_string(index=False))
    print("\n(b) Feature-profile overlap (profile-disjoint split)\n" + overlap_df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
