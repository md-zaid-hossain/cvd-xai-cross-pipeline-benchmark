#!/usr/bin/env python3
"""Protocol Amendment 8 (prespecified): bounded attempt to reproduce the withdrawn between-split SDs.

No code producing the submitted manuscript's between-split SDs (0.0038-0.0042, mean AUC
0.7958-0.7988 for CatBoost/XGBoost/LightGBM/Random Forest/MLP) exists anywhere in the deposited
or local files, and the original seeds are unknown, so exact reproduction is impossible in
principle. This script tests only whether the reported MAGNITUDE arises from a plausible
configuration, under exactly two prespecified configurations, both on the five models above,
both at seeds 42/101/202/303/404, both using the Section 5 split function (identical to
src/revision/section5_five_seed.py's `split`, imported from it, so the splits are bit-identical
to the deposited Section 5 splits).

Configuration A -- per-split refitting (matches the Section 5 procedure). Hyperparameters as
    published; early stopping / tree counts re-derived at each split from that split's own
    validation set. XGBoost, LightGBM (standalone), CatBoost and Random Forest reuse the
    already-deposited revision-results/five_seed/proba_*_seed{s}.npy arrays rather than
    refitting (Section 5 used this exact procedure, these exact seeds and this exact split
    function, so refitting would reproduce the same arrays at the cost of GPU time). Only MLP is
    newly fit for Configuration A: it is not part of Section 5's model set.
Configuration B -- fixed capacity. Hyperparameters as published AND tree/epoch counts fixed at
    the published seed-42 values (186 / 103 / 217 trees for XGBoost / LightGBM / CatBoost, the
    values already established in Section 5 as both the manuscript's published counts and the
    seed-42 derived counts). No eval_set, no early stopping; the model trains for exactly that
    many estimators/iterations at every seed. Random Forest has no capacity-selection step in
    either configuration (n_estimators fixed at 300 throughout the protocol), so Configuration A
    and B give identical Random Forest results by construction and the single Section 5 array is
    reused for both. MLP has no published epoch count in the manuscript; Configuration B fixes
    its iteration count at Configuration A's own seed-42 n_iter_ (early_stopping=False,
    n_iter_no_change set above that count so training-loss convergence cannot truncate it
    before max_iter is reached; n_iter_ actually run is reported for every fit, not just at
    seed 42, so any truncation would be visible).

Binding reporting rule: report whatever both configurations give. Neither configuration is
adjusted to approach 0.0038-0.0042. If neither reproduces the reported magnitude, the result is
recorded as "not reproduced under either prespecified configuration" and no third configuration
is attempted under this amendment.

Run from the repository root (original Anaconda environment; XGBoost/LightGBM/CatBoost on GPU,
asserted before every fit, no silent CPU fallback):
    python src/revision/section5_amendment8_bounded_reproduction.py --data-file <stage-1 CSV>
Writes: revision-results/five_seed/amendment8_bounded_reproduction.json
        revision-results/five_seed/amendment8_per_seed_auc.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np  # numpy / scikit-learn before torch (duplicate-OpenMP issue in this environment)
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "five_seed"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


s5 = _load("s5", ROOT / "src" / "revision" / "section5_five_seed.py")  # split(), assert_gpu(), devices, PUBLISHED_COUNTS
s3 = _load("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")  # build_cohort()

SEEDS = [42, 101, 202, 303, 404]
MODELS = ["CatBoost", "XGBoost", "LightGBM", "Random Forest", "MLP"]
SUFFIX = {"XGBoost": "xgb", "LightGBM": "lgbm_standalone", "CatBoost": "cat", "Random Forest": "rf"}
ORIGINAL = {"mean_auc_range": [0.7958, 0.7988], "sd_range": [0.0038, 0.0042]}
ANALYTIC_SE = 0.00437  # Hanley-McNeil at AUC 0.80, n = 10,281 (Amendment 6)


# ---- Configuration B: fixed-capacity fits (no eval_set, no early stopping) ----
def fit_xgb_fixed(d, s, n_estimators):
    s5.assert_gpu()
    m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                          random_state=s, verbosity=0, tree_method="hist", device=s5.XGB_DEVICE)
    m.fit(d["X_train"], d["y_train"])
    return m.predict_proba(d["X_test"])[:, 1]


def fit_lgb_fixed(d, s, n_estimators):
    s5.assert_gpu()
    m = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=63, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=s, verbose=-1, device=s5.LGB_DEVICE)
    m.fit(d["X_train"], d["y_train"])
    return m.predict_proba(d["X_test"])[:, 1]


def fit_cat_fixed(d, s, iterations):
    s5.assert_gpu()
    m = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.05, l2_leaf_reg=3, bagging_temperature=0.5,
                           random_state=s, verbose=0, task_type=s5.CAT_TASK, allow_writing_files=False)
    m.fit(d["X_train"], d["y_train"])
    return m.predict_proba(d["X_test"])[:, 1]


# ---- MLP: fit fresh for both configurations (not part of Section 5's model set) ----
def fit_mlp_A(d, s):
    """Configuration A: published hyperparameters, early stopping re-derived at this split."""
    m = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", alpha=1e-5, learning_rate_init=0.001,
                      solver="adam", max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=s)
    m.fit(d["X_train_sc"], d["y_train"])
    return m.predict_proba(d["X_test_sc"])[:, 1], int(m.n_iter_)


def fit_mlp_B(d, s, max_iter_fixed):
    """Configuration B: published hyperparameters, iteration count fixed at the Config-A seed-42 value,
    no validation-based early stopping. n_iter_no_change is set above max_iter_fixed so training-loss
    convergence cannot truncate training before max_iter_fixed is reached; n_iter_ is reported regardless."""
    m = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", alpha=1e-5, learning_rate_init=0.001,
                      solver="adam", max_iter=max_iter_fixed, early_stopping=False,
                      n_iter_no_change=max_iter_fixed + 1, random_state=s)
    m.fit(d["X_train_sc"], d["y_train"])
    return m.predict_proba(d["X_test_sc"])[:, 1], int(m.n_iter_)


def stats_row(model: str, config: str, aucs: dict[int, float], extra: dict) -> dict:
    a = np.array([aucs[s] for s in SEEDS])
    return {"model": model, "configuration": config, "auc_by_seed": {str(s): float(aucs[s]) for s in SEEDS},
            "mean_auc": float(a.mean()), "sd_auc_ddof1": float(a.std(ddof=1)),
            "min_auc": float(a.min()), "max_auc": float(a.max()), **extra}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    a = ap.parse_args()

    X, y = s3.build_cohort(a.data_file)
    splits = {s: s5.split(X, y, s) for s in SEEDS}
    for s in SEEDS:  # split identity check against the deposited Section 5 files
        yt = np.load(OUT / f"y_test_seed{s}.npy").astype(int)
        if not np.array_equal(splits[s]["y_test"], yt):
            raise SystemExit(f"SPLIT IDENTITY CHECK FAILED at seed {s}: section5_five_seed.split() does not "
                              f"reproduce the deposited y_test_seed{s}.npy.")
    print(f"Split identity check: PASS at all {len(SEEDS)} seeds (bit-identical to the deposited Section 5 splits).")

    rows = []

    # ---------------- Configuration A ----------------
    for model, suf in SUFFIX.items():  # XGBoost, LightGBM, CatBoost, Random Forest: reuse deposited Section 5 arrays
        aucs = {}
        for s in SEEDS:
            p = np.load(OUT / f"proba_{suf}_seed{s}.npy").astype(float)
            aucs[s] = float(roc_auc_score(splits[s]["y_test"], p))
        rows.append(stats_row(model, "A_per_split_refitting", aucs,
                              {"source": f"reused deposited revision-results/five_seed/proba_{suf}_seed{{s}}.npy"}))

    mlp_A_aucs, mlp_A_n_iter = {}, {}
    for s in SEEDS:
        p, n_iter = fit_mlp_A(splits[s], s)
        mlp_A_aucs[s] = float(roc_auc_score(splits[s]["y_test"], p))
        mlp_A_n_iter[s] = n_iter
    rows.append(stats_row("MLP", "A_per_split_refitting", mlp_A_aucs,
                          {"source": "fit fresh (not part of Section 5's model set)",
                           "n_iter_by_seed": {str(s): mlp_A_n_iter[s] for s in SEEDS}}))
    fixed_mlp_iter = mlp_A_n_iter[42]

    # ---------------- Configuration B ----------------
    PC = s5.PUBLISHED_COUNTS  # {"xgboost": 186, "lightgbm": 103, "catboost": 217}
    for model, fitfn, n in (
        ("XGBoost", fit_xgb_fixed, PC["xgboost"]),
        ("LightGBM", fit_lgb_fixed, PC["lightgbm"]),
        ("CatBoost", fit_cat_fixed, PC["catboost"]),
    ):
        aucs = {}
        for s in SEEDS:
            p = fitfn(splits[s], s, n)
            aucs[s] = float(roc_auc_score(splits[s]["y_test"], p))
        rows.append(stats_row(model, "B_fixed_capacity", aucs,
                              {"fixed_capacity": n, "fixed_capacity_source": "published (= Section 5 seed-42 derived)"}))

    rows.append(stats_row("Random Forest", "B_fixed_capacity",
                          {s: float(roc_auc_score(splits[s]["y_test"], np.load(OUT / "proba_rf_seed{}.npy".format(s)).astype(float)))
                           for s in SEEDS},
                          {"note": "identical to Configuration A by construction: n_estimators fixed at 300 in both "
                                   "configurations; no capacity-selection step exists to fix.",
                           "fixed_capacity": 300}))

    mlp_B_aucs, mlp_B_n_iter = {}, {}
    for s in SEEDS:
        p, n_iter = fit_mlp_B(splits[s], s, fixed_mlp_iter)
        mlp_B_aucs[s] = float(roc_auc_score(splits[s]["y_test"], p))
        mlp_B_n_iter[s] = n_iter
    rows.append(stats_row("MLP", "B_fixed_capacity", mlp_B_aucs,
                          {"fixed_capacity": fixed_mlp_iter,
                           "fixed_capacity_source": "Configuration A's own seed-42 n_iter_ (no published MLP epoch count exists)",
                           "n_iter_by_seed_actually_run": {str(s): mlp_B_n_iter[s] for s in SEEDS},
                           "truncated_before_fixed_capacity": {str(s): mlp_B_n_iter[s] < fixed_mlp_iter for s in SEEDS}}))

    df = pd.DataFrame(rows)
    long_rows = [{"configuration": r["configuration"], "model": r["model"], "seed": s, "auc": r["auc_by_seed"][str(s)]}
                 for r in rows for s in SEEDS]
    pd.DataFrame(long_rows).to_csv(OUT / "amendment8_per_seed_auc.csv", index=False, lineterminator="\n")

    summary = {}
    for config in ("A_per_split_refitting", "B_fixed_capacity"):
        sub = [r for r in rows if r["configuration"] == config]
        means = [r["mean_auc"] for r in sub]; sds = [r["sd_auc_ddof1"] for r in sub]
        summary[config] = {
            "per_model": {r["model"]: {k: v for k, v in r.items() if k not in ("model", "configuration")} for r in sub},
            "mean_auc_range_across_models": [min(means), max(means)],
            "sd_range_across_models": [min(sds), max(sds)],
        }

    sd_lo, sd_hi = ORIGINAL["sd_range"]
    reproduced_A = sd_lo <= summary["A_per_split_refitting"]["sd_range_across_models"][1] and \
                   summary["A_per_split_refitting"]["sd_range_across_models"][0] <= sd_hi
    reproduced_B = sd_lo <= summary["B_fixed_capacity"]["sd_range_across_models"][1] and \
                   summary["B_fixed_capacity"]["sd_range_across_models"][0] <= sd_hi
    conclusion = ("not reproduced under either prespecified configuration"
                  if not (reproduced_A or reproduced_B) else
                  "reproduced: " + ("Configuration A" if reproduced_A else "") +
                  (" and " if reproduced_A and reproduced_B else "") + ("Configuration B" if reproduced_B else ""))

    out = {
        "protocol_section": "Amendment 8 (prespecified)", "amendment_commit": "5fc0f37671b2b1bc9f5d5493ba3fafba7e32b120",
        "seeds": SEEDS, "models": MODELS, "original_submitted_values": ORIGINAL, "analytic_se_hanley_mcneil_amendment6": ANALYTIC_SE,
        "configuration_A_per_split_refitting": summary["A_per_split_refitting"],
        "configuration_B_fixed_capacity": summary["B_fixed_capacity"],
        "binding_conclusion": conclusion,
        "secondary_use_amendment6_matched_comparison": {
            "note": "A vs B on the same model set and splits is a matched test of the refitting-stabiliser effect "
                    "reported in Amendment 6 (where the comparison was between retrained §5 models and a single "
                    "fixed model over 200 drawn test sets, a different design).",
            # Mean of the five per-model between-split SDs. (Corrected: the first committed version took the
            # midpoint of the SD range here; see revision-results/RUN_LOG.md, Amendment 8 correction.)
            "mean_sd_configuration_A": float(np.mean([v["sd_auc_ddof1"] for v in summary["A_per_split_refitting"]["per_model"].values()])),
            "mean_sd_configuration_B": float(np.mean([v["sd_auc_ddof1"] for v in summary["B_fixed_capacity"]["per_model"].values()])),
            "mean_sd_definition": "mean over the five models of the between-split SD (ddof = 1)",
        },
    }
    with open(OUT / "amendment8_bounded_reproduction.json", "w", encoding="utf-8", newline="\n") as fh:
        import json; json.dump(out, fh, indent=2); fh.write("\n")

    pd.set_option("display.width", 200)
    print("\n" + df[["model", "configuration", "mean_auc", "sd_auc_ddof1", "min_auc", "max_auc"]].round(5).to_string(index=False))
    print(f"\nConfiguration A: mean-AUC range {summary['A_per_split_refitting']['mean_auc_range_across_models']}, "
          f"SD range {summary['A_per_split_refitting']['sd_range_across_models']}")
    print(f"Configuration B: mean-AUC range {summary['B_fixed_capacity']['mean_auc_range_across_models']}, "
          f"SD range {summary['B_fixed_capacity']['sd_range_across_models']}")
    print(f"Submitted: mean AUC {ORIGINAL['mean_auc_range']}, SD {ORIGINAL['sd_range']}")
    print(f"Binding conclusion: {conclusion}")


if __name__ == "__main__":
    main()
