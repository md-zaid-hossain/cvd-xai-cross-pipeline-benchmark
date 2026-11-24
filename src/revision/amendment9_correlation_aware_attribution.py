#!/usr/bin/env python3
"""Protocol Amendment 9 (prespecified, pushed alone in 9c9f8b5): correlation-aware attribution, XGBoost pipeline.

Run specification (fixed in the amendment):
  Model      retrained seed-42 XGBoost (artifacts/models-revision/xgb_seed42.json), raw (unscaled) features.
  Rows       Section 4.1 common sample of the primary test set (n = 1,000; common_sample.load_common_sample()).
  Output     raw log-odds margin for all three attributions. TreeSHAP in this shap version explains the
             best-iteration trees (Section 4.2 additivity check), so the Partition explainer is given the same
             function, model.predict(X, output_margin=True). Additivity is recorded for all three.
  Background 200 training rows: np.random.seed(42); np.random.choice(len(X_train), 200, replace=False)
             (interventional TreeSHAP and the Partition masker).
  Hierarchy  |Spearman rho| of the 13 features on the training set; distance 1 - |rho|; average linkage.
  Partition  shap.PartitionExplainer with shap.maskers.Partition(background, clustering=Z), max_evals = 2,000.
             Full expansion checked by recomputing the first 50 records at max_evals = 4,000.
  Ranks      scipy.stats.rankdata(-mean|attribution|, method='average').

(a) TreeSHAP, feature_perturbation "interventional" and "tree_path_dependent": mean |SHAP|, ranks, Spearman
    between the two rankings, rank change (interventional minus tree_path_dependent).
(b) Partition (Owen value) attribution: ranking, Spearman with each ranking from (a), dendrogram structure.
(c) For each method: summed mean |attribution| of {SBP, DBP, PP, MAP} and each member's share of that sum.

Binding reporting rule: whatever is obtained is reported; no setting is tuned toward the expectation.

Run from the repository root:
    python src/revision/amendment9_correlation_aware_attribution.py --data-file <stage-1 CSV>
Writes: revision-results/xai/amendment9_attribution_by_feature.csv
        revision-results/xai/amendment9_correlation_aware_attribution.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "xai"
_spec = importlib.util.spec_from_file_location("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)
sys.path.insert(0, str(ROOT / "src" / "revision"))
from common_sample import load_common_sample  # noqa: E402

FEAT = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose", "Smoking", "Alcohol", "Active", "PP", "MAP",
        "RiskCount"]
HAEMO = ["SBP", "DBP", "PP", "MAP"]
MAX_EVALS, CHECK_EVALS, CHECK_N = 2000, 4000, 50


def ranks(v: np.ndarray) -> np.ndarray:
    return rankdata(-v, method="average")


def dendrogram_structure(Z: np.ndarray) -> list[dict]:
    members = {i: [FEAT[i]] for i in range(len(FEAT))}
    out = []
    for k, (a, b, h, n) in enumerate(Z):
        a, b = int(a), int(b)
        members[len(FEAT) + k] = members[a] + members[b]
        out.append({"merge": k + 1, "joins": [members[a], members[b]], "distance_1_minus_abs_rho": float(h),
                    "abs_spearman_at_merge": float(1 - h), "cluster_size": int(n)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    a = ap.parse_args()

    import shap
    import xgboost as xgb

    X, y = s3.build_cohort(a.data_file)
    (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, _sc) = s3.primary_split(X, y)
    if not np.array_equal(y_test, np.load(ROOT / "artifacts" / "predictions" / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    xai_idx, sample_sha = load_common_sample()
    Xs = X_test[xai_idx].astype(float)

    model = xgb.XGBClassifier()
    model.load_model(str(ROOT / "artifacts" / "models-revision" / "xgb_seed42.json"))
    model.set_params(device="cpu")
    margin = lambda Z: model.predict(np.asarray(Z, dtype=float), output_margin=True)
    fx = margin(Xs)

    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), 200, replace=False)
    bg = X_train[bg_idx].astype(float)

    res, checks, timing = {}, {}, {}
    # ---- (a) TreeSHAP, both perturbation settings
    t0 = time.time()
    e_int = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="raw")
    sv_int = np.asarray(e_int.shap_values(Xs))
    timing["treeshap_interventional_s"] = time.time() - t0
    t0 = time.time()
    e_tpd = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv_tpd = np.asarray(e_tpd.shap_values(Xs))
    timing["treeshap_tree_path_dependent_s"] = time.time() - t0
    for name, sv, e in (("interventional", sv_int, e_int), ("tree_path_dependent", sv_tpd, e_tpd)):
        base = float(np.ravel(e.expected_value)[0])
        checks[f"additivity_max_abs_err_{name}"] = float(np.max(np.abs(sv.sum(axis=1) + base - fx)))
        res[name] = sv

    # tree_path_dependent must reproduce the deposited Section 4.2 XGBoost TreeSHAP importances
    dep = pd.read_csv(OUT / "importance_xgb_treeshap.csv").set_index("feature").loc[FEAT, "mean_abs_shap"].values
    checks["tree_path_dependent_vs_section4_2_max_abs_diff"] = float(np.max(np.abs(np.abs(sv_tpd).mean(0) - dep)))

    # ---- (b) Partition (Owen values) on a |Spearman| hierarchy built on the training set
    rho = spearmanr(X_train).correlation
    D = 1 - np.abs(rho)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    masker = shap.maskers.Partition(bg, clustering=Z)
    pe = shap.PartitionExplainer(margin, masker)
    t0 = time.time()
    exp = pe(Xs, max_evals=MAX_EVALS, silent=True)
    timing["partition_s"] = time.time() - t0
    sv_part = np.asarray(exp.values)
    base_p = np.asarray(exp.base_values).reshape(-1)
    checks["additivity_max_abs_err_partition"] = float(np.max(np.abs(sv_part.sum(axis=1) + base_p - fx)))
    exp_chk = pe(Xs[:CHECK_N], max_evals=CHECK_EVALS, silent=True)
    diff = float(np.max(np.abs(np.asarray(exp_chk.values) - sv_part[:CHECK_N])))
    checks["partition_full_expansion_check"] = {
        "first_n_records": CHECK_N, "max_evals_primary": MAX_EVALS, "max_evals_check": CHECK_EVALS,
        "max_abs_diff": diff, "fully_expanded": bool(diff < 1e-9)}
    res["partition_owen"] = sv_part

    # ---- summaries
    methods = ["interventional", "tree_path_dependent", "partition_owen"]
    mabs = {m: np.abs(res[m]).mean(axis=0) for m in methods}
    rk = {m: ranks(mabs[m]) for m in methods}
    df = pd.DataFrame({"feature": FEAT})
    for m in methods:
        df[f"mean_abs_{m}"] = mabs[m]
        df[f"rank_{m}"] = rk[m]
    df["rank_change_interventional_minus_tree_path_dependent"] = rk["interventional"] - rk["tree_path_dependent"]
    df.to_csv(OUT / "amendment9_attribution_by_feature.csv", index=False, lineterminator="\n")

    sp = lambda u, v: float(spearmanr(rk[u], rk[v]).correlation)
    haemo = {}
    for m in methods:
        vals = {f: float(mabs[m][FEAT.index(f)]) for f in HAEMO}
        tot = sum(vals.values())
        haemo[m] = {"summed_mean_abs_attribution": tot,
                    "share_of_all_13_features": float(tot / mabs[m].sum()),
                    "member_share_of_set": {f: vals[f] / tot for f in HAEMO},
                    "member_mean_abs": vals}

    out = {
        "protocol_section": "Amendment 9 (prespecified)", "amendment_commit": "9c9f8b5d0eada3bf8cf055a2dce78d6107a8dd37",
        "run_specification": {
            "model": "artifacts/models-revision/xgb_seed42.json (raw features; best-iteration margin)",
            "rows": "Section 4.1 common sample", "common_sample_n": int(len(xai_idx)), "common_sample_sha256": sample_sha,
            "output": "raw log-odds margin", "background": "200 training rows, np.random.seed(42); np.random.choice(len(X_train), 200, replace=False)",
            "background_row_indices_first5": [int(i) for i in bg_idx[:5]],
            "hierarchy": "|Spearman rho| on the training set (n = %d); distance 1 - |rho|; average linkage" % len(X_train),
            "partition_max_evals": MAX_EVALS, "ranks": "rankdata(-mean|attribution|, method='average')",
            "versions": {"shap": shap.__version__, "xgboost": xgb.__version__, "numpy": np.__version__}},
        "a_treeshap": {
            "spearman_interventional_vs_tree_path_dependent": sp("interventional", "tree_path_dependent"),
            "rank_change_interventional_minus_tree_path_dependent": dict(zip(FEAT, df["rank_change_interventional_minus_tree_path_dependent"].tolist()))},
        "b_partition": {
            "spearman_vs_interventional": sp("partition_owen", "interventional"),
            "spearman_vs_tree_path_dependent": sp("partition_owen", "tree_path_dependent"),
            "dendrogram": dendrogram_structure(Z),
            "abs_spearman_matrix_training_set": {FEAT[i]: {FEAT[j]: float(abs(rho[i, j])) for j in range(13)} for i in range(13)}},
        "c_haemodynamic_set": haemo,
        "per_feature": {m: {f: {"mean_abs": float(mabs[m][i]), "rank": float(rk[m][i])} for i, f in enumerate(FEAT)} for m in methods},
        "checks": checks, "timing_seconds": timing,
        "prespecified_expectation": ("haemodynamic set's total attribution similar across the three methods; its distribution "
                                     "among the four members differs"),
    }
    with open(OUT / "amendment9_correlation_aware_attribution.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    pd.set_option("display.width", 220)
    print(df.round(5).to_string(index=False))
    print("\nSpearman: int vs tpd %.4f | partition vs int %.4f | partition vs tpd %.4f" %
          (sp("interventional", "tree_path_dependent"), sp("partition_owen", "interventional"), sp("partition_owen", "tree_path_dependent")))
    for m in methods:
        h = haemo[m]
        print(f"(c) {m:20s} sum {h['summed_mean_abs_attribution']:.4f} ({h['share_of_all_13_features']:.1%} of all) shares " +
              ", ".join(f"{f} {s:.1%}" for f, s in h["member_share_of_set"].items()))
    print("\nchecks:", json.dumps(checks, indent=1))
    print("timing:", {k: round(v, 1) for k, v in timing.items()})
    for d in dendrogram_structure(Z):
        print(f"  merge {d['merge']:2d} at |rho| {d['abs_spearman_at_merge']:.3f}: {d['joins'][0]} + {d['joins'][1]}")


if __name__ == "__main__":
    main()
