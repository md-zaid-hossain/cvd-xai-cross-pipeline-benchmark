#!/usr/bin/env python3
"""Protocol Section 5: summary outputs from the seeds run (Sections 5.1 descriptive outputs and 5.2).

Reads revision-results/five_seed/ (per-seed y_test and probability arrays, the run manifest and the
seed-42 anchor record) written by section5_five_seed.py. Trains nothing.

- Test AUC per model and seed, 95% CI from a 1,000-resample percentile bootstrap with bootstrap seed = s
  (np.random.default_rng(s); sequential draws; one shared index vector per replicate).
- Within each seed the six ranked models are ranked by AUC (1 = highest; ties average). The standalone
  LightGBM is recorded but never ranked.
- Per-model mean, SD (ddof = 1), min, max; times ranked first; Kendall's W of the model rankings across
  seeds (tie-corrected); within-seed AUC spread vs between-seed SDs.
- Section 5.2 outcome: A (two or more different leaders) or B (same leader at every seed).

Run from the repository root:
    python src/revision/section5_summarise.py
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "five_seed"
PRED = ROOT / "artifacts" / "predictions"
SEED_ORDER = [42, 101, 202, 303, 404]
RANKED = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]
SUFFIX = {"Stacking Ensemble": "stack", "XGBoost": "xgb", "CatBoost": "cat", "FT-Transformer": "ft",
          "SAINT-inspired": "saint", "Random Forest": "rf"}
DEPOSITED = {"Stacking Ensemble": "stack_corrected", "XGBoost": "xgb", "CatBoost": "cat", "FT-Transformer": "ft",
             "SAINT-inspired": "saint", "Random Forest": "rf"}
PUBLISHED_COUNTS = {"xgboost": 186, "lightgbm": 103, "catboost": 217}


def kendall_w(R: np.ndarray) -> dict:
    n, m = R.shape
    S = float(((R.sum(axis=1) - m * (n + 1) / 2) ** 2).sum())
    T = sum(float((c ** 3 - c).sum()) for c in (np.unique(R[:, j], return_counts=True)[1] for j in range(m)))
    W = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    chi2 = m * (n - 1) * W
    return {"W": W, "chi2": chi2, "df": n - 1, "p": float(stats.chi2.sf(chi2, n - 1)), "n_models": n, "m_seeds": m}


def main() -> None:
    man = json.loads((OUT / "five_seed_run_manifest.json").read_text(encoding="utf-8"))
    runs = {r["run"]: r for r in man["runs"]}
    seeds = [s for s in SEED_ORDER if f"seed{s}" in runs]
    anchor = json.loads((OUT / "seed42_anchor_check.json").read_text(encoding="utf-8"))

    rows, lgb_rows = [], []
    for s in seeds:
        y = np.load(OUT / f"y_test_seed{s}.npy").astype(int)
        P = {m: np.load(OUT / f"proba_{SUFFIX[m]}_seed{s}.npy").astype(float) for m in RANKED}
        rng = np.random.default_rng(s); n = len(y); boot = {m: [] for m in RANKED}
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            if y[idx].min() == y[idx].max():
                continue
            for m in RANKED:
                boot[m].append(roc_auc_score(y[idx], P[m][idx]))
        auc = {m: roc_auc_score(y, P[m]) for m in RANKED}
        ranks = dict(zip(RANKED, stats.rankdata([-auc[m] for m in RANKED], method="average")))
        info = runs[f"seed{s}"]["models"]
        for m in RANKED:
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            it = info.get(m, {})
            best = (it.get("best_iteration") if m in ("XGBoost", "CatBoost") else
                    it.get("best_epoch") if m in ("FT-Transformer", "SAINT-inspired") else
                    "/".join(str(v) for v in it.get("trees", {}).values()) if m == "Stacking Ensemble" else None)
            extra = f"; weights from epoch {it['weights_used_epoch']} (restore=final)" if m == "SAINT-inspired" else ""
            rows.append({"model": m, "seed": s, "auc": auc[m], "auc_ci_low": lo, "auc_ci_high": hi,
                         "rank": float(ranks[m]), "best_iteration_or_epoch": f"{best}{extra}" if best is not None else ""})
        lgb_auc = (anchor["standalone_test_auc"]["lightgbm"] if s == 42 else info["LightGBM (standalone)"]["test_auc"])
        lgb_rows.append({"model": "LightGBM (standalone, not ranked)", "seed": s, "auc": lgb_auc, "auc_ci_low": np.nan,
                         "auc_ci_high": np.nan, "rank": np.nan,
                         "best_iteration_or_epoch": str(info["LightGBM (standalone)"]["best_iteration"])})
    long = pd.DataFrame(rows + lgb_rows)
    long.to_csv(OUT / "five_seed_auc_long.csv", index=False, lineterminator="\n")

    ranked = long[long["model"].isin(RANKED)]
    rank_tab = ranked.pivot(index="model", columns="seed", values="rank").loc[RANKED, seeds]
    rank_tab.columns = [f"seed_{s}" for s in seeds]
    rank_tab.to_csv(OUT / "five_seed_rank_table.csv", lineterminator="\n")
    auc_tab = ranked.pivot(index="model", columns="seed", values="auc").loc[RANKED, seeds]

    leaders = {s: str(auc_tab[s].idxmax()) for s in seeds}
    summary = pd.DataFrame({
        "model": RANKED,
        "mean_auc": auc_tab.mean(axis=1).values, "sd_auc_ddof1": auc_tab.std(axis=1, ddof=1).values,
        "min_auc": auc_tab.min(axis=1).values, "max_auc": auc_tab.max(axis=1).values,
        "mean_rank": rank_tab.mean(axis=1).values,
        "times_ranked_first": [sum(leaders[s] == m for s in seeds) for m in RANKED]})
    summary.to_csv(OUT / "five_seed_summary.csv", index=False, lineterminator="\n")

    within = {f"seed_{s}": float(auc_tab[s].max() - auc_tab[s].min()) for s in seeds}
    between_sd = dict(zip(RANKED, auc_tab.std(axis=1, ddof=1).values.astype(float)))
    conc = {"protocol_section": "5.1 descriptive outputs", "seeds_run": seeds,
            "kendall_w_model_rankings_across_seeds": kendall_w(rank_tab.values),
            "within_seed_auc_spread_among_ranked_models": within,
            "within_seed_spread_range": [min(within.values()), max(within.values())],
            "between_seed_sd_per_model": between_sd,
            "between_seed_sd_range": [min(between_sd.values()), max(between_sd.values())]}
    with open(OUT / "five_seed_concordance.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(conc, fh, indent=2); fh.write("\n")

    distinct = sorted(set(leaders.values()))
    outcome = "A" if len(distinct) >= 2 else "B"
    out = {"protocol_section": "5.2", "seeds_run": seeds, "reduction_triggered": runs["seed101"]["exceeds_eight_hours"],
           "leader_per_seed": {str(s): leaders[s] for s in seeds},
           "distinct_leaders": distinct, "k_distinct_leaders": len(distinct),
           "outcome": outcome,
           "outcome_label": "A (unstable ranking)" if outcome == "A" else "B (stable leader)",
           "wording_values": {"k": len(distinct), "between_seed_sd_range": conc["between_seed_sd_range"],
                              "within_seed_spread_range": conc["within_seed_spread_range"]}}
    if outcome == "B":
        lead = distinct[0]
        adv = [float(auc_tab.loc[lead, s] - auc_tab[s].drop(lead).max()) for s in seeds]
        out["wording_values"].update({"model": lead, "mean_advantage": float(np.mean(adv)), "advantage_range": [min(adv), max(adv)]})
    with open(OUT / "five_seed_outcome.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    it_rows = []
    for s in seeds:
        info = runs[f"seed{s}"]["models"]
        trees = info["Stacking Ensemble"]["trees"]
        it_rows.append({"seed": s,
                        "xgboost_best_iteration": info["XGBoost"]["best_iteration"], "xgboost_trees_used": trees["xgboost"],
                        "lightgbm_best_iteration": info["LightGBM (standalone)"]["best_iteration"], "lightgbm_trees_used": trees["lightgbm"],
                        "catboost_best_iteration": info["CatBoost"]["best_iteration"], "catboost_trees_used": trees["catboost"],
                        "deposited_seed42_trees_xgb_lgb_cat": "186/103/217"})
    pd.DataFrame(it_rows).to_csv(OUT / "five_seed_stacking_iterations.csv", index=False, lineterminator="\n")

    y42 = np.load(OUT / "y_test_seed42.npy").astype(int)
    dep_rows = []
    for m in RANKED:
        p = np.load(OUT / f"proba_{SUFFIX[m]}_seed42.npy").astype(float)
        dp = np.load(PRED / f"proba_{DEPOSITED[m]}.npy").astype(float)
        dep_rows.append({"model": m, "auc_seed42_retrained": roc_auc_score(y42, p), "auc_deposited": roc_auc_score(y42, dp),
                         "delta_auc": roc_auc_score(y42, p) - roc_auc_score(y42, dp),
                         "pearson_correlation": float(pearsonr(p, dp)[0]),
                         "max_abs_diff": float(np.max(np.abs(p - dp)))})
    pd.DataFrame(dep_rows).to_csv(OUT / "seed42_vs_deposited.csv", index=False, lineterminator="\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    off = np.linspace(-0.3, 0.3, len(seeds))
    for k, s in enumerate(seeds):
        sub = ranked[ranked["seed"] == s].set_index("model").loc[RANKED]
        x = np.arange(len(RANKED)) + off[k]
        ax.errorbar(x, sub["auc"], yerr=[sub["auc"] - sub["auc_ci_low"], sub["auc_ci_high"] - sub["auc"]],
                    fmt="o", ms=4, capsize=2, lw=1, label=f"seed {s}")
    ax.set_xticks(range(len(RANKED))); ax.set_xticklabels(RANKED, rotation=20, ha="right")
    ax.set_ylabel("Test ROC-AUC (95% bootstrap CI)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Test AUC by seed ({len(seeds)} stratified splits); outcome {outcome}")
    ax.legend(fontsize=8, ncol=len(seeds), loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)  # below the axes, clear of the CIs
    fig.tight_layout(); fig.savefig(OUT / "figure_five_seed_auc.png", dpi=300); plt.close(fig)

    pd.set_option("display.width", 200)
    print("AUC by seed\n" + auc_tab.round(5).to_string())
    print("\nRanks\n" + rank_tab.to_string())
    print("\n" + summary.round(5).to_string(index=False))
    print("\n" + json.dumps({"leaders": out["leader_per_seed"], "outcome": out["outcome_label"],
                             "kendall_w": conc["kendall_w_model_rankings_across_seeds"],
                             "within_seed_spread_range": conc["within_seed_spread_range"],
                             "between_seed_sd_range": conc["between_seed_sd_range"]}, indent=2))
    print("\nseed 42 vs deposited\n" + pd.DataFrame(dep_rows).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
