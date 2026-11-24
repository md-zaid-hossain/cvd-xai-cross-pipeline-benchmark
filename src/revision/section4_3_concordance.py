#!/usr/bin/env python3
"""Protocol Section 4.3: rankings and concordance on the common sample (PRIMARY result).

Implements docs/ANALYSIS_PROTOCOL.md Section 4.3.
- Ranks 1-13 within each pipeline: scipy.stats.rankdata(-importance, method='average').
- Pairwise Spearman rho for the 10 pipeline pairs.
- Kendall's W across the 5 pipelines with tie correction:
      W = 12 S / (m^2 (n^3 - n) - m * sum_j T_j),  T_j = sum (t^3 - t) over tie groups of rater j,
      chi2 = m (n - 1) W,  df = n - 1.
- Each pipeline's top-3 feature set.
- Rank-change table against the ORIGINAL unequal-sample ranks (submitted Table 7 / Figure 6).

Original ranks: the submitted manuscript's Table 7 (17 Oct 2026). Each column was verified
against the original notebooks' stored outputs (Classical_Models_CVD.ipynb cell 35;
Stacking_Ensemble_CVD_GPU.ipynb cell 31; TabNet_CVD_Clean.ipynb cell 21;
FT_Transformer_CVD.ipynb cell 22; SAINT_CVD_Complete.ipynb cell 25) -- all 65 ranks match.
As a further check, the original Spearman matrix and W are recomputed from these ranks and
compared with results/tables/cross_pipeline_spearman_matrix.csv (read only; never modified)
and the published W = 0.657, chi2(12) = 39.43.

Pipeline numbering and column order follow the protocol: 1 MLP permutation, 2 XGBoost TreeSHAP,
3 TabNet attention, 4 FT-Transformer KernelSHAP, 5 SAINT-inspired permutation.

Run from the repository root:
    python src/revision/section4_3_concordance.py
"""
from __future__ import annotations

import itertools
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

ROOT = Path(__file__).resolve().parents[2]
XAI = ROOT / "revision-results" / "xai"
FEAT = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
        "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
PIPES = [  # (protocol number, label, file, importance column)
    (1, "MLP Perm.", "importance_mlp_permutation.csv", "importance_mean_auc_drop"),
    (2, "XGBoost SHAP", "importance_xgb_treeshap.csv", "mean_abs_shap"),
    (3, "TabNet Attn.", "importance_tabnet_attention.csv", "normalised_attention_importance"),
    (4, "FT-Transformer SHAP", "importance_ft_kernelshap.csv", "mean_abs_shap_class1"),
    (5, "SAINT-inspired Perm.", "importance_saint_permutation.csv", "importance_mean_auc_drop"),
]
LABELS = [p[1] for p in PIPES]
# Submitted Table 7 (verified against notebook outputs; see docstring).
ORIGINAL = {
    "SBP": [4, 1, 1, 1, 3], "Age": [2, 2, 2, 3, 2], "MAP": [6, 3, 5, 2, 8], "PP": [3, 7, 7, 6, 1],
    "Cholesterol": [1, 4, 4, 4, 13], "BMI": [8, 5, 3, 5, 7], "DBP": [5, 9, 9, 9, 5],
    "Active": [7, 6, 12, 10, 6], "Glucose": [10, 8, 6, 7, 11], "RiskCount": [9, 10, 8, 12, 10],
    "Smoking": [12, 11, 13, 8, 9], "Alcohol": [13, 12, 11, 13, 4], "Gender": [11, 13, 10, 11, 12],
}


def kendall_w(R: np.ndarray) -> dict:
    """R: n items x m raters, ranks. Tie-corrected Kendall's W."""
    n, m = R.shape
    S = float(((R.sum(axis=1) - m * (n + 1) / 2) ** 2).sum())
    T = 0.0
    for j in range(m):
        _, counts = np.unique(R[:, j], return_counts=True)
        T += float((counts ** 3 - counts).sum())
    W = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    chi2 = m * (n - 1) * W
    return {"W": W, "chi2": chi2, "df": n - 1, "p": float(stats.chi2.sf(chi2, n - 1)),
            "m_raters": m, "n_items": n, "tie_correction_sum_T": T}


def spearman_matrix(R: pd.DataFrame) -> pd.DataFrame:
    M = pd.DataFrame(np.eye(len(R.columns)), index=R.columns, columns=R.columns)
    for a, b in itertools.combinations(R.columns, 2):
        M.loc[a, b] = M.loc[b, a] = stats.spearmanr(R[a], R[b])[0]
    return M


def main() -> None:
    imp = pd.DataFrame({lab: pd.read_csv(XAI / f).set_index("feature").loc[FEAT, col] for _, lab, f, col in PIPES})
    ranks = imp.apply(lambda c: pd.Series(stats.rankdata(-c.values, method="average"), index=c.index))
    ranks.index.name = "feature"
    rk = ranks.copy()
    rk["average_rank"] = ranks.mean(axis=1)
    rk = rk.sort_values("average_rank")
    rk.to_csv(XAI / "rank_matrix_common_sample.csv", lineterminator="\n")

    sp = spearman_matrix(ranks)
    sp.index.name = "pipeline"
    sp.round(6).to_csv(XAI / "spearman_matrix_common_sample.csv", lineterminator="\n")
    pairs = [{"pipeline_a": a, "pipeline_b": b, "spearman_rho": float(sp.loc[a, b])}
             for a, b in itertools.combinations(LABELS, 2)]

    kw = kendall_w(ranks.values)
    orig = pd.DataFrame(ORIGINAL, index=LABELS).T.loc[FEAT]
    kw_orig = kendall_w(orig.values.astype(float))
    sp_orig = spearman_matrix(orig)
    committed = pd.read_csv(ROOT / "results" / "tables" / "cross_pipeline_spearman_matrix.csv").set_index("Pipeline")
    committed_diff = float(np.max(np.abs(committed.loc[LABELS, LABELS].values - sp_orig.values.round(3))))

    top3 = {lab: list(imp[lab].sort_values(ascending=False).index[:3]) for lab in LABELS}
    kw_out = {
        "protocol_section": "4.3", "status": "PRIMARY reported concordance result (common sample, n = 1,000)",
        "kendall_w_common_sample": kw,
        "pairwise_spearman_common_sample": pairs,
        "spearman_range_common_sample": [min(p["spearman_rho"] for p in pairs), max(p["spearman_rho"] for p in pairs)],
        "top3_common_sample": top3,
        "original_unequal_sample_sensitivity": {
            "source": "submitted Table 7, verified against notebook stored outputs",
            "kendall_w_recomputed": kw_orig,
            "published_W_chi2": {"W": 0.657, "chi2": 39.43, "df": 12},
            "spearman_matrix_max_abs_diff_vs_committed_cross_pipeline_spearman_matrix_csv": committed_diff,
        },
    }
    with open(XAI / "kendall_w_common_sample.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(kw_out, fh, indent=2)
        fh.write("\n")

    rows = []
    for f in FEAT:
        for k, lab in enumerate(LABELS):
            o, c = float(orig.loc[f, lab]), float(ranks.loc[f, lab])
            rows.append({"feature": f, "pipeline_number": PIPES[k][0], "pipeline": lab,
                         "original_rank_unequal_sample": o, "common_sample_rank": c,
                         "change_common_minus_original": c - o})
    rc = pd.DataFrame(rows)
    rc.to_csv(XAI / "rank_comparison_original_vs_common.csv", index=False, lineterminator="\n")

    fig, ax = plt.subplots(figsize=(8, 7))
    show = rk[LABELS]
    im = ax.imshow(show.values, cmap="viridis_r", vmin=1, vmax=13, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{n}. {l}" for n, l, _, _ in PIPES], rotation=30, ha="right")
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show.index)
    for i in range(show.shape[0]):
        for j in range(5):
            v = show.values[i, j]
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8, color="black" if v < 7 else "white")  # dark text on light (low-rank) cells
    fig.colorbar(im, ax=ax, label="Rank (1 = most important)")
    ax.set_title(f"Feature ranks on the common sample (n = 1,000); Kendall's W = {kw['W']:.3f}")
    fig.tight_layout(); fig.savefig(XAI / "figure_rank_heatmap_common_sample.png", dpi=300); plt.close(fig)

    pd.set_option("display.width", 220)
    print("RANKS (common sample), sorted by average rank\n" + rk.to_string())
    print("\nSPEARMAN (common sample)\n" + sp.round(3).to_string())
    print(f"\nKendall W common = {kw['W']:.4f}, chi2({kw['df']}) = {kw['chi2']:.2f}, p = {kw['p']:.3g}")
    print(f"Kendall W original (recomputed) = {kw_orig['W']:.4f}, chi2 = {kw_orig['chi2']:.2f}; "
          f"original Spearman vs committed csv max |diff| = {committed_diff:.4f}")
    print("\nTOP-3:", json.dumps(top3))


if __name__ == "__main__":
    main()
