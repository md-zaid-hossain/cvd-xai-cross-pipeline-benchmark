#!/usr/bin/env python3
"""Figure 4: pairwise DeLong p-value heatmap for the 15 models, from deposited output only.

The visual encoding is the notebook's (notebooks/core/Statistical_Significance_Tests.ipynb, cell 15): the same
colour map on -log10 p (0 to 4), white cell grid, grey diagonal, significance labels and their colour rule, model
order, family divider lines and family label colours, colour bar ticks, and the two legends. What differs:
  - the notebook took p_matrix, model_names and y_test from its own earlier cells, so it cannot run on its own; here
    p_matrix is rebuilt from the deposited results/tables/validated_delong_all_pairs.csv (105 pairs) and n from
    artifacts/predictions/y_test.npy;
  - the figure is drawn at its print width (165.1 mm, the manuscript text width) with every text at least 6.5 pt,
    so its layout is recomputed: the notebook's fixed axes positions at 16 x 12 in cannot hold larger labels;
  - the bold in-image title is omitted (the caption carries it); the grey line giving n, sidedness and alpha is kept.
Checks: 105 pairs, 15 models, a symmetric matrix and a unit diagonal.

Run from the repository root:
    python src/revision/figure4_delong_heatmap.py
Writes: results/figures/main/Figure_4_DeLong_Heatmap_v2.pdf   (vector; the 15 x 15 cells are an embedded image, as
        matplotlib's imshow always produces)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "results" / "tables" / "validated_delong_all_pairs.csv"
Y_TEST = ROOT / "artifacts" / "predictions" / "y_test.npy"
OUT = ROOT / "results" / "figures" / "main" / "Figure_4_DeLong_Heatmap_v2.pdf"
WIDTH_IN = 165.1 / 25.4
PT_LABEL, PT_CELL, PT_SMALL = 7.0, 6.5, 6.5   # tick labels; cell symbols; colour bar, legends, subtitle

# Row/column order and display labels as in the submitted figure (families: 7 classical, 4 boosting and stacking,
# 4 deep learning and self-supervised)
MODELS = [("Decision Tree", "Decision Tree"), ("Naive Bayes", "Naive Bayes"), ("k-NN", "k-NN"),
          ("Logistic Regression", "Logistic Reg."), ("Random Forest", "Random Forest"), ("SVM (RBF)", "SVM (RBF)"),
          ("MLP", "MLP"), ("XGBoost", "XGBoost"), ("LightGBM", "LightGBM"), ("CatBoost", "CatBoost"),
          ("Stacking Ensemble", "Stacking Ensemble"), ("TabNet", "TabNet"), ("FT-Transformer", "FT-Transformer"),
          ("VIME-inspired SSL", "VIME-inspired"), ("SAINT-inspired", "SAINT-inspired")]


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    if len(pairs) != 105:
        raise SystemExit(f"expected 105 pairs, found {len(pairs)}")
    idx = {name: i for i, (name, _) in enumerate(MODELS)}
    n_models = len(MODELS)
    p_matrix = np.ones((n_models, n_models))
    for a, b, p in zip(pairs["Model A"].str.strip(), pairs["Model B"].str.strip(), pairs["p"]):
        p_matrix[idx[a], idx[b]] = p_matrix[idx[b], idx[a]] = p
    assert np.allclose(p_matrix, p_matrix.T) and np.all(np.diag(p_matrix) == 1)
    model_names = [label for _, label in MODELS]
    n_test = len(np.load(Y_TEST))

    log_p = -np.log10(np.clip(p_matrix, 1e-10, 1.0))
    np.fill_diagonal(log_p, np.nan)
    colors_map = [(0.90, 0.24, 0.14), (0.97, 0.55, 0.32), (1.00, 0.87, 0.68), (0.85, 0.96, 0.85),
                  (0.22, 0.70, 0.52), (0.01, 0.30, 0.22)]
    cmap = LinearSegmentedColormap.from_list("delong_sig", colors_map, N=512)
    cmap.set_bad("#E2E2E2")

    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 1.02), facecolor="#F6F6F4")
    ax = fig.add_axes([0.205, 0.235, 0.62, 0.62])
    cax = fig.add_axes([0.845, 0.29, 0.018, 0.51])
    im = ax.imshow(log_p, cmap=cmap, vmin=0, vmax=4, aspect="equal", interpolation="nearest")
    for x in np.arange(-0.5, n_models, 1):
        ax.axhline(x, color="white", lw=0.8)
        ax.axvline(x, color="white", lw=0.8)
    for k in range(n_models):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, facecolor="#E2E2E2", edgecolor="white", lw=0.8, zorder=3))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            sym = sig_stars(p_matrix[i, j])
            ax.text(j, i, sym, ha="center", va="center", fontsize=PT_CELL, fontweight="bold" if sym != "n.s." else "normal",
                    color="white" if log_p[i, j] > 1.5 else "#1C1C1C", zorder=5)
    for boundary in [6.5, 10.5]:
        ax.axhline(boundary, color="#222222", lw=1.5, zorder=6)
        ax.axvline(boundary, color="#222222", lw=1.5, zorder=6)
    family_colours = ["#5C3690"] * 7 + ["#0D6B8A"] * 4 + ["#B5370F"] * 4
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=42, ha="right", rotation_mode="anchor", fontsize=PT_LABEL)
    ax.set_yticklabels(model_names, fontsize=PT_LABEL)
    for label, colour in zip(ax.get_xticklabels(), family_colours):
        label.set_color(colour)
    for label, colour in zip(ax.get_yticklabels(), family_colours):
        label.set_color(colour)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["1.0", "0.1", "0.01", "0.001", "<0.0001"], fontsize=PT_SMALL)
    cbar.ax.set_title("p-value", fontsize=PT_SMALL, pad=6)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2)

    fig.text(0.515, 0.885, f"{n_models} models · held-out test set (n = {n_test:,}) · two-sided · α = 0.05",
             ha="center", fontsize=PT_SMALL, color="#555555")
    family_handles = [mpatches.Patch(facecolor="#5C3690", edgecolor="none", label="Classical ML (7 models)"),
                      mpatches.Patch(facecolor="#0D6B8A", edgecolor="none", label="Gradient Boosting & Stacking (4)"),
                      mpatches.Patch(facecolor="#B5370F", edgecolor="none", label="Deep Learning & Self-Supervised (4)")]
    fig.legend(handles=family_handles, loc="lower center", ncol=3, frameon=True, fontsize=PT_SMALL,
               bbox_to_anchor=(0.515, 0.045), fancybox=True, framealpha=0.95, edgecolor="#CCCCCC", handlelength=1.4,
               handleheight=1.0, columnspacing=1.4)
    legend_handles = [mpatches.Patch(facecolor="#014F3C", edgecolor="none", label="*** p < 0.001"),
                      mpatches.Patch(facecolor="#3DB882", edgecolor="none", label="**  p < 0.01"),
                      mpatches.Patch(facecolor="#CEECCE", edgecolor="#AAAAAA", label="*   p < 0.05"),
                      mpatches.Patch(facecolor="#E55C3C", edgecolor="none", label="n.s. p ≥ 0.05")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, fontsize=PT_SMALL,
               bbox_to_anchor=(0.515, 0.0), fancybox=True, framealpha=0.95, edgecolor="#CCCCCC", handlelength=1.4,
               handleheight=1.0, columnspacing=1.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.03, facecolor="#F6F6F4",
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
