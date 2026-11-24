#!/usr/bin/env python3
"""Manuscript Figure 6: cross-pipeline feature-rank heatmap on the common 1,000-record sample.

Regenerated from revision-results/xai/rank_matrix_common_sample.csv (Section 4.3, SAINT-inspired labels
corrected). The plotting style reproduces the submitted figure (word/media/image6.png of the 17 Oct
manuscript): reversed red-yellow-green colour map on ranks 1-13, white cell borders, bold integer
labels (white on the darkest cells), colour bar ticks 1/4/7/10/13, two-line pipeline labels. No source
code for the submitted figure exists, so the style was matched from the image. Changes from it: the
data; row order follows the common-sample mean rank (as in Table 7); the baked-in title now reads
"Figure 6" and "Cross-Pipeline", matching the caption; the anomaly line (double dagger / dagger) is
removed because both anomalies were artefacts of the SAINT labelling defect.

Run from the repository root:
    python src/revision/figure6_rank_heatmap.py
Writes: results/figures/main/Figure_6_Cross_Pipeline_Rank_Heatmap.png
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision-results" / "xai" / "rank_matrix_common_sample.csv"
OUT = ROOT / "results" / "figures" / "main" / "Figure_6_Cross_Pipeline_Rank_Heatmap.png"
COLUMNS = {"MLP Perm.": "MLP\nPermutation", "XGBoost SHAP": "XGBoost\nSHAP", "TabNet Attn.": "TabNet\nAttention",
           "FT-Transformer SHAP": "FT-Transformer\nSHAP", "SAINT-inspired Perm.": "SAINT\nPermutation"}


def main() -> None:
    df = pd.read_csv(SRC).set_index("feature")
    ranks = df[list(COLUMNS)].rename(columns=COLUMNS)  # row order as in the CSV (mean-rank order)
    if not ((ranks.values >= 1) & (ranks.values <= 13)).all():
        raise SystemExit("ranks outside 1-13")

    # Same aspect ratio as the submitted image (1100 x 886 px), so the manuscript's existing frame
    # (5.0 x 3.88 in) and top crop (srcRect t = 3.583%) apply without distortion. As in the submitted
    # image, the first title line sits inside that cropped top band.
    fig, ax = plt.subplots(figsize=(10, 10 * 886 / 1100))
    sns.heatmap(ranks, cmap="RdYlGn_r", vmin=1, vmax=13, annot=False, linewidths=2, linecolor="white",
                cbar_kws={"ticks": [1, 4, 7, 10, 13], "shrink": 0.7, "aspect": 25,
                          "label": "Feature Rank\n(1 = most important, 13 = least)"}, ax=ax)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            v = int(ranks.iat[i, j])
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if (v <= 3 or v >= 11) else "black")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11, rotation=0, length=0)
    ax.tick_params(axis="y", labelsize=12, rotation=0, length=0)
    ax.figure.axes[-1].tick_params(labelsize=13)
    band = 0.03583  # manuscript crop: top 3.583% of the image height
    fig.text(0.5, 1 - band / 2, "Figure 6. Cross-Pipeline Feature Importance Rank Heatmap",
             ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 1 - band - 0.022, "5 pipelines · 13 features · common 1,000-record sample · "
             "green = important · red = unimportant", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.085)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
