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
Revision (2026-11-26): no in-image title or subtitle (the manuscript caption carries both), the fifth column is
labelled "SAINT-inspired Permutation", and a vector PDF is written as the submitted figure. Earlier versions drew
the two title lines and the label "SAINT Permutation", which the submitted figure does not carry.

Run from the repository root:
    python src/revision/figure6_rank_heatmap.py
Writes: results/figures/main/Figure_6_Cross_Pipeline_Rank_Heatmap.pdf  (vector; the submitted Figure 6)
        results/figures/main/Figure_6_Cross_Pipeline_Rank_Heatmap.png  (300 dpi raster of the same figure)
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
OUT_PDF = OUT.with_suffix(".pdf")
PDF_META = {"Creator": None, "Producer": None, "CreationDate": None}
COLUMNS = {"MLP Perm.": "MLP\nPermutation", "XGBoost SHAP": "XGBoost\nSHAP", "TabNet Attn.": "TabNet\nAttention",
           "FT-Transformer SHAP": "FT-Transformer\nSHAP", "SAINT-inspired Perm.": "SAINT-inspired\nPermutation"}


def main() -> None:
    df = pd.read_csv(SRC).set_index("feature")
    ranks = df[list(COLUMNS)].rename(columns=COLUMNS)  # row order as in the CSV (mean-rank order)
    if not ((ranks.values >= 1) & (ranks.values <= 13)).all():
        raise SystemExit("ranks outside 1-13")

    # Canvas of the submitted image (1100 x 886 px aspect); the saved files are cropped to the drawing.
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
    fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.085)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=PDF_META)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)} and {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
