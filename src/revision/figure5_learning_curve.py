#!/usr/bin/env python3
"""Manuscript Figure 5: Random Forest learning curve, regenerated from the Amendment 7 re-execution.

Reads revision-results/lowcompute/rf_learning_curve.csv (committed with Amendment 7). Figure only, no new
analysis. Style matches the submitted image (word/media/image5.png of the 17 Oct manuscript): matplotlib
defaults, 2070 x 1466 px at 300 dpi, mean AUC with SD error bars and caps, light grid, same axis labels.
No source code for the submitted figure exists, so the style was matched from the image. Changes from the
submitted figure: the data, and the title, because at the full training size the re-execution is a single
fit (SD 0 by construction), so "five subsamples per size" no longer holds for every point.

Run from the repository root:
    python src/revision/figure5_learning_curve.py
Writes: results/figures/main/Figure_5_RF_Learning_Curve.png
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision-results" / "lowcompute" / "rf_learning_curve.csv"
OUT = ROOT / "results" / "figures" / "main" / "Figure_5_RF_Learning_Curve.png"


def main() -> None:
    lc = pd.read_csv(SRC)
    fig, ax = plt.subplots(figsize=(2070 / 300, 1466 / 300))
    ax.errorbar(lc["n_train_rows"], lc["mean_auc"], yerr=lc["sd_auc_ddof1"], fmt="-o", capsize=4, linewidth=1.5,
                markersize=7, elinewidth=1.5)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Held-out test ROC-AUC")
    ax.set_title("Random Forest learning curve (re-executed; one fit at full size)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
