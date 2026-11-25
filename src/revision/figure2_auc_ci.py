#!/usr/bin/env python3
"""Figure 2: test-set ROC-AUC of the 15 models with 95% bootstrap confidence intervals, from deposited output only.

Inputs (nothing else is read):
  results/tables/validated_primary_metrics.csv        point AUC per model (and the plotting order)
  revision-results/lowcompute/bootstrap_auc_by_model.csv
                                                       95% percentile bootstrap interval per model
                                                       (Protocol Section 3A(b): 2,000 resamples, seed 42)
The two files must agree on every model's AUC (to 1e-12) or the script stops.

Before this script there was no deposited code for Figure 2: the submitted figure was drawn from 1,000-resample
intervals that survive only as printed notebook output (and, for the corrected stacking ensemble, as README text).
This script therefore plots the deposited 2,000-resample intervals; see docs/CODE_MANUSCRIPT_DISCREPANCIES.md item 10.

Drawn at the journal's double-column width (190 mm); no in-image title (the caption carries it).

Run from the repository root:
    python src/revision/figure2_auc_ci.py
Writes: results/figures/main/Figure_2_AUC_95CI.pdf   (vector)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "tables" / "validated_primary_metrics.csv"
BOOT = ROOT / "revision-results" / "lowcompute" / "bootstrap_auc_by_model.csv"
OUT = ROOT / "results" / "figures" / "main" / "Figure_2_AUC_95CI.pdf"
WIDTH_IN = 190 / 25.4


def main() -> None:
    m = pd.read_csv(METRICS)
    b = pd.read_csv(BOOT)
    m["Model"] = m["Model"].str.strip()
    b["model"] = b["model"].str.strip()
    df = m[["Model", "AUC"]].merge(b[["model", "auc", "auc_ci_low", "auc_ci_high", "bootstrap_resamples", "bootstrap_seed"]],
                                   left_on="Model", right_on="model", how="outer", validate="one_to_one")
    if df.isna().any().any() or len(df) != 15:
        raise SystemExit(f"model sets differ between the two inputs:\n{df}")
    if not np.allclose(df["AUC"], df["auc"], rtol=0, atol=1e-12):
        raise SystemExit("point AUCs differ between validated_primary_metrics.csv and bootstrap_auc_by_model.csv")
    df = df.sort_values("AUC", ascending=False).reset_index(drop=True)  # highest AUC at the top

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.80 * WIDTH_IN))
    y = np.arange(len(df))[::-1]
    lo = df["AUC"] - df["auc_ci_low"]
    hi = df["auc_ci_high"] - df["AUC"]
    eb = ax.errorbar(df["AUC"], y, xerr=[lo, hi], fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.4,
                     capsize=3, markersize=5)
    ax.axvline(0.80, linestyle="--", linewidth=1, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.set_xlim(0.72, 0.815)
    ax.set_xticks([0.72, 0.74, 0.76, 0.78, 0.80])
    ax.set_xlabel(f"ROC-AUC (95% bootstrap CI; {int(df['bootstrap_resamples'].iloc[0]):,} resamples)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    # the drawn values must be the deposited values, exactly
    xd, yd = eb.lines[0].get_data()
    assert np.array_equal(np.asarray(xd), df["AUC"].to_numpy())
    segs = eb.lines[2][0].get_segments()
    assert np.allclose([s[0][0] for s in segs], df["auc_ci_low"], rtol=0, atol=1e-12)
    assert np.allclose([s[1][0] for s in segs], df["auc_ci_high"], rtol=0, atol=1e-12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    print(f"wrote {OUT.relative_to(ROOT)}")
    for r in df.itertuples():
        print(f"  {r.Model:20s} {r.AUC:.6f} [{r.auc_ci_low:.6f}, {r.auc_ci_high:.6f}]")


if __name__ == "__main__":
    main()
