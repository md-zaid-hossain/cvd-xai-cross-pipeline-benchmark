#!/usr/bin/env python3
"""Protocol Section 3A(d): decision-curve analysis.

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(d). Deposited test
probabilities for all 15 models at the observed test prevalence pi_s.
Threshold grid: np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2) (91 values).
A record is treated as positive when proba >= p_t.
Net benefit (Vickers & Elkin, 2006): NB = TP/n - (FP/n) * p_t / (1 - p_t).
Treat-all: pi_s - (1 - pi_s) * p_t / (1 - p_t). Treat-none: 0.
Point estimates only (no CIs), as prespecified.

"Exceeds both reference strategies" is taken strictly: NB > treat-all
and NB > 0 at that threshold. Ranges are reported as maximal runs of
consecutive grid thresholds.

Run from the repository root:
    python src/revision/section3A_d_decision_curve.py
Writes:
    revision-results/lowcompute/decision_curve_net_benefit.csv
    revision-results/lowcompute/decision_curve_summary.csv
    revision-results/lowcompute/figure_decision_curve_top6.png
"""
from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import MODELS  # noqa: E402

TOP_SIX = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]
PT_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)


def net_benefit(y: np.ndarray, proba: np.ndarray, pt: float) -> float:
    n = len(y)
    pos = proba >= pt
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return tp / n - (fp / n) * pt / (1 - pt)


def runs(mask: np.ndarray, grid: np.ndarray) -> str:
    """Maximal runs of consecutive True grid points, e.g. '0.05-0.62; 0.70-0.71'."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            out.append(f"{grid[start]:.2f}-{grid[end]:.2f}")
            start = None
    return "; ".join(out) if out else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(args.predictions_dir / "y_test.npy").astype(int)
    pi_s = float(y.mean())
    proba = {m: np.load(args.predictions_dir / f"proba_{s}.npy").astype(float).reshape(-1)
             for m, s in MODELS.items()}

    treat_all = np.array([pi_s - (1 - pi_s) * pt / (1 - pt) for pt in PT_GRID])
    nb = {m: np.array([net_benefit(y, p, pt) for pt in PT_GRID]) for m, p in proba.items()}

    rows = []
    for m in MODELS:
        rows += [{"strategy": m, "threshold": pt, "net_benefit": v} for pt, v in zip(PT_GRID, nb[m])]
    rows += [{"strategy": "treat-all", "threshold": pt, "net_benefit": v} for pt, v in zip(PT_GRID, treat_all)]
    rows += [{"strategy": "treat-none", "threshold": pt, "net_benefit": 0.0} for pt in PT_GRID]
    pd.DataFrame(rows).to_csv(args.output_dir / "decision_curve_net_benefit.csv", index=False, lineterminator="\n")

    top_matrix = np.vstack([nb[m] for m in TOP_SIX])
    top_spread = top_matrix.max(axis=0) - top_matrix.min(axis=0)
    i_max = int(np.argmax(top_spread))

    summary = []
    for m in MODELS:
        beats = (nb[m] > treat_all) & (nb[m] > 0)
        summary.append({
            "model": m, "in_top_six": m in TOP_SIX,
            "thresholds_nb_exceeds_both": runs(beats, PT_GRID),
            "n_thresholds_exceeding_both": int(beats.sum()),
            "max_net_benefit": float(nb[m].max()),
            "nb_at_0.10": float(nb[m][np.isclose(PT_GRID, 0.10)][0]),
            "nb_at_0.20": float(nb[m][np.isclose(PT_GRID, 0.20)][0]),
            "nb_at_0.50": float(nb[m][np.isclose(PT_GRID, 0.50)][0]),
        })
    summ = pd.DataFrame(summary)
    summ["max_top_six_nb_difference"] = float(top_spread.max())
    summ["threshold_of_max_top_six_nb_difference"] = float(PT_GRID[i_max])
    summ["test_prevalence_pi_s"] = pi_s
    summ.to_csv(args.output_dir / "decision_curve_summary.csv", index=False, lineterminator="\n")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in TOP_SIX:
        ax.plot(PT_GRID, nb[m], lw=1.4, label=m)
    ax.plot(PT_GRID, treat_all, "k--", lw=1.2, label="Treat all")
    ax.axhline(0, color="grey", lw=1.2, ls=":", label="Treat none")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.05, max(0.55, float(top_matrix.max()) + 0.03))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curves, six highest-AUC models (test prevalence {pi_s:.4f})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_decision_curve_top6.png", dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"pi_s = {pi_s:.6f}; thresholds = {len(PT_GRID)} ({PT_GRID[0]}..{PT_GRID[-1]})")
    print(summ[["model", "thresholds_nb_exceeds_both", "n_thresholds_exceeding_both",
                "max_net_benefit", "nb_at_0.10", "nb_at_0.20", "nb_at_0.50"]].to_string(index=False))
    print(f"\nmax NB difference among top six: {top_spread.max():.6f} at p_t = {PT_GRID[i_max]:.2f}")
    print(f"mean NB difference among top six across grid: {top_spread.mean():.6f}")


if __name__ == "__main__":
    main()
