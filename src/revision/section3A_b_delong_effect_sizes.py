#!/usr/bin/env python3
"""Protocol Section 3A(b): DeLong effect sizes with paired bootstrap CIs.

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(b) with the Section 3A
conventions. Families, DeLong and BH are taken unchanged from
src/run_validated_stats.py. Bootstrap: 2,000 paired resamples,
numpy default_rng(42), sequential draws, one index vector per replicate
shared by every model and pair; percentile 95% CI (numpy linear).

Run from the repository root:
    python src/revision/section3A_b_delong_effect_sizes.py

Writes the Section 7 outputs for 3A(b):
    revision-results/lowcompute/bootstrap_auc_by_model.csv
    revision-results/lowcompute/delong_effect_sizes_top6.csv
    revision-results/lowcompute/delong_effect_sizes_vs_mlp.csv
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys

sys.dont_write_bytecode = True  # keep src/ free of __pycache__ (it would enter the manifest)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import MODELS, delong_test, fdr_bh  # noqa: E402  (unchanged, per Section 3A)

N_BOOT = 2000
BOOT_SEED = 42
Z95 = 1.959964
TOP_SIX_EXPECTED = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer",
                    "SAINT-inspired", "Random Forest"]
VS_MLP = ["XGBoost", "LightGBM", "CatBoost", "Stacking Ensemble", "TabNet",
          "FT-Transformer", "VIME-inspired SSL", "SAINT-inspired", "Random Forest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(args.predictions_dir / "y_test.npy").astype(int)
    proba = {name: np.load(args.predictions_dir / f"proba_{s}.npy").astype(float).reshape(-1)
             for name, s in MODELS.items()}
    auc = {name: roc_auc_score(y, p) for name, p in proba.items()}

    ranked = sorted(MODELS, key=lambda m: auc[m], reverse=True)
    top_six = ranked[:6]
    if top_six != TOP_SIX_EXPECTED:
        raise SystemExit(f"top six {top_six} differs from the protocol's {TOP_SIX_EXPECTED}")

    # Paired bootstrap (Section 3A(b) code block).
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    boot = {name: [] for name in MODELS}
    discarded = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            discarded += 1
            continue
        for name, p in proba.items():
            boot[name].append(roc_auc_score(yb, p[idx]))
    boot = {k: np.asarray(v) for k, v in boot.items()}

    per_model = []
    for name in MODELS:
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        per_model.append({"model": name, "auc": auc[name], "auc_ci_low": lo, "auc_ci_high": hi,
                          "primary_rank": ranked.index(name) + 1,
                          "bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED,
                          "discarded_single_class": discarded})
    pd.DataFrame(per_model).to_csv(args.output_dir / "bootstrap_auc_by_model.csv",
                                   index=False, lineterminator="\n")

    def pair_row(a: str, b: str) -> dict:
        auc_a, auc_b, z, p, var = delong_test(y, proba[a], proba[b])
        delta = auc_a - auc_b
        se = math.sqrt(var) if var > 0 else float("nan")
        blo, bhi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
        return {"model_a": a, "model_b": b, "auc_a": auc_a, "auc_b": auc_b, "delta_auc": delta,
                "bootstrap_ci_low": blo, "bootstrap_ci_high": bhi,
                "bootstrap_ci_includes_0": bool(blo <= 0 <= bhi),
                "delong_ci_low": delta - Z95 * se, "delong_ci_high": delta + Z95 * se,
                "delong_se": se, "z": z, "p": p}

    # Top-six family: itertools.combinations over AUC-ranked top six, as run_validated_stats.py.
    top = pd.DataFrame([pair_row(a, b) for a, b in itertools.combinations(top_six, 2)])
    top["bh_q"] = fdr_bh(top["p"])
    top["bh_significant_0.05"] = top["bh_q"] < 0.05
    top.to_csv(args.output_dir / "delong_effect_sizes_top6.csv", index=False, lineterminator="\n")

    # vs-MLP family: order and membership as run_validated_stats.py; Model B is always MLP.
    mlp = pd.DataFrame([pair_row(a, "MLP") for a in VS_MLP])
    mlp["bh_q"] = fdr_bh(mlp["p"])
    mlp["bh_significant_0.05"] = mlp["bh_q"] < 0.05
    mlp.to_csv(args.output_dir / "delong_effect_sizes_vs_mlp.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}, discarded single-class: {discarded}")
    print(pd.DataFrame(per_model)[["model", "auc", "auc_ci_low", "auc_ci_high", "primary_rank"]].to_string(index=False))
    cols = ["model_a", "model_b", "delta_auc", "bootstrap_ci_low", "bootstrap_ci_high",
            "delong_ci_low", "delong_ci_high", "z", "p", "bh_q"]
    print("\nTOP-SIX FAMILY"); print(top[cols].to_string(index=False))
    print("\nVS-MLP FAMILY"); print(mlp[cols].to_string(index=False))


if __name__ == "__main__":
    main()
