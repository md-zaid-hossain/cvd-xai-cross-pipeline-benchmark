#!/usr/bin/env python3
"""Protocol Amendment 5 (POST HOC): replacement for the ill-posed Outcome A comparison.

POST HOC. These quantities were specified after the five-seed results were produced
(Amendment 5, committed and pushed alone in dde58e6 before this script ran). They are
NOT prespecified. Descriptive only; no interval estimates; no stop condition.

Input: revision-results/five_seed/five_seed_auc_long.csv (six ranked models x five seeds).
Notation: A[m, s] = test AUC of model m at seed s; M = 6 models, S = 5 seeds.

  (i)   between-seed SD per model: SD over seeds of A[m, .] (ddof = 1); mean over models
  (ii)  within-seed SD per seed:   SD over models of A[., s] (ddof = 1); mean over seeds
  (iii) ratio mean(i) / mean(ii)
  (iv)  the same with ranges (max - min) instead of SDs
  (v)   two-way additive main-effect decomposition A = mu + a_m + b_s + e:
          a_m = row mean - mu, b_s = column mean - mu, e = residual
          SS_model = S * sum(a_m^2), SS_seed = M * sum(b_s^2), SS_res = sum(e^2),
          SS_total = sum((A - mu)^2) = SS_model + SS_seed + SS_res (checked)
        Effect SDs are reported under two stated definitions (the amendment does not fix one):
          sd_ddof1 = SD of the effects with ddof = 1 (a_m over M; b_s over S)
          sd_ddof0 = root mean square of the effects (ddof = 0)
        Residual SD: sqrt(SS_res / ((M-1)(S-1))) (two-way ANOVA residual df) and the RMS of e.
  (vi)  per-model rank vector across seeds, mean rank, rank range
  plus  Kendall's W (tie-corrected) of the model rankings across seeds, chi2 = S(M-1)W, df = M-1.

Run from the repository root:
    python src/revision/section5_posthoc_amendment5.py
Writes: revision-results/five_seed/five_seed_posthoc_amendment5.json
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "five_seed"
RANKED = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]
SEEDS = [42, 101, 202, 303, 404]


def main() -> None:
    long = pd.read_csv(OUT / "five_seed_auc_long.csv")
    long = long[long["model"].isin(RANKED)]
    A = long.pivot(index="model", columns="seed", values="auc").loc[RANKED, SEEDS]
    R = long.pivot(index="model", columns="seed", values="rank").loc[RANKED, SEEDS]
    X = A.values
    M, S = X.shape

    between_sd = A.std(axis=1, ddof=1)          # (i) per model, over seeds
    within_sd = A.std(axis=0, ddof=1)           # (ii) per seed, over models
    between_rng = A.max(axis=1) - A.min(axis=1)
    within_rng = A.max(axis=0) - A.min(axis=0)

    mu = X.mean()
    a = X.mean(axis=1) - mu                      # model effects
    b = X.mean(axis=0) - mu                      # seed effects
    e = X - mu - a[:, None] - b[None, :]
    ss_model, ss_seed, ss_res = S * (a ** 2).sum(), M * (b ** 2).sum(), (e ** 2).sum()
    ss_total = ((X - mu) ** 2).sum()

    n, m = R.values.shape
    Ssum = float(((R.values.sum(axis=1) - m * (n + 1) / 2) ** 2).sum())
    T = sum(float((c ** 3 - c).sum()) for c in (np.unique(R.values[:, j], return_counts=True)[1] for j in range(m)))
    W = 12 * Ssum / (m ** 2 * (n ** 3 - n) - m * T)
    chi2 = m * (n - 1) * W

    out = {
        "protocol_section": "Amendment 5 (POST HOC) — not prespecified; descriptive only; no interval estimates",
        "amendment_commit": "dde58e609672f77f07479ae8f68bc311561c3614",
        "auc_matrix": {mod: {str(s): float(A.loc[mod, s]) for s in SEEDS} for mod in RANKED},
        "i_between_seed_sd_ddof1": {"per_model": between_sd.to_dict(), "mean_over_models": float(between_sd.mean()),
                                    "range": [float(between_sd.min()), float(between_sd.max())]},
        "ii_within_seed_sd_ddof1": {"per_seed": {str(k): float(v) for k, v in within_sd.items()},
                                    "mean_over_seeds": float(within_sd.mean()),
                                    "range": [float(within_sd.min()), float(within_sd.max())]},
        "iii_ratio_mean_between_sd_to_mean_within_sd": float(between_sd.mean() / within_sd.mean()),
        "iv_ranges": {"between_seed_range_per_model": between_rng.to_dict(), "mean_between_seed_range": float(between_rng.mean()),
                      "within_seed_range_per_seed": {str(k): float(v) for k, v in within_rng.items()},
                      "mean_within_seed_range": float(within_rng.mean()),
                      "ratio": float(between_rng.mean() / within_rng.mean())},
        "v_two_way_main_effects": {
            "grand_mean": float(mu),
            "seed_effects": dict(zip(map(str, SEEDS), b.tolist())), "model_effects": dict(zip(RANKED, a.tolist())),
            "seed_effect_sd_ddof1": float(b.std(ddof=1)), "model_effect_sd_ddof1": float(a.std(ddof=1)),
            "seed_effect_sd_ddof0_rms": float(np.sqrt((b ** 2).mean())), "model_effect_sd_ddof0_rms": float(np.sqrt((a ** 2).mean())),
            "seed_to_model_effect_sd_ratio_ddof1": float(b.std(ddof=1) / a.std(ddof=1)),
            "residual_sd_anova_df_(M-1)(S-1)": float(np.sqrt(ss_res / ((M - 1) * (S - 1)))),
            "residual_rms_ddof0": float(np.sqrt((e ** 2).mean())),
            "ss": {"seed": float(ss_seed), "model": float(ss_model), "residual": float(ss_res), "total": float(ss_total),
                   "additivity_check_abs_error": float(abs(ss_total - ss_seed - ss_model - ss_res))},
            "ss_share": {"seed": float(ss_seed / ss_total), "model": float(ss_model / ss_total), "residual": float(ss_res / ss_total)},
        },
        "vi_ranks": {mod: {"ranks_by_seed": {str(s): float(R.loc[mod, s]) for s in SEEDS},
                           "mean_rank": float(R.loc[mod].mean()), "rank_range": [float(R.loc[mod].min()), float(R.loc[mod].max())]}
                     for mod in RANKED},
        "kendall_w_across_seeds": {"W": W, "chi2": chi2, "df": n - 1, "p": float(stats.chi2.sf(chi2, n - 1)),
                                   "tie_correction_sum_T": T,
                                   "null_tested": ("H0: the five seeds' rankings of the six models are mutually independent "
                                                   "(no agreement beyond chance). Rejecting it says the rankings agree more than "
                                                   "random orderings would; it does not test whether any model is better, "
                                                   "nor whether the leader is stable.")},
    }
    with open(OUT / "five_seed_posthoc_amendment5.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    v = out["v_two_way_main_effects"]
    print(f"(i)   between-seed SD: mean {out['i_between_seed_sd_ddof1']['mean_over_models']:.5f}  range {between_sd.min():.5f}-{between_sd.max():.5f}")
    print(f"(ii)  within-seed SD:  mean {out['ii_within_seed_sd_ddof1']['mean_over_seeds']:.5f}  range {within_sd.min():.5f}-{within_sd.max():.5f}")
    print(f"(iii) ratio {out['iii_ratio_mean_between_sd_to_mean_within_sd']:.4f}")
    print(f"(iv)  mean between-seed range {between_rng.mean():.5f}; mean within-seed range {within_rng.mean():.5f}; ratio {out['iv_ranges']['ratio']:.4f}")
    print(f"(v)   seed-effect SD ddof1 {v['seed_effect_sd_ddof1']:.5f} (rms {v['seed_effect_sd_ddof0_rms']:.5f}); "
          f"model-effect SD ddof1 {v['model_effect_sd_ddof1']:.5f} (rms {v['model_effect_sd_ddof0_rms']:.5f}); "
          f"ratio {v['seed_to_model_effect_sd_ratio_ddof1']:.4f}")
    print(f"      residual SD (df 20) {v['residual_sd_anova_df_(M-1)(S-1)']:.5f}; residual rms {v['residual_rms_ddof0']:.5f}")
    print(f"      SS shares: seed {v['ss_share']['seed']:.4f}, model {v['ss_share']['model']:.4f}, residual {v['ss_share']['residual']:.4f}; additivity error {v['ss']['additivity_check_abs_error']:.2e}")
    for mod in RANKED:
        r = out["vi_ranks"][mod]; print(f"(vi)  {mod:18s} ranks {list(r['ranks_by_seed'].values())} mean {r['mean_rank']:.1f} range {r['rank_range']}")
    k = out["kendall_w_across_seeds"]; print(f"Kendall W {k['W']:.4f}, chi2({k['df']}) = {k['chi2']:.3f}, p = {k['p']:.4g}")


if __name__ == "__main__":
    main()
