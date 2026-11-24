#!/usr/bin/env python3
"""Protocol Amendment 6 (POST HOC): retraining as a variance stabiliser (manuscript §5.6).

POST HOC. Specified after the five-seed results (Amendment 6, committed and pushed alone in
9a1b366 before this file was added). NOT prespecified. Descriptive only; no stop condition;
no interval estimates (Amendment 5). No mechanism is attributed; the effect is attributed to
retraining as a whole, not to early stopping.

Prompt: the Section 5 between-seed AUC SD (0.0008-0.0017) was about three times smaller than the
per-seed bootstrap SE (~0.0044) implied.

Diagnostic (no retraining; reads saved arrays, saved seed-42 models and the stage-1 CSV):
  Split check   the five seeds' test sets are distinct (pairwise overlap ~0.15), each saved split equals
                the split recomputed from its seed, and every Section 5 AUC reproduces on its own y_test.
  (i)   Reference: analytic Hanley-McNeil (1982) SE of AUC at n = 10,281 (5,083 positive, 5,198
        negative; positive rate 0.49441), AUC = 0.80. No model, no simulation. Cross-check (labelled as
        such): the DeLong SE from each seed-42 model's predictions (fast_delong, src/run_validated_stats.py).
  (ii)  Corroboration: SD of a FIXED seed-42 model's AUC over 200 test sets drawn by the protocol's first
        split step, train_test_split(idx, test_size=0.15, stratify=y, random_state=k), k = 1000..1199,
        for the seed-42 XGBoost (artifacts/models-revision/xgb_seed42.json) and Random Forest
        (rf_seed42.joblib) models. LEAKAGE: ~70% of each drawn test set was in that model's training data,
        so the AUC LEVEL is inflated; only the SD is interpretable.
  (iii) Contrast: observed between-seed SD per model from Section 5 (models retrained at each split).
  (iv)  Ratio (iii)/(i). (iii) rests on five values, so it carries about 1/sqrt(2*(5-1)) = 35% relative
        uncertainty; the ratio is indicative, not precise.
  Partial cross-evaluation  Section 5 saved predictions, not model objects, so only seed-42 models exist:
        XGBoost, RF and SAINT-inspired (identical to Section 5's seed-42 models) and the deposited original
        FT-Transformer (not Section 5's retrain). Each is scored on all five seeds' test sets (s = 42 row only;
        same leakage caveat; only the SD across t is of interest). CatBoost and stacking have no saved model.
  Capacity table  early-stopping counts and best epochs per seed against that seed's mean AUC over the six
        ranked models. Data only; no relationship is fitted (five seeds).

Run from the repository root (original Anaconda environment; torch for FT/SAINT inference):
    python src/revision/section5_posthoc_amendment6.py --data-file <stage-1 CSV>
Writes: revision-results/five_seed/five_seed_posthoc_amendment6.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np  # numpy / scikit-learn before torch (duplicate-OpenMP issue in this environment)
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "five_seed"
SEEDS = [42, 101, 202, 303, 404]
DRAW_SEEDS = list(range(1000, 1200))   # 200 test-set draws for (ii); recorded so the result reproduces
RANKED = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]
SUF = {"Stacking Ensemble": "stack", "XGBoost": "xgb", "CatBoost": "cat", "FT-Transformer": "ft",
       "SAINT-inspired": "saint", "Random Forest": "rf"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


s3 = _load("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import fast_delong  # noqa: E402  (unchanged)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg))


def delong_se(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-y)
    _, cov = fast_delong(p[order][None, :], int(y.sum()))
    return float(math.sqrt(cov[0, 0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    a = ap.parse_args()

    import xgboost as xgb
    import torch
    import torch.nn.functional as F
    pipes = _load("pipes", ROOT / "src" / "revision" / "section4_2_pipelines.py")

    X, y = s3.build_cohort(a.data_file); N = len(y); idx = np.arange(N)
    man = {r["run"]: r for r in json.loads((OUT / "five_seed_run_manifest.json").read_text(encoding="utf-8"))["runs"]}
    long = pd.read_csv(OUT / "five_seed_auc_long.csv"); long = long[long["model"].isin(RANKED)]
    A = long.pivot(index="model", columns="seed", values="auc").loc[RANKED, SEEDS]

    # ---------------- split check
    tests, split_rows = {}, []
    for s in SEEDS:
        sp = np.load(OUT / f"split_indices_seed{s}.npz"); te = sp["test"]
        _, _, _, _, i_tmp, i_te = train_test_split(X, y, idx, test_size=0.15, random_state=s, stratify=y)
        _, _, _, _, i_tr, i_va = train_test_split(X[i_tmp], y[i_tmp], i_tmp, test_size=0.1765, random_state=s, stratify=y[i_tmp])
        yt = np.load(OUT / f"y_test_seed{s}.npy")
        tests[s] = te
        split_rows.append({
            "seed": s, "n_test": int(len(te)), "positive_rate": float(yt.mean()),
            "saved_split_equals_recomputed_from_seed": bool(np.array_equal(te, i_te) and np.array_equal(sp["train"], i_tr) and np.array_equal(sp["val"], i_va)),
            "y_test_file_equals_y_at_test_indices": bool(np.array_equal(yt, y[te])),
            "section5_aucs_reproduce_on_own_y_test": bool(all(
                abs(roc_auc_score(yt, np.load(OUT / f"proba_{SUF[m]}_seed{s}.npy")) - man[f"seed{s}"]["models"][m]["test_auc"]) < 1e-12 for m in RANKED)),
            "sha256_sorted_test_indices": hashlib.sha256(np.sort(te).astype(np.int64).tobytes()).hexdigest()})
    overlap = {f"{s}-{t}": float(len(set(tests[s].tolist()) & set(tests[t].tolist())) / len(tests[s]))
               for s, t in itertools.combinations(SEEDS, 2)}

    # ---------------- (i) analytic reference
    y42 = np.load(OUT / "y_test_seed42.npy"); n_pos = int(y42.sum()); n_neg = len(y42) - n_pos
    se_hm = hanley_mcneil_se(0.80, n_pos, n_neg)
    se_delong = {m: delong_se(y42, np.load(OUT / f"proba_{SUF[m]}_seed42.npy").astype(float)) for m in RANKED}

    # ---------------- seed-42 models (no retraining)
    sc42 = joblib.load(ROOT / "artifacts" / "models-revision" / "scaler_seed42.joblib")
    xg = xgb.XGBClassifier(); xg.load_model(str(ROOT / "artifacts" / "models-revision" / "xgb_seed42.json"))
    rf = joblib.load(ROOT / "artifacts" / "models-revision" / "rf_seed42.joblib")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft, _ = pipes.ft_model(); ft.to(dev).eval()
    sa, sack = pipes.saint_model(); sa.to(dev).eval()

    def p_ft(Xr):
        with torch.no_grad():
            return F.softmax(ft(torch.tensor(sc42.transform(Xr).astype(np.float32), device=dev)), 1)[:, 1].cpu().numpy()

    def p_saint(Xr):
        Xs = sack["scaler"].transform(Xr.astype(np.float32)).astype(np.float32)
        with torch.no_grad():
            return np.concatenate([F.softmax(sa(torch.tensor(Xs[i:i + 256], device=dev)), 1)[:, 1].cpu().numpy()
                                   for i in range(0, len(Xs), 256)])

    fixed = {"XGBoost": lambda Xr: xg.predict_proba(Xr)[:, 1],
             "Random Forest": lambda Xr: rf.predict_proba(sc42.transform(Xr))[:, 1],
             "SAINT-inspired": p_saint, "FT-Transformer (deposited original)": p_ft}

    # ---------------- partial cross-evaluation (s = 42 row)
    tr42 = np.load(OUT / "split_indices_seed42.npz")["train"]; in_tr = np.zeros(N, bool); in_tr[tr42] = True
    cross = {}
    for name, fn in fixed.items():
        row = {str(t): float(roc_auc_score(y[tests[t]], fn(X[tests[t]]))) for t in SEEDS}
        off = np.array([row[str(t)] for t in SEEDS if t != 42])
        cross[name] = {"auc_by_test_set": row, "diagonal_auc_42_42": row["42"],
                       "sd_over_t_ne_42_ddof1": float(off.std(ddof=1)), "n_off_diagonal": int(len(off))}
    leak_share = {str(t): float(in_tr[tests[t]].mean()) for t in SEEDS}

    # ---------------- (ii) 200 draws, fixed seed-42 XGBoost and RF
    corr = {}
    for name in ("XGBoost", "Random Forest"):
        p_all = fixed[name](X)                                    # every cohort row once
        aucs = []
        for k in DRAW_SEEDS:
            _, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=k)
            aucs.append(roc_auc_score(y[te], p_all[te]))
        aucs = np.array(aucs)
        corr[name] = {"sd_over_200_draws_ddof1": float(aucs.std(ddof=1)), "mean_auc_over_draws_LEAKY_LEVEL": float(aucs.mean()),
                      "draw_random_states": f"{DRAW_SEEDS[0]}..{DRAW_SEEDS[-1]}", "n_draws": len(DRAW_SEEDS)}

    # ---------------- (iii) and (iv)
    between = A.std(axis=1, ddof=1)
    contrast = {m: {"between_seed_sd_ddof1": float(between[m]), "ratio_to_hanley_mcneil_se": float(between[m] / se_hm)} for m in RANKED}

    # ---------------- capacity table (data only)
    cap = []
    for s in SEEDS:
        md = man[f"seed{s}"]["models"]
        cap.append({"seed": s, "mean_auc_six_models": float(A[s].mean()),
                    "xgboost_best_iteration": md["XGBoost"]["best_iteration"],
                    "lightgbm_best_iteration": md["LightGBM (standalone)"]["best_iteration"],
                    "catboost_best_iteration": md["CatBoost"]["best_iteration"],
                    "stacking_trees_xgb_lgb_cat": "/".join(str(v) for v in md["Stacking Ensemble"]["trees"].values()),
                    "ft_best_epoch": md["FT-Transformer"]["best_epoch"], "ft_epochs_run": md["FT-Transformer"]["epochs_run"],
                    "saint_best_epoch": md["SAINT-inspired"]["best_epoch"], "saint_epochs_run": md["SAINT-inspired"]["epochs_run"]})

    out = {
        "protocol_section": "Amendment 6 (POST HOC) — not prespecified; descriptive only; no interval estimates; no mechanism attributed",
        "amendment_commit": "9a1b366ab02ada3534dc2fb5c5ae2ffe7904dd23",
        "split_check": {"per_seed": split_rows, "pairwise_test_overlap_fraction": overlap},
        "i_reference_hanley_mcneil_se": {"auc": 0.80, "n_test": int(len(y42)), "n_positive": n_pos, "n_negative": n_neg,
                                         "positive_rate": float(y42.mean()), "se": se_hm},
        "i_crosscheck_delong_se_seed42_predictions": se_delong,
        "ii_corroboration_fixed_model_200_draws": {
            **corr, "leakage_caveat": "about 70% of each drawn test set was in the seed-42 model's training data; the AUC LEVEL is inflated and only the SD is interpretable"},
        "iii_contrast_between_seed_sd_retrained": contrast,
        "iv_note": "(iii) rests on five values: relative uncertainty about 1/sqrt(2*(5-1)) = 0.354; the ratio is indicative, not precise",
        "partial_cross_evaluation_s42_row": {
            "models": cross, "share_of_each_test_set_in_seed42_training_data": leak_share,
            "caveat": "Section 5 saved predictions, not model objects; only seed-42 models exist, so only the s = 42 row is available. "
                      "Off-diagonal cells leak (~70% of each test set was seed-42 training data): LEVEL inflated; only the SD across t is "
                      "of interest, and with four off-diagonal values it is imprecise. CatBoost and stacking have no saved model. "
                      "The FT-Transformer is the deposited original, not Section 5's seed-42 retrain."},
        "capacity_table_data_only": cap,
    }
    with open(OUT / "five_seed_posthoc_amendment6.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    print(f"split check: all saved==recomputed {all(r['saved_split_equals_recomputed_from_seed'] for r in split_rows)}; "
          f"all AUCs reproduce {all(r['section5_aucs_reproduce_on_own_y_test'] for r in split_rows)}; "
          f"overlap {min(overlap.values()):.4f}-{max(overlap.values()):.4f}")
    print(f"(i)   Hanley-McNeil SE (AUC 0.80, {n_pos}/{n_neg}): {se_hm:.5f} | DeLong cross-check: " +
          ", ".join(f"{m} {v:.5f}" for m, v in se_delong.items()))
    print("(ii)  fixed-model SD over 200 draws: " + ", ".join(f"{m} {v['sd_over_200_draws_ddof1']:.5f}" for m, v in corr.items()))
    print("(iii) between-seed SD (retrained): " + ", ".join(f"{m} {v['between_seed_sd_ddof1']:.5f}" for m, v in contrast.items()))
    print("(iv)  ratio (iii)/(i): " + ", ".join(f"{m} {v['ratio_to_hanley_mcneil_se']:.3f}" for m, v in contrast.items()))
    print("partial cross-eval SD over t!=42: " + ", ".join(f"{m} {v['sd_over_t_ne_42_ddof1']:.5f}" for m, v in cross.items()))


if __name__ == "__main__":
    main()
