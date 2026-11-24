#!/usr/bin/env python3
"""Protocol Amendment 10 (prespecified, pushed alone in 8867e05): external validation on NHANES.

Every mapping is fixed by the amendment and its run specification; nothing here is tuned on NHANES.

Cohorts   primary = NHANES 2013-14 + 2015-16 (auscultatory BPX); secondary = 2017-March 2020 (oscillometric BPXO),
          reported separately, never pooled. Age 40-65 inclusive. No survey weights.
Mapping   BP = mean of non-missing readings 1-3; a record with DBP = 0 on ANY reading is excluded. Sex RIAGENDR 1 -> 1,
          2 -> 0. BMI = BMXBMI. Cleaning as the primary cohort: SBP in [80, 200], DBP in [40, 130]; BMI in [10, 60],
          SBP - DBP > 0. Cholesterol LBXTC <200 / 200-239 / >=240 -> 1/2/3. Glucose LBXSGL <100 / 100-125 / >=126.
          Smoking: SMQ020 = 1 and SMQ040 in {1,2} -> 1, otherwise 0. Alcohol (2013-16): ALQ120Q 1-365 -> 1, 0 -> 0,
          ALQ101 = 2 and ALQ110 = 2 (skip) -> 0; (2017-20): ALQ121 1-10 -> 1, 0 -> 0, ALQ111 = 2 (skip) -> 0.
          Active: weekly minutes = PAQ625*PAD630 + PAQ670*PAD675 + PAQ640*PAD645 + 2*(PAQ610*PAD615 + PAQ655*PAD660),
          an item being 0 when its gate (PAQ620/665/635/605/650) = 2; >= 150 -> 1. PP, MAP, risk_count as the primary
          cohort; column order = protocol Section 1.1 FEATURES. Outcome: any MCQ160B-F = 1 -> 1; all five = 2 -> 0;
          otherwise excluded. Complete case: missing, refused (7, 77, 777, 7777) or don't know (9, 99, 999, 9999)
          excludes the record, except the structural skips above.
Models    Stacking Ensemble and CatBoost: no saved model exists, so each is refitted at seed 42 on the PRIMARY training
          split (CatBoost as the Section 5 anchor; stacking as src/run_corrected_stacking.py) and must pass the
          fidelity gate (|dAUC| <= 0.001 and r >= 0.999 vs the deposited test arrays) before any NHANES scoring.
          XGBoost, Random Forest, MLP: Section 3 retrained artefacts. FT-Transformer, SAINT-inspired: deposited
          checkpoints. Inputs: RF, MLP, FT get scaler_seed42.transform(X) (FT as float32); SAINT gets its checkpoint
          scaler, batch 256, rows ordered by cycle then SEQN; XGBoost, CatBoost, stacking get raw X. No scaler refitted.
Reported  per model, cohort and scenario: (i) N, events, prevalence; (ii) AUC, 1,000-resample percentile bootstrap
          95% CI, seed 42 (one shared index vector per replicate across models); (iii) calibration intercept and
          slope as submitted (clip 1e-6; LogisticRegression(C=inf) of y on logit p); (iv) the same after
          intercept-only recalibration (intercept re-estimated on the NHANES cohort with logit p as offset, slope 1;
          in-sample); (v) Brier before and after; (vi) AUC drop vs the model's primary-split AUC; (vii) marginal
          distributions of the 13 features beside the primary cohort's.
Scenarios primary mapping; sensitivity (A) fasting LBXGLU for glucose; (B) ever-smoker SMQ020 = 1; (C) cholesterol
          <190 / 190-229 / >=230. Exactly these.

Run from the repository root (original Anaconda environment; CatBoost refit on GPU as the Section 5 anchor):
    python src/revision/amendment10_nhanes_external_validation.py --data-file <stage-1 CSV> --nhanes-dir <XPT dir>
NHANES files: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<2013|2015|2017>/DataFiles/<NAME>.xpt
Writes revision-results/nhanes/: external_validation_metrics.csv, cohort_flow.csv, feature_marginals.csv,
    external_validation.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "artifacts" / "predictions"
MREV = ROOT / "artifacts" / "models-revision"
OUT = ROOT / "revision-results" / "nhanes"
R = 42
FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose", "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
MODELS = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest", "MLP"]
DEPOSITED = {"Stacking Ensemble": "stack_corrected", "XGBoost": "xgb", "CatBoost": "cat", "FT-Transformer": "ft",
             "SAINT-inspired": "saint", "Random Forest": "rf", "MLP": "mlp"}
CYCLES = {"primary": {"2013-14": "{}_H", "2015-16": "{}_I"}, "secondary": {"2017-20": "P_{}"}}
SCENARIOS = ["primary mapping", "A fasting glucose", "B ever-smoker", "C cholesterol 190/230"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


s3 = _load("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")


# ============================================================================ NHANES cohort
def read_cycle(raw: Path, cyc: str, pat: str, hashes: dict) -> pd.DataFrame:
    names = ["DEMO", "BMX", "BPXO" if cyc == "2017-20" else "BPX", "TCHOL", "BIOPRO", "GLU", "SMQ", "ALQ", "PAQ", "MCQ"]
    base = None
    for n in names:
        f = raw / f"{pat.format(n)}.xpt"
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        df = pd.read_sas(f, format="xport", encoding="latin-1")
        assert df["SEQN"].is_unique
        base = df if base is None else base.merge(df, on="SEQN", how="left", suffixes=("", f"_{n}"))
    num = base.select_dtypes("number").columns
    tiny = base[num].notna() & (base[num].abs() < 1e-70)     # XPT stores 0 as ~5.4e-79
    base[num] = base[num].mask(tiny, 0.0)
    base.insert(0, "cycle", cyc)
    if cyc == "2017-20":
        for k in (1, 2, 3):
            base[f"SBP{k}"], base[f"DBP{k}"] = base[f"BPXOSY{k}"], base[f"BPXODI{k}"]
    else:
        for k in (1, 2, 3):
            base[f"SBP{k}"], base[f"DBP{k}"] = base[f"BPXSY{k}"], base[f"BPXDI{k}"]
    return base


def build_nhanes(df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, list]:
    flow = [("all participants", len(df))]
    d = df[df["RIDAGEYR"].between(40, 65)].copy(); flow.append(("age 40-65 inclusive", len(d)))
    S, D = d[["SBP1", "SBP2", "SBP3"]], d[["DBP1", "DBP2", "DBP3"]]
    d = d[~(D == 0).any(axis=1)]; flow.append(("no reading with DBP = 0", len(d)))
    d["SBP"] = d[["SBP1", "SBP2", "SBP3"]].mean(axis=1, skipna=True)
    d["DBP"] = d[["DBP1", "DBP2", "DBP3"]].mean(axis=1, skipna=True)
    d = d[d["SBP"].notna() & d["DBP"].notna()]; flow.append(("at least one BP reading", len(d)))
    d = d[d["SBP"].between(80, 200) & d["DBP"].between(40, 130)]; flow.append(("stage 1: SBP 80-200, DBP 40-130", len(d)))
    d["BMI"] = d["BMXBMI"]
    d = d[d["BMI"].between(10, 60) & ((d["SBP"] - d["DBP"]) > 0)]; flow.append(("stage 2: BMI 10-60, SBP - DBP > 0", len(d)))

    d["Gender"] = d["RIAGENDR"].map({1: 1.0, 2: 0.0})
    d["Age"] = d["RIDAGEYR"]
    tc = d["LBXTC"]
    cut = (190, 230) if scenario.startswith("C") else (200, 240)
    d["Cholesterol"] = np.select([tc < cut[0], tc < cut[1], tc >= cut[1]], [1.0, 2.0, 3.0], default=np.nan)
    g = d["LBXGLU"] if scenario.startswith("A") else d["LBXSGL"]
    d["Glucose"] = np.select([g < 100, g < 126, g >= 126], [1.0, 2.0, 3.0], default=np.nan)
    if scenario.startswith("B"):
        d["Smoking"] = np.select([d["SMQ020"] == 1, d["SMQ020"] == 2], [1.0, 0.0], default=np.nan)
    else:
        d["Smoking"] = np.select([(d["SMQ020"] == 1) & d["SMQ040"].isin([1, 2]), (d["SMQ020"] == 1) & (d["SMQ040"] == 3),
                                  d["SMQ020"] == 2], [1.0, 0.0, 0.0], default=np.nan)
    if d["cycle"].iloc[0] == "2017-20" if len(d) else False:
        q = d["ALQ121"]
        d["Alcohol"] = np.select([q.between(1, 10), q == 0, d["ALQ111"] == 2], [1.0, 0.0, 0.0], default=np.nan)
    else:
        q = d["ALQ120Q"]
        d["Alcohol"] = np.select([q.between(1, 365), q == 0, (d["ALQ101"] == 2) & (d["ALQ110"] == 2)], [1.0, 0.0, 0.0],
                                 default=np.nan)
    items = [("PAQ620", "PAQ625", "PAD630", 1), ("PAQ665", "PAQ670", "PAD675", 1), ("PAQ635", "PAQ640", "PAD645", 1),
             ("PAQ605", "PAQ610", "PAD615", 2), ("PAQ650", "PAQ655", "PAD660", 2)]
    total = pd.Series(0.0, index=d.index)
    for gate, days, mins, w in items:
        ok_yes = (d[gate] == 1) & d[days].between(1, 7) & d[mins].notna() & (d[mins] >= 0) & ~d[mins].isin([7777, 9999])
        val = np.where(d[gate] == 2, 0.0, np.where(ok_yes, w * d[days] * d[mins], np.nan))
        total = total + val
    d["active_minutes_week"] = total
    d["Active"] = np.where(total.notna(), (total >= 150).astype(float), np.nan)
    d["PP"] = d["SBP"] - d["DBP"]
    d["MAP"] = (d["SBP"] + 2 * d["DBP"]) / 3
    d["RiskCount"] = (d["Cholesterol"] > 1).astype(float) + (d["Glucose"] > 1).astype(float) + d["Smoking"] + d["Alcohol"] + (1 - d["Active"])
    mcq = d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    d["y"] = np.where((mcq == 1).any(axis=1), 1.0, np.where((mcq == 2).all(axis=1), 0.0, np.nan))

    for col in ["Gender", "Cholesterol", "Glucose", "Smoking", "Alcohol", "Active"]:
        d = d[d[col].notna()]; flow.append((f"complete: {col}", len(d)))
    d = d[d["y"].notna()]; flow.append(("outcome determinable", len(d)))
    d = d.sort_values(["cycle", "SEQN"]).reset_index(drop=True)          # fixed row order (SAINT batches)
    return d, flow


# ============================================================================ models
def refit_catboost(Xtr, ytr, Xva, yva):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3, bagging_temperature=0.5,
                           eval_metric="AUC", early_stopping_rounds=30, random_state=R, verbose=0, task_type="GPU",
                           allow_writing_files=False)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    return m


def refit_stacking(Xtrain, ytrain):
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    params = {
        "xgb": dict(n_estimators=186, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8, min_child_weight=3,
                    gamma=.1, reg_alpha=.1, reg_lambda=1.5, eval_metric="auc", random_state=R, verbosity=0,
                    tree_method="hist", device="cpu", n_jobs=-1),
        "lgb": dict(n_estimators=103, num_leaves=63, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                    min_child_samples=20, reg_alpha=.1, reg_lambda=1.0, random_state=R, verbose=-1, device="cpu", n_jobs=-1),
        "cat": dict(iterations=217, depth=6, learning_rate=.05, l2_leaf_reg=3, bagging_temperature=.5, eval_metric="AUC",
                    random_seed=R, verbose=0, task_type="CPU", allow_writing_files=False, thread_count=-1),
        "rf": dict(n_estimators=300, max_depth=10, min_samples_leaf=5, max_features="sqrt", random_state=R, n_jobs=-1)}

    def make_models(seed):  # src/run_corrected_stacking.py, unchanged
        p = {k: v.copy() for k, v in params.items()}
        p["xgb"]["random_state"] = seed; p["lgb"]["random_state"] = seed; p["cat"]["random_seed"] = seed; p["rf"]["random_state"] = seed
        return [xgb.XGBClassifier(**p["xgb"]), lgb.LGBMClassifier(**p["lgb"]), CatBoostClassifier(**p["cat"]),
                RandomForestClassifier(**p["rf"])]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=R)
    oof = np.zeros((len(Xtrain), 4))
    for fold, (tri, vi) in enumerate(skf.split(Xtrain, ytrain), 1):
        for j, m in enumerate(make_models(R + fold)):
            m.fit(Xtrain[tri], ytrain[tri]); oof[vi, j] = m.predict_proba(Xtrain[vi])[:, 1]
    meta = LogisticRegression(C=1, max_iter=2000, random_state=R).fit(oof, ytrain)
    final = make_models(R)
    for m in final:
        m.fit(Xtrain, ytrain)
    return final, meta


def calib(y, p):  # as submitted (src/run_validated_stats.py)
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c)).reshape(-1, 1)
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000).fit(lg, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def recalibrate_intercept(y, p):  # intercept re-estimated with logit(p) as offset, slope fixed at 1
    c = np.clip(p, 1e-6, 1 - 1e-6)
    off = np.log(c / (1 - c))
    a = brentq(lambda t: float(np.sum(y - expit(off + t))), -30, 30)
    return a, expit(off + a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", type=Path, required=True)
    ap.add_argument("--nhanes-dir", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import xgboost as xgb
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise SystemExit("GPU unavailable: the CatBoost refit follows the Section 5 anchor on GPU; stopping.")
    pipes = _load("pipes", ROOT / "src" / "revision" / "section4_2_pipelines.py")

    # ---------------- primary cohort, split, and the two refits with their fidelity gate
    X, y = s3.build_cohort(a.data_file)
    (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, _) = s3.primary_split(X, y)
    y_dep = np.load(PRED / "y_test.npy").astype(int)
    if not np.array_equal(y_test, y_dep):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    t0 = time.time()
    cat = refit_catboost(X_train, y_train, X_val, y_val)
    stack_base, stack_meta = refit_stacking(X_train, y_train)
    refit_seconds = time.time() - t0
    p_cat_test = cat.predict_proba(X_test)[:, 1]
    p_stack_test = stack_meta.predict_proba(np.column_stack([m.predict_proba(X_test)[:, 1] for m in stack_base]))[:, 1]
    gate = {}
    for name, p, f in (("CatBoost", p_cat_test, "cat"), ("Stacking Ensemble", p_stack_test, "stack_corrected")):
        dep = np.load(PRED / f"proba_{f}.npy").astype(float)
        dauc = float(roc_auc_score(y_dep, p) - roc_auc_score(y_dep, dep)); r = float(pearsonr(p, dep)[0])
        gate[name] = {"refit_auc": float(roc_auc_score(y_dep, p)), "deposited_auc": float(roc_auc_score(y_dep, dep)),
                      "abs_delta_auc": abs(dauc), "pearson_r": r, "max_abs_diff": float(np.max(np.abs(p - dep))),
                      "pass": abs(dauc) <= 0.001 and r >= 0.999}
    gate["catboost_best_iteration"] = int(cat.get_best_iteration())
    print("FIDELITY GATE:", json.dumps(gate, indent=1))
    if not all(v["pass"] for k, v in gate.items() if isinstance(v, dict)):
        with open(OUT / "external_validation.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"protocol_section": "Amendment 10", "fidelity_gate": gate, "status": "STOPPED: fidelity gate failed; nothing scored"}, fh, indent=2); fh.write("\n")
        raise SystemExit("FIDELITY GATE FAILED: no NHANES record is scored (Amendment 10).")

    # ---------------- the other models
    sc42 = joblib.load(MREV / "scaler_seed42.joblib")
    xg = xgb.XGBClassifier(); xg.load_model(str(MREV / "xgb_seed42.json")); xg.set_params(device="cpu")
    rf = joblib.load(MREV / "rf_seed42.joblib")
    mlp = joblib.load(MREV / "mlp_seed42.joblib")
    dev = torch.device("cuda")
    ft, _ = pipes.ft_model(); ft.to(dev).eval()
    sa, sack = pipes.saint_model(); sa.to(dev).eval()

    def predict_all(Xr: np.ndarray) -> dict:
        Xs = sc42.transform(Xr)
        with torch.no_grad():
            p_ft = F.softmax(ft(torch.tensor(Xs.astype(np.float32), device=dev)), dim=1)[:, 1].cpu().numpy()
            Xsa = sack["scaler"].transform(Xr.astype(np.float32)).astype(np.float32)
            p_sa = np.concatenate([F.softmax(sa(torch.tensor(Xsa[i:i + 256], device=dev)), dim=1)[:, 1].cpu().numpy()
                                   for i in range(0, len(Xsa), 256)])
        base = np.column_stack([m.predict_proba(Xr)[:, 1] for m in stack_base])
        return {"Stacking Ensemble": stack_meta.predict_proba(base)[:, 1], "XGBoost": xg.predict_proba(Xr)[:, 1],
                "CatBoost": cat.predict_proba(Xr)[:, 1], "FT-Transformer": p_ft.astype(float), "SAINT-inspired": p_sa.astype(float),
                "Random Forest": rf.predict_proba(Xs)[:, 1], "MLP": mlp.predict_proba(Xs)[:, 1]}

    # loading check on the primary test set: every model reproduces its deposited array
    pt = predict_all(X_test)
    loading = {m: {"max_abs_diff_vs_deposited": float(np.max(np.abs(pt[m] - np.load(PRED / f"proba_{DEPOSITED[m]}.npy")))),
                   "auc_primary_test": float(roc_auc_score(y_dep, pt[m]))} for m in MODELS}
    primary_auc = {m: float(roc_auc_score(y_dep, np.load(PRED / f"proba_{DEPOSITED[m]}.npy"))) for m in MODELS}
    print("LOADING CHECK:", json.dumps(loading, indent=1))

    # ---------------- NHANES
    hashes = {}
    raw = {c: pd.concat([read_cycle(a.nhanes_dir, cyc, pat, hashes) for cyc, pat in cycles.items()], ignore_index=True, sort=False)
           for c, cycles in CYCLES.items()}
    rows, flows, marg = [], [], []
    feat_cols = FEAT_NAMES
    for cohort in ("primary", "secondary"):
        for scen in SCENARIOS:
            d, flow = build_nhanes(raw[cohort], scen)
            flows += [{"cohort": cohort, "scenario": scen, "step": s, "n": n} for s, n in flow]
            Xn = d[feat_cols].to_numpy(float)
            yn = d["y"].to_numpy(int)
            P = predict_all(Xn)
            rng = np.random.default_rng(R); boot = {m: [] for m in MODELS}; disc = 0
            for _ in range(1000):
                ix = rng.integers(0, len(yn), len(yn))
                if yn[ix].min() == yn[ix].max():
                    disc += 1; continue
                for m in MODELS:
                    boot[m].append(roc_auc_score(yn[ix], P[m][ix]))
            for m in MODELS:
                auc = float(roc_auc_score(yn, P[m])); lo, hi = np.percentile(boot[m], [2.5, 97.5])
                ci, cs = calib(yn, P[m]); aint, prec = recalibrate_intercept(yn, P[m]); ci2, cs2 = calib(yn, prec)
                rows.append({"cohort": cohort, "scenario": scen, "model": m, "n": int(len(yn)), "events": int(yn.sum()),
                             "prevalence": float(yn.mean()), "auc": auc, "auc_ci_low": float(lo), "auc_ci_high": float(hi),
                             "primary_split_auc": primary_auc[m], "auc_drop": primary_auc[m] - auc,
                             "calibration_intercept": ci, "calibration_slope": cs, "mean_predicted": float(P[m].mean()),
                             "recalibration_offset": float(aint), "calibration_intercept_after": ci2, "calibration_slope_after": cs2,
                             "brier_before": float(brier_score_loss(yn, P[m])), "brier_after": float(brier_score_loss(yn, prec)),
                             "bootstrap_discarded": disc})
            if scen == "primary mapping":
                for j, f in enumerate(feat_cols):
                    v = Xn[:, j]
                    marg.append({"cohort": f"NHANES {cohort}", "feature": f, "n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "p25": float(np.percentile(v, 25)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                                 **({f"share_{int(k)}": float((v == k).mean()) for k in np.unique(X[:, j])} if f in ("Gender", "Cholesterol", "Glucose", "Smoking", "Alcohol", "Active") else {})})
    for j, f in enumerate(feat_cols):
        v = X[:, j].astype(float)
        marg.append({"cohort": "primary cohort (68,534)", "feature": f, "n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "p25": float(np.percentile(v, 25)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                     **({f"share_{int(k)}": float((v == k).mean()) for k in np.unique(v)} if f in ("Gender", "Cholesterol", "Glucose", "Smoking", "Alcohol", "Active") else {})})

    met = pd.DataFrame(rows)
    met.to_csv(OUT / "external_validation_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(flows).to_csv(OUT / "cohort_flow.csv", index=False, lineterminator="\n")
    pd.DataFrame(marg).to_csv(OUT / "feature_marginals.csv", index=False, lineterminator="\n")
    out = {"protocol_section": "Amendment 10 (prespecified)", "amendment_commit": "8867e052118999347fa70edf0117ac469ff8e1c9",
           "fidelity_gate": gate, "refit_seconds": refit_seconds, "loading_check_primary_test": loading,
           "primary_split_auc": primary_auc, "nhanes_files_sha256": hashes,
           "nhanes_source": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<2013|2015|2017>/DataFiles/<NAME>.xpt",
           "primary_label_prevalence": float(y.mean()), "survey_weights": "not applied",
           "notes": ["Outcome is self-reported diagnosed CVD history (MCQ160B-F), a different construct from the primary label; "
                     "a poor result cannot distinguish model failure from label mismatch.",
                     "Recalibration (iv) is in-sample on each NHANES cohort.",
                     "Sensitivity analyses (A)-(C) are run on both cohorts; no others."]}
    with open(OUT / "external_validation.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")

    pd.set_option("display.width", 250)
    show = met[["cohort", "scenario", "model", "n", "events", "prevalence", "auc", "auc_ci_low", "auc_ci_high", "auc_drop",
                "calibration_intercept", "calibration_slope", "calibration_intercept_after", "brier_before", "brier_after"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
