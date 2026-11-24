#!/usr/bin/env python3
"""Protocol Section 4.2: the five explanation pipelines on the common sample.

Implements docs/ANALYSIS_PROTOCOL.md Section 4.2 (pipeline numbering as in the
protocol). Every pipeline reads the Section 4.1 sample through
common_sample.load_common_sample(), which refuses a file whose SHA-256 differs
from xai_common_sample_manifest.json. Rows are passed in ascending xai_idx order.

  1 mlp    MLP permutation importance (Section 2.2 retrained MLP), X_test_sc[xai_idx];
           sklearn permutation_importance(scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1)
           (Classical_Models_CVD.ipynb cell 35).
  2 xgb    XGBoost TreeSHAP (Section 2.3 retrained XGBoost), raw X_test[xai_idx];
           shap.TreeExplainer(model).shap_values(X); importance = mean |SHAP|
           (Stacking_Ensemble_CVD_GPU.ipynb cells 29, 31).
  3 tabnet TabNet attention, artifacts/models/tabnet_cvd_model.zip, X_test_sc[xai_idx];
           explain_matrix, _ = tabnet.explain(X); importance = explain_matrix.sum(axis=0),
           normalised to sum to 1 (pytorch-tabnet's feature_importances_ formula).
  4 ft     FT-Transformer KernelSHAP, artifacts/models/ft_transformer_cvd.pt, X_test_sc[xai_idx];
           np.random.seed(42); background = X_train_sc[np.random.choice(len(X_train_sc), 200,
           replace=False)]; shap.KernelExplainer(class-probability function, background);
           shap_values(X, nsamples=100); importance = mean |SHAP| for class 1
           (FT_Transformer_CVD.ipynb cell 21). --ft-rows N explains only the first N sample rows
           (timing probe; writes no importance file).
  5 saint  SAINT-inspired permutation, artifacts/models/saint_cvd_model_metadata_corrected.pt,
           canonical feature order and the checkpoint's own scaler as in src/run_saint_sensitivity.py;
           importance = baseline AUC - permuted AUC; n_repeats=10; permutations from
           np.random.default_rng(42), features outer (0-12), repeats inner, one rng.permutation(n)
           per feature per repeat (the nesting of SAINT_CVD_Complete.ipynb cell 25);
           inference batch_size=256 in fixed row order.

Before explaining, each model predicts the full 10,281-record test set and is compared with
the deposited artifacts/predictions/proba_*.npy (a loading check, recorded, not a gate).
numpy/scikit-learn are imported before torch: importing torch first triggers a duplicate
OpenMP runtime error in the original Anaconda environment.

Run from the repository root:
    python src/revision/section4_2_pipelines.py --pipeline {mlp,xgb,tabnet,saint,ft} --data-file <CSV>
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True

from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402  (before torch, see docstring)
import pandas as pd
import sklearn  # noqa: F401
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "revision"))
from common_sample import load_common_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

FEAT_NAMES = ["Gender", "Age", "BMI", "SBP", "DBP", "Cholesterol", "Glucose",
              "Smoking", "Alcohol", "Active", "PP", "MAP", "RiskCount"]
XAI = ROOT / "revision-results" / "xai"
PRED = ROOT / "artifacts" / "predictions"
RECORDS = XAI / "pipeline_run_records.json"


def versions(*pkgs: str) -> dict:
    return {"python": sys.version.split()[0], **{p: md.version(p) for p in pkgs}}


def loading_check(name: str, proba_full: np.ndarray, y_test: np.ndarray) -> dict:
    dep = np.load(PRED / f"proba_{name}.npy").astype(float).reshape(-1)
    p = np.asarray(proba_full, dtype=float).reshape(-1)
    return {"deposited_array": f"proba_{name}.npy",
            "max_abs_diff": float(np.max(np.abs(p - dep))),
            "pearson_correlation": float(np.corrcoef(p, dep)[0, 1]),
            "auc_deposited": float(roc_auc_score(y_test, dep)),
            "auc_reproduced": float(roc_auc_score(y_test, p))}


def save_record(key: str, record: dict) -> None:
    data = json.loads(RECORDS.read_text(encoding="utf-8")) if RECORDS.exists() else {}
    data[key] = record
    with open(RECORDS, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def write_csv(df: pd.DataFrame, fname: str) -> None:
    df.to_csv(XAI / fname, index=False, lineterminator="\n")


# ---------------------------------------------------------------- pipeline 1
def run_mlp(ctx: dict) -> dict:
    import joblib
    from sklearn.inspection import permutation_importance
    mlp = joblib.load(ROOT / "artifacts" / "models-revision" / "mlp_seed42.joblib")
    check = loading_check("mlp", mlp.predict_proba(ctx["X_test_sc"])[:, 1], ctx["y_test"])
    sec3 = np.load(ROOT / "revision-results" / "fidelity" / "proba_mlp_retrained_seed42.npy")
    check["max_abs_diff_vs_section3_retrained"] = float(np.max(np.abs(mlp.predict_proba(ctx["X_test_sc"])[:, 1] - sec3)))
    i = ctx["xai_idx"]
    pi = permutation_importance(mlp, ctx["X_test_sc"][i], ctx["y_test"][i], scoring="roc_auc",
                                n_repeats=10, random_state=42, n_jobs=-1)
    df = pd.DataFrame({"feature": FEAT_NAMES, "importance_mean_auc_drop": pi.importances_mean,
                       "sd_ddof0_sklearn": pi.importances_std,
                       "sd_ddof1": pi.importances.std(axis=1, ddof=1)})
    for r in range(10):
        df[f"drop_r{r + 1}"] = pi.importances[:, r]
    write_csv(df, "importance_mlp_permutation.csv")
    return {"importance": df.set_index("feature")["importance_mean_auc_drop"], "loading_check": check,
            "device": "cpu (scikit-learn)", "versions": versions("scikit-learn", "numpy")}


# ---------------------------------------------------------------- pipeline 2
def run_xgb(ctx: dict) -> dict:
    import shap
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(ROOT / "artifacts" / "models-revision" / "xgb_seed42.json"))
    full = model.predict_proba(ctx["X_test"])[:, 1]
    check = loading_check("xgb", full, ctx["y_test"])
    sec3 = np.load(ROOT / "revision-results" / "fidelity" / "proba_xgb_retrained_seed42.npy")
    check["max_abs_diff_vs_section3_retrained"] = float(np.max(np.abs(full - sec3)))
    check["best_iteration"] = int(model.best_iteration)
    X = ctx["X_test"][ctx["xai_idx"]]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    # Additivity: which trees did TreeSHAP explain? (record, do not change the method)
    margin_all = model.get_booster().predict(xgb.DMatrix(X), output_margin=True,
                                             iteration_range=(0, model.get_booster().num_boosted_rounds()))
    margin_best = model.predict(X, output_margin=True)
    recon = sv.sum(axis=1) + float(np.ravel(explainer.expected_value)[0])
    check["n_trees_in_booster"] = int(model.get_booster().num_boosted_rounds())
    check["shap_additivity_max_err_vs_all_trees"] = float(np.max(np.abs(recon - margin_all)))
    check["shap_additivity_max_err_vs_best_iteration_trees"] = float(np.max(np.abs(recon - margin_best)))
    df = pd.DataFrame({"feature": FEAT_NAMES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    write_csv(df, "importance_xgb_treeshap.csv")
    return {"importance": df.set_index("feature")["mean_abs_shap"], "loading_check": check,
            "device": "cpu (shap TreeExplainer)", "versions": versions("xgboost", "shap", "numpy")}


# ---------------------------------------------------------------- pipeline 3
def run_tabnet(ctx: dict) -> dict:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    tabnet = TabNetClassifier(device_name="cuda" if torch.cuda.is_available() else "cpu")
    tabnet.load_model(str(ROOT / "artifacts" / "models" / "tabnet_cvd_model.zip"))
    check = loading_check("tabnet", tabnet.predict_proba(ctx["X_test_sc"])[:, 1], ctx["y_test"])
    X = ctx["X_test_sc"][ctx["xai_idx"]]
    explain_matrix, _ = tabnet.explain(X)
    s = explain_matrix.sum(axis=0)
    imp = s / s.sum()
    # Side check for the discrepancy record: the training-set formula (what fit() stores).
    tr = tabnet.explain(ctx["X_train_sc"])[0].sum(axis=0)
    check["training_set_feature_importances"] = dict(zip(FEAT_NAMES, (tr / tr.sum()).round(4).tolist()))
    df = pd.DataFrame({"feature": FEAT_NAMES, "normalised_attention_importance": imp,
                       "explain_matrix_column_sum": s})
    write_csv(df, "importance_tabnet_attention.csv")
    return {"importance": df.set_index("feature")["normalised_attention_importance"], "loading_check": check,
            "device": str(tabnet.device), "versions": versions("pytorch-tabnet", "torch", "numpy")}


# ---------------------------------------------------------------- pipeline 5
def saint_model():
    import torch
    import torch.nn as nn

    class FeatureEmbedding(nn.Module):
        def __init__(self, n_features, d_model):
            super().__init__(); self.n_features = n_features
            self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
            self.feature_positions = nn.Parameter(torch.randn(1, n_features, d_model))

        def forward(self, x):
            return torch.stack([self.feature_embeddings[i](x[:, i:i + 1]) for i in range(self.n_features)], dim=1) + self.feature_positions

    class SelfAttentionBlock(nn.Module):
        def __init__(self, d_model, n_heads, dropout=.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

        def forward(self, x):
            a, _ = self.attention(x, x, x); x = self.norm1(x + a); f = self.ffn(x); return self.norm2(x + f)

    class IntersampleAttentionBlock(SelfAttentionBlock):
        def forward(self, x):
            xt = x.transpose(0, 1); a, _ = self.attention(xt, xt, xt); xt = self.norm1(xt + a)
            f = self.ffn(xt); xt = self.norm2(xt + f); return xt.transpose(0, 1)

    class SAINT(nn.Module):
        def __init__(self, n_features, n_classes=2, d_model=64, n_heads=8, n_layers=4, dropout=.15):
            super().__init__()
            self.feature_embedding = FeatureEmbedding(n_features, d_model); self.blocks = nn.ModuleList()
            for _ in range(n_layers):
                self.blocks.extend([SelfAttentionBlock(d_model, n_heads, dropout), IntersampleAttentionBlock(d_model, n_heads, dropout)])
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Sequential(nn.Linear(d_model * n_features, 128), nn.LayerNorm(128), nn.GELU(),
                                            nn.Dropout(dropout), nn.Linear(128, n_classes))

        def forward(self, x):
            x = self.feature_embedding(x)
            for b in self.blocks:
                x = b(x)
            return self.classifier(self.norm(x).reshape(x.size(0), -1))

    ck = torch.load(ROOT / "artifacts" / "models" / "saint_cvd_model_metadata_corrected.pt", map_location="cpu", weights_only=False)
    model = SAINT(ck["n_features"], **ck["hyperparameters"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, ck


def run_saint(ctx: dict) -> dict:
    import torch
    model, ck = saint_model()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    # Canonical order (run_saint_sensitivity.py) equals the Section 1.1 column order; X cast to float32
    # before the checkpoint's own scaler, as that script does.
    Xs_full = ck["scaler"].transform(ctx["X_test"].astype(np.float32)).astype(np.float32)

    @torch.no_grad()
    def predict(X):  # fixed row order, batch 256 (SAINT_CVD_Complete.ipynb cell 25)
        out = []
        for s in range(0, len(X), 256):
            out.append(torch.softmax(model(torch.tensor(X[s:s + 256], device=dev)), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    check = loading_check("saint", predict(Xs_full), ctx["y_test"])
    check["checkpoint_scaler_max_abs_diff_vs_section12_scaler_mean"] = float(np.max(np.abs(ck["scaler"].mean_ - ctx["scaler"].mean_)))
    i = ctx["xai_idx"]; X = Xs_full[i]; y = ctx["y_test"][i]; n = len(i)
    base = roc_auc_score(y, predict(X))
    rng = np.random.default_rng(42)
    drops = np.empty((13, 10))
    for j in range(13):
        for r in range(10):
            perm = rng.permutation(n)
            Xp = X.copy(); Xp[:, j] = X[perm, j]
            drops[j, r] = base - roc_auc_score(y, predict(Xp))
    df = pd.DataFrame({"feature": FEAT_NAMES, "importance_mean_auc_drop": drops.mean(axis=1),
                       "sd_ddof0": drops.std(axis=1), "sd_ddof1": drops.std(axis=1, ddof=1)})
    for r in range(10):
        df[f"drop_r{r + 1}"] = drops[:, r]
    df["baseline_auc_common_sample"] = base
    write_csv(df, "importance_saint_permutation.csv")
    return {"importance": df.set_index("feature")["importance_mean_auc_drop"], "loading_check": check,
            "device": str(dev), "versions": versions("torch", "numpy", "scikit-learn")}


# ---------------------------------------------------------------- pipeline 4
def ft_model():
    import torch
    import torch.nn as nn

    class FeatureTokenizer(nn.Module):  # FT_Transformer_CVD.ipynb cell 13, unchanged
        def __init__(self, n_features, d_token):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x):
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    class TransformerBlock(nn.Module):
        def __init__(self, d_token, n_heads, d_ffn, dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(d_token)
            self.ffn = nn.Sequential(nn.Linear(d_token, d_ffn), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(d_ffn, d_token), nn.Dropout(dropout))

        def forward(self, x):
            h = self.norm1(x); h, _ = self.attn(h, h, h); x = x + h
            return x + self.ffn(self.norm2(x))

    class FTTransformer(nn.Module):
        def __init__(self, n_features, d_token=64, n_heads=8, n_layers=3, d_ffn=256, dropout=0.1, n_classes=2):
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_features, d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            self.layers = nn.ModuleList([TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Sequential(nn.Linear(d_token, d_ffn // 2), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d_ffn // 2, n_classes))

        def forward(self, x):
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for layer in self.layers:
                tokens = layer(tokens)
            return self.head(self.norm(tokens[:, 0]))

    ck = torch.load(ROOT / "artifacts" / "models" / "ft_transformer_cvd.pt", map_location="cpu", weights_only=False)
    model = FTTransformer(13, **ck["hyperparams"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, ck


def run_ft(ctx: dict, rows: int | None) -> dict:
    import shap
    import torch
    import torch.nn.functional as F
    model, ck = ft_model()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    X_train_sc = ctx["X_train_sc"].astype(np.float32)   # FT notebook casts the scaled matrices to float32
    X_test_sc = ctx["X_test_sc"].astype(np.float32)

    def ftt_predict_proba(X_np):  # FT_Transformer_CVD.ipynb cell 21
        with torch.no_grad():
            return F.softmax(model(torch.tensor(X_np.astype(np.float32)).to(dev)), dim=1).cpu().numpy()

    check = loading_check("ft", ftt_predict_proba(X_test_sc)[:, 1], ctx["y_test"])
    check["checkpoint_scaler_max_abs_diff_vs_section12_scaler_mean"] = float(np.max(np.abs(np.asarray(ck["scaler_mean"]) - ctx["scaler"].mean_)))
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train_sc), 200, replace=False)
    background = X_train_sc[bg_idx]
    X = X_test_sc[ctx["xai_idx"]]
    if rows is not None:
        X = X[:rows]
    explainer = shap.KernelExplainer(ftt_predict_proba, background)
    t0 = time.time()
    sv = explainer.shap_values(X, nsamples=100)
    elapsed = time.time() - t0
    sv = np.asarray(sv)
    sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    out = {"loading_check": check, "device": str(dev), "versions": versions("shap", "torch", "numpy"),
           "rows_explained": int(len(X)), "shap_seconds": elapsed, "seconds_per_row": elapsed / len(X),
           "background_idx_first5": bg_idx[:5].tolist()}
    if rows is None:
        df = pd.DataFrame({"feature": FEAT_NAMES, "mean_abs_shap_class1": np.abs(sv1).mean(axis=0)})
        write_csv(df, "importance_ft_kernelshap.csv")
        out["importance"] = df.set_index("feature")["mean_abs_shap_class1"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, choices=["mlp", "xgb", "tabnet", "saint", "ft"])
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--ft-rows", type=int, default=None, help="timing probe: explain only the first N sample rows")
    args = parser.parse_args()

    xai_idx, sha = load_common_sample()
    X, y = s3.build_cohort(args.data_file)
    (X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler) = s3.primary_split(X, y)
    if not np.array_equal(y_test, np.load(PRED / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    ctx = dict(xai_idx=xai_idx, X_test=X_test, X_test_sc=X_test_sc, X_train_sc=X_train_sc, y_test=y_test, scaler=scaler)

    t0 = time.time()
    if args.pipeline == "ft":
        res = run_ft(ctx, args.ft_rows)
    else:
        res = {"mlp": run_mlp, "xgb": run_xgb, "tabnet": run_tabnet, "saint": run_saint}[args.pipeline](ctx)
    wall = time.time() - t0

    number = {"mlp": 1, "xgb": 2, "tabnet": 3, "ft": 4, "saint": 5}[args.pipeline]
    key = f"pipeline_{number}_{args.pipeline}" + (f"_timing_probe_{args.ft_rows}_rows" if args.ft_rows else "")
    rec = {k: v for k, v in res.items() if k != "importance"}
    rec.update({"common_sample_sha256": sha, "wall_seconds": wall,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    save_record(key, rec)
    print(json.dumps(rec, indent=2))
    if "importance" in res:
        imp = res["importance"].sort_values(ascending=False)
        print("\nimportance (descending):")
        print(imp.to_string())


if __name__ == "__main__":
    main()
