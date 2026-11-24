#!/usr/bin/env python3
"""Protocol Section 5: five-seed stability analysis.

Implements docs/ANALYSIS_PROTOCOL.md Section 5.1 (amendments in Section 8 apply).

Modes (run in protocol order):
  --anchor                  seed-42 anchor check: standalone XGBoost/LightGBM/CatBoost with early
                            stopping -> tree counts (best_iteration + 1) -> corrected stacking ->
                            fidelity gate vs proba_stack_corrected.npy and the 20% structural guard.
  --seed S --saint-restore {final,best}
                            one full seed: six ranked models + standalone LightGBM. At seed 42 the
                            anchor's boosters and stacking are reused, not refitted.
  --summarise               build the Section 7 summary outputs from the seeds run.

Device policy (Section 5.1): ONE device for all of Section 5 -- the GPU (GTX 1650), set explicitly:
XGBoost device='cuda', LightGBM device='gpu', CatBoost task_type='GPU', PyTorch 'cuda'. The
notebook pattern `'cuda' if GPU_AVAILABLE else 'cpu'` is NOT used; GPU presence is asserted
before every model fit and a failed assertion stops the run (no CPU fallback). scikit-learn
Random Forest and the logistic-regression meta-learner run on CPU, identically at every seed.

Seed s replaces RANDOM_STATE everywhere: both train_test_split steps (Section 1.2, scaler refit
on the seed's training set), every model's random state, StratifiedKFold(random_state=s) and
fold-model seeds s + fold, and np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all.

Training procedures are copied from the executed notebooks:
  XGBoost   Section 2.3 / Stacking_Ensemble_CVD_GPU.ipynb cell 15 (raw X, eval on X_val, ES 30)
  LightGBM  cell 16 (raw X, callbacks early_stopping(30) + log_evaluation(-1))
  CatBoost  cell 17 (raw X, eval_set=(X_val, y_val), ES 30)
  RF        Section 2.1 (scaled X)
  Stacking  src/run_corrected_stacking.py with per-seed counts = best_iteration + 1 for all three
            boosters, held fixed in all five OOF folds and the final refit (no ES inside folds)
  FT        FT_Transformer_CVD.ipynb cells 11, 13, 15 (float32 scaled X; AdamW 1e-3/1e-4;
            CosineAnnealingLR(T_max=100, eta_min=1e-5); clip 1.0; <=150 epochs; patience 20 on
            val AUC; best state restored with copy.deepcopy, as the notebook does)
  SAINT     SAINT_CVD_Complete.ipynb cells 10, 12, 15, 17, 18, 19 (float32 X scaled; class-weighted
            CE; AdamW 1e-3/1e-4; CosineAnnealingLR(T_max=50, eta_min=1e-6); clip 1.0; <=50 epochs;
            patience 10 on val AUC; batch 256). --saint-restore must be given explicitly: the
            notebook's `model.state_dict().copy()` is a shallow copy, so the executed code restores
            the LAST-epoch weights ('final'); 'best' restores the best-validation-epoch weights.

Run from the repository root (original Anaconda environment):
    python src/revision/section5_five_seed.py --anchor --data-file <stage-1 CSV>
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata as md
import importlib.util
import json
import math
import subprocess
import sys
import time

sys.dont_write_bytecode = True

from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # numpy / scikit-learn before torch (duplicate-OpenMP issue in this environment)
import pandas as pd
import sklearn
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb  # noqa: E402
import lightgbm as lgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-results" / "five_seed"
PRED = ROOT / "artifacts" / "predictions"
MANIFEST = OUT / "five_seed_run_manifest.json"
_spec = importlib.util.spec_from_file_location("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)

XGB_DEVICE, LGB_DEVICE, CAT_TASK, TORCH_DEVICE = "cuda", "gpu", "GPU", "cuda"
PUBLISHED_COUNTS = {"xgboost": 186, "lightgbm": 103, "catboost": 217}
GUARD = {k: (math.ceil(v * 0.8), math.floor(v * 1.2)) for k, v in PUBLISHED_COUNTS.items()}  # 149-223, 83-123, 174-260
DEPOSITED_STACK_AUC = 0.800367
RANKED = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]
SUFFIX = {"Stacking Ensemble": "stack", "XGBoost": "xgb", "CatBoost": "cat", "FT-Transformer": "ft",
          "SAINT-inspired": "saint", "Random Forest": "rf"}


# ------------------------------------------------------------------ infrastructure
def assert_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("GPU UNAVAILABLE -> STOP (Section 5.1: no CPU fallback; restart on CPU from scratch or wait).")


def environment() -> dict:
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"python": sys.version.split()[0],
            **{p: md.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch"]},
            "gpu_and_driver": drv, "cuda_torch": torch.version.cuda,
            "device": {"xgboost": XGB_DEVICE, "lightgbm": LGB_DEVICE, "catboost": CAT_TASK, "torch": TORCH_DEVICE,
                       "random_forest_and_meta_learner": "cpu (scikit-learn)"}}


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"runs": []}


def save_manifest(m: dict) -> None:
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=2); fh.write("\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(s: int) -> None:
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split(X: np.ndarray, y: np.ndarray, s: int) -> dict:
    """Section 1.2 with seed s; row indices into the analytical cohort carried alongside."""
    idx = np.arange(len(y))
    X_tmp, X_te, y_tmp, y_te, i_tmp, i_te = train_test_split(X, y, idx, test_size=0.15, random_state=s, stratify=y)
    X_tr, X_va, y_tr, y_va, i_tr, i_va = train_test_split(X_tmp, y_tmp, i_tmp, test_size=0.1765, random_state=s, stratify=y_tmp)
    sc = StandardScaler()
    return dict(X_train=X_tr, X_val=X_va, X_test=X_te, y_train=y_tr, y_val=y_va, y_test=y_te,
                i_train=i_tr, i_val=i_va, i_test=i_te,
                X_train_sc=sc.fit_transform(X_tr), X_val_sc=sc.transform(X_va), X_test_sc=sc.transform(X_te))


def timed(fn, *a, **k):
    assert_gpu()
    t0 = time.time(); r = fn(*a, **k); return r, time.time() - t0


# ------------------------------------------------------------------ boosters (standalone, early stopping)
def fit_xgb(d, s):
    m = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5, eval_metric="auc",
                          early_stopping_rounds=30, random_state=s, verbosity=0, tree_method="hist", device=XGB_DEVICE)
    m.fit(d["X_train"], d["y_train"], eval_set=[(d["X_val"], d["y_val"])], verbose=False)
    return m.predict_proba(d["X_test"])[:, 1], int(m.best_iteration)


def fit_lgb(d, s):
    m = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=s, verbose=-1, device=LGB_DEVICE)
    m.fit(d["X_train"], d["y_train"], eval_set=[(d["X_val"], d["y_val"])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return m.predict_proba(d["X_test"])[:, 1], int(m.best_iteration_)


def fit_cat(d, s):
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3, bagging_temperature=0.5,
                           eval_metric="AUC", early_stopping_rounds=30, random_state=s, verbose=0, task_type=CAT_TASK,
                           allow_writing_files=False)  # log files only; no effect on the model
    m.fit(d["X_train"], d["y_train"], eval_set=(d["X_val"], d["y_val"]), verbose=False)
    return m.predict_proba(d["X_test"])[:, 1], int(m.best_iteration_)


# ------------------------------------------------------------------ corrected stacking, per-seed counts
def fit_stack(d, s, counts):
    params = {
        "xgb": dict(n_estimators=counts["xgboost"], max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                    min_child_weight=3, gamma=.1, reg_alpha=.1, reg_lambda=1.5, eval_metric="auc", random_state=s,
                    verbosity=0, tree_method="hist", device=XGB_DEVICE, n_jobs=-1),
        "lgb": dict(n_estimators=counts["lightgbm"], num_leaves=63, max_depth=6, learning_rate=.05, subsample=.8,
                    colsample_bytree=.8, min_child_samples=20, reg_alpha=.1, reg_lambda=1.0, random_state=s,
                    verbose=-1, device=LGB_DEVICE, n_jobs=-1),
        "cat": dict(iterations=counts["catboost"], depth=6, learning_rate=.05, l2_leaf_reg=3, bagging_temperature=.5,
                    eval_metric="AUC", random_seed=s, verbose=0, task_type=CAT_TASK, allow_writing_files=False,
                    thread_count=-1),
        "rf": dict(n_estimators=300, max_depth=10, min_samples_leaf=5, max_features="sqrt", random_state=s, n_jobs=-1),
    }

    def make_models(seed):
        p = {k: v.copy() for k, v in params.items()}
        p["xgb"]["random_state"] = seed; p["lgb"]["random_state"] = seed; p["cat"]["random_seed"] = seed; p["rf"]["random_state"] = seed
        return [xgb.XGBClassifier(**p["xgb"]), lgb.LGBMClassifier(**p["lgb"]), CatBoostClassifier(**p["cat"]),
                RandomForestClassifier(**p["rf"])]

    Xtr, ytr, Xte = d["X_train"], d["y_train"], d["X_test"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    oof = np.zeros((len(Xtr), 4)); folds = []
    for fold, (tri, vi) in enumerate(skf.split(Xtr, ytr), 1):
        for j, m in enumerate(make_models(s + fold)):
            assert_gpu(); m.fit(Xtr[tri], ytr[tri]); oof[vi, j] = m.predict_proba(Xtr[vi])[:, 1]
        folds.append({"fold": fold, **{n: float(roc_auc_score(ytr[vi], oof[vi, j]))
                                       for j, n in enumerate(["XGBoost", "LightGBM", "CatBoost", "Random Forest"])}})
    meta = LogisticRegression(C=1, max_iter=2000, random_state=s).fit(oof, ytr)
    ptest = np.zeros((len(Xte), 4))
    for j, m in enumerate(make_models(s)):
        assert_gpu(); m.fit(Xtr, ytr); ptest[:, j] = m.predict_proba(Xte)[:, 1]
    return meta.predict_proba(ptest)[:, 1], {"oof_folds": folds, "meta_coef": meta.coef_[0].tolist(),
                                            "meta_intercept": float(meta.intercept_[0])}


# ------------------------------------------------------------------ Random Forest (Section 2.1)
def fit_rf(d, s):
    m = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, max_features="sqrt",
                               random_state=s, n_jobs=-1)
    m.fit(d["X_train_sc"], d["y_train"])
    return m.predict_proba(d["X_test_sc"])[:, 1], None


# ------------------------------------------------------------------ FT-Transformer (notebook cells 11, 13, 15)
class FeatureTokenizer(nn.Module):
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


def fit_ft(d, s):
    seed_everything(s)
    dev = torch.device(TORCH_DEVICE)
    Xtr, Xva, Xte = (d[k].astype(np.float32) for k in ("X_train_sc", "X_val_sc", "X_test_sc"))
    t = lambda X, y: (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    train_loader = DataLoader(TensorDataset(*t(Xtr, d["y_train"])), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(*t(Xva, d["y_val"])), batch_size=2048, shuffle=False)
    test_loader = DataLoader(TensorDataset(*t(Xte, d["y_test"])), batch_size=2048, shuffle=False)
    model = FTTransformer(13, d_token=64, n_heads=8, n_layers=3, d_ffn=256, dropout=0.1, n_classes=2).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    best_auc, best_state, no_improve, best_epoch, epoch = 0.0, None, 0, 0, 0
    for epoch in range(1, 151):
        assert_gpu(); model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            optimizer.zero_grad(); loss = criterion(model(Xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); vp = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                vp.append(F.softmax(model(Xb.to(dev)), dim=1)[:, 1].cpu().numpy())
        val_auc = roc_auc_score(d["y_val"], np.concatenate(vp))
        scheduler.step()
        if val_auc > best_auc:
            best_auc, best_state, no_improve, best_epoch = val_auc, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            no_improve += 1
        if no_improve >= 20:
            break
    model.load_state_dict(best_state); model.eval(); tp = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            tp.append(F.softmax(model(Xb.to(dev)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(tp), {"best_epoch": best_epoch, "epochs_run": epoch, "best_val_auc": float(best_auc)}


# ------------------------------------------------------------------ SAINT-inspired (notebook cells 10-19)
class FeatureEmbedding(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__(); self.n_features = n_features; self.d_model = d_model
        self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.feature_positions = nn.Parameter(torch.randn(1, n_features, d_model))

    def forward(self, x):
        return torch.stack([self.feature_embeddings[i](x[:, i:i + 1]) for i in range(self.n_features)], dim=1) + self.feature_positions


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        a, _ = self.attention(x, x, x); x = self.norm1(x + a); return self.norm2(x + self.ffn(x))


class IntersampleAttentionBlock(SelfAttentionBlock):
    def forward(self, x):
        xt = x.transpose(0, 1); a, _ = self.attention(xt, xt, xt); xt = self.norm1(xt + a)
        xt = self.norm2(xt + self.ffn(xt)); return xt.transpose(0, 1)


class SAINT(nn.Module):
    def __init__(self, n_features, n_classes=2, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_features = n_features; self.n_classes = n_classes; self.d_model = d_model
        self.feature_embedding = FeatureEmbedding(n_features, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(SelfAttentionBlock(d_model, n_heads, dropout))
            self.blocks.append(IntersampleAttentionBlock(d_model, n_heads, dropout))
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model * n_features, 128), nn.LayerNorm(128), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.feature_embedding(x)
        for b in self.blocks:
            x = b(x)
        return self.classifier(self.norm(x).reshape(x.size(0), -1))


class SAINTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X); self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def fit_saint(d, s, restore, diagnostic=False):
    seed_everything(s)
    diag_state = None
    dev = torch.device(TORCH_DEVICE)
    # Cell 10 casts X to float32 before the split; cell 12 fits the scaler on that float32 training set.
    sc = StandardScaler()
    Xtr = sc.fit_transform(d["X_train"].astype(np.float32)); Xva = sc.transform(d["X_val"].astype(np.float32))
    Xte = sc.transform(d["X_test"].astype(np.float32))
    model = SAINT(n_features=13, n_classes=2, d_model=64, n_heads=8, n_layers=4, dropout=0.15).to(dev)
    counts = np.bincount(d["y_train"])
    cw = torch.FloatTensor([1.0 / c for c in counts]).to(dev); cw = cw / cw.sum() * len(counts)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    tl = DataLoader(SAINTDataset(Xtr, d["y_train"]), batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    vl = DataLoader(SAINTDataset(Xva, d["y_val"]), batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    te = DataLoader(SAINTDataset(Xte, d["y_test"]), batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    best_auc, best_state, patience_counter, best_epoch, epoch = 0, None, 0, 0, 0
    for epoch in range(1, 51):
        assert_gpu(); model.train()
        for Xb, yb in tl:
            Xb, yb = Xb.to(dev), yb.to(dev)
            optimizer.zero_grad(); loss = criterion(model(Xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for Xb, yb in vl:
                vp.extend(F.softmax(model(Xb.to(dev)), dim=1)[:, 1].cpu().numpy()); vy.extend(yb.numpy())
        val_auc = roc_auc_score(np.array(vy), np.array(vp))
        scheduler.step()
        if val_auc > best_auc:
            best_auc, patience_counter, best_epoch = val_auc, 0, epoch
            best_state = model.state_dict().copy() if restore == "final" else copy.deepcopy(model.state_dict())
            if diagnostic:  # Amendment 4(b): true best-epoch weights, diagnostic only
                diag_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    def predict_test():
        model.eval(); out = []
        with torch.no_grad():
            for Xb, _ in te:
                out.extend(F.softmax(model(Xb.to(dev)), dim=1)[:, 1].cpu().numpy())
        return np.array(out)

    model.load_state_dict(best_state)
    tp = predict_test()          # the Section 5 result (as executed: final-epoch weights under 'final')
    info = {"best_epoch": best_epoch, "epochs_run": epoch, "best_val_auc": float(best_auc), "restore": restore,
            "weights_used_epoch": epoch if restore == "final" else best_epoch}
    if diagnostic:
        model.load_state_dict(diag_state)
        info["diagnostic_best_epoch_weights"] = {
            "note": "Amendment 4(b): diagnostic only; does not feed any Section 5 result",
            "best_epoch": best_epoch, "test_auc_best_epoch_weights": float(roc_auc_score(d["y_test"], predict_test())),
            "test_auc_final_epoch_weights_used": float(roc_auc_score(d["y_test"], tp))}
    return tp, info


# ------------------------------------------------------------------ modes
def load_data(data_file: Path):
    X, y = s3.build_cohort(data_file)
    return X, y


def save_split(d, s):
    np.savez(OUT / f"split_indices_seed{s}.npz", train=d["i_train"], val=d["i_val"], test=d["i_test"])
    np.save(OUT / f"y_test_seed{s}.npy", d["y_test"])


def run_anchor(X, y) -> None:
    s = 42
    d = split(X, y, s)
    if not np.array_equal(d["y_test"], np.load(PRED / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    save_split(d, s)
    t_start = time.time()
    (p_xgb, bi_xgb), t_xgb = timed(fit_xgb, d, s)
    (p_lgb, bi_lgb), t_lgb = timed(fit_lgb, d, s)
    (p_cat, bi_cat), t_cat = timed(fit_cat, d, s)
    counts = {"xgboost": bi_xgb + 1, "lightgbm": bi_lgb + 1, "catboost": bi_cat + 1}
    (p_stack, stack_info), t_stack = timed(fit_stack, d, s, counts)
    wall = time.time() - t_start

    dep = np.load(PRED / "proba_stack_corrected.npy").astype(float)
    auc_stack = float(roc_auc_score(d["y_test"], p_stack))
    corr = float(pearsonr(p_stack, dep)[0])
    dauc = auc_stack - float(roc_auc_score(d["y_test"], dep))
    gate = abs(dauc) <= 0.001 and corr >= 0.999
    guard = {k: {"derived_trees": counts[k], "published": PUBLISHED_COUNTS[k],
                 "pct_difference": 100 * (counts[k] - PUBLISHED_COUNTS[k]) / PUBLISHED_COUNTS[k],
                 "allowed_range": list(GUARD[k]),
                 "breach": abs(counts[k] - PUBLISHED_COUNTS[k]) / PUBLISHED_COUNTS[k] > 0.20} for k in counts}
    breach = any(g["breach"] for g in guard.values())
    result = {
        "protocol_section": "5.1 seed-42 anchor check", "finished_utc": now(),
        "fidelity_gate": {"stack_auc_retrained": auc_stack, "stack_auc_deposited": float(roc_auc_score(d["y_test"], dep)),
                          "abs_delta_auc": abs(dauc), "pearson_correlation": corr,
                          "criteria": "|dAUC| <= 0.001 and r >= 0.999", "pass": gate},
        "tree_counts_reported_not_gating": {"best_iteration": {"xgboost": bi_xgb, "lightgbm": bi_lgb, "catboost": bi_cat},
                                            "trees_used_best_iteration_plus_1": counts, "published": PUBLISHED_COUNTS},
        "structural_guard_20pct": {**guard, "any_breach": breach},
        "decision": "PROCEED" if (gate and not breach) else "STOP AND REPORT (Section 5.1 anchor rule 4)",
        "standalone_test_auc": {"xgboost": float(roc_auc_score(d["y_test"], p_xgb)),
                                "lightgbm": float(roc_auc_score(d["y_test"], p_lgb)),
                                "catboost": float(roc_auc_score(d["y_test"], p_cat))},
        "stacking_detail": stack_info,
        "environment": environment(),
        "wall_seconds": {"xgboost": t_xgb, "lightgbm": t_lgb, "catboost": t_cat, "stacking": t_stack, "total": wall},
    }
    with open(OUT / "seed42_anchor_check.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2); fh.write("\n")
    for name, p in (("xgb", p_xgb), ("cat", p_cat), ("stack", p_stack)):
        np.save(OUT / f"proba_{name}_seed{s}.npy", p)
    np.save(OUT / f"proba_lgbm_standalone_seed{s}.npy", p_lgb)
    m = manifest()
    m["runs"].append({"run": "anchor_seed42", "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
                      "finished_utc": result["finished_utc"], "environment": result["environment"],
                      "wall_seconds": result["wall_seconds"],
                      "models": {"XGBoost": {"best_iteration": bi_xgb}, "LightGBM (standalone)": {"best_iteration": bi_lgb},
                                 "CatBoost": {"best_iteration": bi_cat}, "Stacking Ensemble": {"trees": counts}},
                      "decision": result["decision"]})
    m["run_order"] = [r["run"] for r in m["runs"]]
    save_manifest(m)
    print(json.dumps({k: result[k] for k in ("fidelity_gate", "tree_counts_reported_not_gating", "structural_guard_20pct",
                                             "decision", "standalone_test_auc", "wall_seconds")}, indent=2))


def run_seed(X, y, s: int, restore: str) -> None:
    """One full seed (Section 5.1). Wall time: first model fit start -> last test prediction end."""
    done = {r["run"] for r in manifest()["runs"]}
    if f"seed{s}" in done:
        raise SystemExit(f"seed {s} already run; reruns must be logged explicitly")
    if s != 42 and "anchor_seed42" not in done:
        raise SystemExit("the seed-42 anchor check must run first")
    d = split(X, y, s)
    if s == 42 and not np.array_equal(d["y_test"], np.load(PRED / "y_test.npy").astype(int)):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")
    save_split(d, s)
    proba, info, wall = {}, {}, {}
    t_start = time.time()
    if s == 42:  # anchor boosters and stacking are reused, not refitted (Section 5.1)
        anchor = json.loads((OUT / "seed42_anchor_check.json").read_text(encoding="utf-8"))
        for name in ("XGBoost", "CatBoost", "Stacking Ensemble"):
            proba[name] = np.load(OUT / f"proba_{SUFFIX[name]}_seed42.npy")
        bi = anchor["tree_counts_reported_not_gating"]["best_iteration"]
        counts = anchor["tree_counts_reported_not_gating"]["trees_used_best_iteration_plus_1"]
        info.update({"XGBoost": {"best_iteration": bi["xgboost"]}, "CatBoost": {"best_iteration": bi["catboost"]},
                     "LightGBM (standalone)": {"best_iteration": bi["lightgbm"]},
                     "Stacking Ensemble": {"trees": counts, "reused_from": "anchor_seed42"}})
        wall.update({"reused_from_anchor": anchor["wall_seconds"]})
    else:
        (proba["XGBoost"], bx), wall["XGBoost"] = timed(fit_xgb, d, s)
        (p_lgb, bl), wall["LightGBM (standalone)"] = timed(fit_lgb, d, s)
        (proba["CatBoost"], bc), wall["CatBoost"] = timed(fit_cat, d, s)
        counts = {"xgboost": bx + 1, "lightgbm": bl + 1, "catboost": bc + 1}
        (proba["Stacking Ensemble"], sinfo), wall["Stacking Ensemble"] = timed(fit_stack, d, s, counts)
        np.save(OUT / f"proba_lgbm_standalone_seed{s}.npy", p_lgb)
        info.update({"XGBoost": {"best_iteration": bx}, "CatBoost": {"best_iteration": bc},
                     "LightGBM (standalone)": {"best_iteration": bl, "test_auc": float(roc_auc_score(d["y_test"], p_lgb))},
                     "Stacking Ensemble": {"trees": counts, **sinfo}})
    (proba["Random Forest"], _), wall["Random Forest"] = timed(fit_rf, d, s)
    (proba["FT-Transformer"], info["FT-Transformer"]), wall["FT-Transformer"] = timed(fit_ft, d, s)
    (proba["SAINT-inspired"], info["SAINT-inspired"]), wall["SAINT-inspired"] = timed(
        fit_saint, d, s, restore, diagnostic=(s == 42))
    wall["total_this_run"] = time.time() - t_start
    for name in RANKED:
        np.save(OUT / f"proba_{SUFFIX[name]}_seed{s}.npy", proba[name])
        info.setdefault(name, {})["test_auc"] = float(roc_auc_score(d["y_test"], proba[name]))
    m = manifest()
    m["runs"].append({"run": f"seed{s}", "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
                      "finished_utc": now(), "environment": environment(), "saint_restore": restore,
                      "wall_seconds": wall, "eight_hour_limit_seconds": 28800,
                      "exceeds_eight_hours": (s == 101 and wall["total_this_run"] > 28800), "models": info})
    m["run_order"] = [r["run"] for r in m["runs"]]
    save_manifest(m)
    print(json.dumps({"seed": s, "wall_seconds": wall,
                      "test_auc": {k: info[k]["test_auc"] for k in RANKED},
                      "stacking_trees": info["Stacking Ensemble"]["trees"],
                      "ft": info["FT-Transformer"], "saint": info["SAINT-inspired"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--anchor", action="store_true")
    g.add_argument("--seed", type=int, choices=[42, 101, 202, 303, 404])
    g.add_argument("--summarise", action="store_true")
    p.add_argument("--saint-restore", choices=["final", "best"], default=None,
                   help="required with --seed; see docstring")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert_gpu()
    X, y = load_data(a.data_file)
    if a.anchor:
        if (OUT / "seed42_anchor_check.json").exists():
            raise SystemExit("anchor already run; reruns must be logged explicitly")
        run_anchor(X, y)
    elif a.seed is not None:
        if a.saint_restore != "final":
            raise SystemExit("Amendment 4(a) (d5d795f): Section 5 uses --saint-restore final (as executed)")
        run_seed(X, y, a.seed, a.saint_restore)
    else:
        raise SystemExit("summarise is run after all seeds")


if __name__ == "__main__":
    main()
