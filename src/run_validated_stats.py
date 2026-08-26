#!/usr/bin/env python3
"""Recompute primary metrics, calibration, paired DeLong tests, and figures."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODELS = {
    "Decision Tree": "dt",
    "Naive Bayes": "nb",
    "k-NN": "knn",
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "SVM (RBF)": "svm",
    "MLP": "mlp",
    "XGBoost": "xgb",
    "LightGBM": "lgbm",
    "CatBoost": "cat",
    "Stacking Ensemble": "stack_corrected",
    "TabNet": "tabnet",
    "FT-Transformer": "ft",
    "VIME-inspired SSL": "vime",
    "SAINT-inspired": "saint",
}


def fdr_bh(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values without an extra dependency."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Holm family-wise adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks + 1.0
    return result


def fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    positives = predictions[:, :positive_count]
    negatives = predictions[:, positive_count:]
    model_count, negative_count = predictions.shape[0], negatives.shape[1]
    tx = np.empty_like(positives, dtype=float)
    ty = np.empty_like(negatives, dtype=float)
    tz = np.empty_like(predictions, dtype=float)
    for row in range(model_count):
        tx[row] = compute_midrank(positives[row])
        ty[row] = compute_midrank(negatives[row])
        tz[row] = compute_midrank(predictions[row])
    aucs = tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.atleast_2d(np.cov(v01, bias=False)) / positive_count
    covariance += np.atleast_2d(np.cov(v10, bias=False)) / negative_count
    return aucs, covariance


def delong_test(y_true: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float, float, float, float]:
    order = np.argsort(-y_true)
    aucs, covariance = fast_delong(np.vstack([first, second])[:, order], int(y_true.sum()))
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T)[0, 0])
    difference = float(aucs[0] - aucs[1])
    if variance <= 0 or not np.isfinite(variance):
        p_value = 1.0 if abs(difference) < 1e-15 else np.nan
        return aucs[0], aucs[1], np.nan, p_value, variance
    z_score = difference / math.sqrt(variance)
    return aucs[0], aucs[1], z_score, 2 * stats.norm.sf(abs(z_score)), variance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("artifacts/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced-results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.load(args.predictions_dir / "y_test.npy").astype(int)
    probabilities = {
        name: np.load(args.predictions_dir / f"proba_{suffix}.npy").astype(float).reshape(-1)
        for name, suffix in MODELS.items()
    }
    predictions = {
        name: np.load(args.predictions_dir / f"pred_{suffix}.npy").astype(int).reshape(-1)
        for name, suffix in MODELS.items()
    }
    if not all(len(values) == len(y_true) for values in probabilities.values()):
        raise ValueError("Prediction arrays do not share the held-out test-set length")

    metric_rows = []
    for name in MODELS:
        proba, pred = probabilities[name], predictions[name]
        metric_rows.append(
            {
                "Model": name,
                "AUC": roc_auc_score(y_true, proba),
                "Accuracy": accuracy_score(y_true, pred),
                "F1": f1_score(y_true, pred),
                "Precision": precision_score(y_true, pred),
                "Recall": recall_score(y_true, pred),
                "Brier": brier_score_loss(y_true, proba),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("AUC", ascending=False)
    metrics.to_csv(args.output_dir / "validated_primary_metrics.csv", index=False)

    pair_rows = []
    for first, second in itertools.combinations(MODELS, 2):
        auc1, auc2, z_score, p_value, variance = delong_test(
            y_true, probabilities[first], probabilities[second]
        )
        pair_rows.append(
            {
                "Model A": first,
                "Model B": second,
                "AUC A": auc1,
                "AUC B": auc2,
                "Delta AUC": auc1 - auc2,
                "z": z_score,
                "p": p_value,
                "variance_difference": variance,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    valid = pairs["p"].notna()
    pairs.loc[valid, "BH_q_all105"] = fdr_bh(pairs.loc[valid, "p"])
    pairs.loc[valid, "Holm_p_all105"] = holm_adjust(pairs.loc[valid, "p"])
    pairs.to_csv(args.output_dir / "validated_delong_all_pairs.csv", index=False)

    advanced = [
        "XGBoost", "LightGBM", "CatBoost", "Stacking Ensemble", "TabNet",
        "FT-Transformer", "VIME-inspired SSL", "SAINT-inspired", "Random Forest",
    ]
    versus_mlp = []
    for name in advanced:
        auc1, auc2, z_score, p_value, _ = delong_test(y_true, probabilities[name], probabilities["MLP"])
        versus_mlp.append(
            {"Model vs MLP": name, "AUC model": auc1, "AUC MLP": auc2,
             "Delta AUC": auc1 - auc2, "z": z_score, "p": p_value}
        )
    mlp_frame = pd.DataFrame(versus_mlp)
    mlp_frame["BH_q"] = fdr_bh(mlp_frame["p"])
    mlp_frame["BH_significant_0.05"] = mlp_frame["BH_q"] < 0.05
    mlp_frame.to_csv(args.output_dir / "validated_delong_vs_mlp_bh.csv", index=False)

    ranked = sorted(MODELS, key=lambda name: roc_auc_score(y_true, probabilities[name]), reverse=True)
    top_six = ranked[:6]
    top_rows = []
    for first, second in itertools.combinations(top_six, 2):
        auc1, auc2, z_score, p_value, _ = delong_test(y_true, probabilities[first], probabilities[second])
        top_rows.append(
            {"Model A": first, "Model B": second, "Delta AUC": auc1 - auc2,
             "z": z_score, "p": p_value}
        )
    top_frame = pd.DataFrame(top_rows)
    top_frame["BH_q"] = fdr_bh(top_frame["p"])
    top_frame["BH_significant_0.05"] = top_frame["BH_q"] < 0.05
    top_frame.to_csv(args.output_dir / "validated_delong_top6_bh.csv", index=False)

    calibration_rows, curves = [], {}
    for name, proba in probabilities.items():
        clipped = np.clip(proba, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000).fit(logit, y_true)
        observed, mean_predicted = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        curves[name] = (mean_predicted, observed)
        calibration_rows.append(
            {"Model": name, "AUC": roc_auc_score(y_true, proba),
             "Brier": brier_score_loss(y_true, proba),
             "Calibration intercept": float(model.intercept_[0]),
             "Calibration slope": float(model.coef_[0, 0])}
        )
    calibration = pd.DataFrame(calibration_rows).sort_values("AUC", ascending=False)
    calibration.to_csv(args.output_dir / "exact_primary_calibration.csv", index=False)

    plt.figure(figsize=(8, 7))
    for name in top_six:
        mean_predicted, observed = curves[name]
        plt.plot(mean_predicted, observed, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed proportion")
    plt.title("Calibration curves for the six highest-AUC primary models")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_dir / "exact_primary_calibration_top6.png", dpi=300)
    plt.close()

    matrix = np.ones((len(ranked), len(ranked)))
    for _, row in pairs.iterrows():
        i, j = ranked.index(row["Model A"]), ranked.index(row["Model B"])
        matrix[i, j] = matrix[j, i] = row["p"]
    plt.figure(figsize=(12, 10))
    image = plt.imshow(
        matrix, cmap="viridis_r",
        norm=colors.LogNorm(vmin=max(np.nanmin(matrix[matrix > 0]), 1e-6), vmax=1),
    )
    plt.colorbar(image, label="DeLong p-value (log scale)")
    plt.xticks(range(len(ranked)), ranked, rotation=90, fontsize=8)
    plt.yticks(range(len(ranked)), ranked, fontsize=8)
    plt.title("Validated pairwise DeLong tests (midrank implementation)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "validated_delong_heatmap.png", dpi=300)
    plt.close()

    print(f"Validated {len(y_true):,} held-out observations across {len(MODELS)} models")
    print(f"Top six: {', '.join(top_six)}")
    print(metrics.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
