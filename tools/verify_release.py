#!/usr/bin/env python3
"""Validate repository structure, notebook execution state, arrays, and metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_SUFFIX = {
    "Decision Tree": "dt", "Naive Bayes": "nb", "k-NN": "knn",
    "Logistic Regression": "lr", "Random Forest": "rf", "SVM (RBF)": "svm",
    "MLP": "mlp", "XGBoost": "xgb", "LightGBM": "lgbm", "CatBoost": "cat",
    "Stacking Ensemble": "stack_corrected", "TabNet": "tabnet",
    "FT-Transformer": "ft", "VIME-inspired SSL": "vime", "SAINT-inspired": "saint",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    failures: list[str] = []
    required = [
        "README.md", "CITATION.cff", ".zenodo.json", "LICENSE.md",
        "results/tables/validated_primary_metrics.csv", "artifacts/predictions/y_test.npy",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    for notebook in sorted((ROOT / "notebooks" / "core").glob("*.ipynb")):
        document = json.loads(notebook.read_text(encoding="utf-8"))
        code_cells = [cell for cell in document.get("cells", []) if cell.get("cell_type") == "code"]
        stored_errors = [
            (index, output.get("ename"), output.get("evalue"))
            for index, cell in enumerate(code_cells)
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if stored_errors:
            failures.append(f"{notebook.name}: stored errors {stored_errors[:3]}")
        if not any(cell.get("execution_count") is not None for cell in code_cells):
            failures.append(f"{notebook.name}: no executed code cells")

    prediction_dir = ROOT / "artifacts" / "predictions"
    y_true = np.load(prediction_dir / "y_test.npy").astype(int)
    if len(y_true) != 10_281 or set(np.unique(y_true)) != {0, 1}:
        failures.append("y_test.npy is not the expected binary 10,281-record test set")

    expected = pd.read_csv(ROOT / "results" / "tables" / "validated_primary_metrics.csv").set_index("Model")
    for model, suffix in MODEL_SUFFIX.items():
        proba = np.load(prediction_dir / f"proba_{suffix}.npy").astype(float).reshape(-1)
        pred = np.load(prediction_dir / f"pred_{suffix}.npy").astype(int).reshape(-1)
        if len(proba) != len(y_true) or len(pred) != len(y_true):
            failures.append(f"{model}: array length mismatch")
            continue
        if not np.isfinite(proba).all() or proba.min() < 0 or proba.max() > 1:
            failures.append(f"{model}: invalid probability values")
        actual = {
            "AUC": roc_auc_score(y_true, proba), "Accuracy": accuracy_score(y_true, pred),
            "F1": f1_score(y_true, pred), "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred), "Brier": brier_score_loss(y_true, proba),
        }
        for column, value in actual.items():
            if abs(value - float(expected.loc[model, column])) > 1e-12:
                failures.append(f"{model}: {column} differs from the published table")

    manifest = ROOT / "provenance" / "SHA256SUMS.txt"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected_hash, relative = line.split("  ", 1)
            target = ROOT / relative
            if not target.is_file() or sha256(target) != expected_hash:
                failures.append(f"checksum mismatch: {relative}")
    else:
        failures.append("missing checksum manifest")

    if failures:
        print("RELEASE VALIDATION: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("RELEASE VALIDATION: PASS")
    print(f"- {len(list((ROOT / 'notebooks' / 'core').glob('*.ipynb')))} core notebooks are readable and error-free")
    print(f"- {len(MODEL_SUFFIX)} model outputs reconcile with validated_primary_metrics.csv")
    print(f"- Held-out test set: {len(y_true):,} records")


if __name__ == "__main__":
    main()
