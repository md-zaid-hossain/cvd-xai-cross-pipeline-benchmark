#!/usr/bin/env python3
"""Protocol Section 3A(c): omnibus ROC test (all AUCs equal).

Implements docs/ANALYSIS_PROTOCOL.md Section 3A(c). theta (AUCs) and S
(DeLong covariance) come from ONE joint call to fast_delong (unchanged,
from src/run_validated_stats.py) on all 15 probability vectors, rows in
MODELS order, records ordered by np.argsort(-y_test) as in delong_test.

Primary: 15 models, contrast L (14 x 15) with row i = e_i - e_15.
Secondary: top-six family; the six models taken in MODELS order, contrast
(5 x 6) row i = e_i - e_6 (chi-square is invariant to the full-rank
contrast chosen). Their covariance is the 6 x 6 submatrix of the joint S.

chi2 = (L theta)' (L S L')^-1 (L theta); df = rank(L S L') from
np.linalg.matrix_rank at its default tolerance. If rank < number of
contrast rows, np.linalg.pinv replaces the inverse (the fallback).

Run from the repository root:
    python src/revision/section3A_c_omnibus_roc.py
Writes: revision-results/lowcompute/omnibus_roc_test.json
"""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_validated_stats import MODELS, fast_delong  # noqa: E402

TOP_SIX = ["Stacking Ensemble", "XGBoost", "CatBoost", "FT-Transformer", "SAINT-inspired", "Random Forest"]


def omnibus(theta: np.ndarray, S: np.ndarray, names: list[str]) -> dict:
    k = len(theta)
    L = np.hstack([np.eye(k - 1), -np.ones((k - 1, 1))])  # row i = e_i - e_k
    M = L @ S @ L.T
    diff = L @ theta
    rank = int(np.linalg.matrix_rank(M))
    full_rank = rank == k - 1
    M_inv = np.linalg.inv(M) if full_rank else np.linalg.pinv(M)
    chi2 = float(diff @ M_inv @ diff)
    df = rank
    eig = np.linalg.eigvalsh(M)
    return {
        "models": names, "n_models": k, "contrast_rows": k - 1,
        "contrast": f"row i = e_i - e_{k} (each model minus '{names[-1]}')",
        "chi2": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df)),
        "numerical_rank_LSLt": rank, "full_rank": full_rank,
        "pseudo_inverse_used": not full_rank,
        "smallest_eigenvalue_LSLt": float(eig.min()), "largest_eigenvalue_LSLt": float(eig.max()),
        "condition_number_LSLt": float(np.linalg.cond(M)),
        "matrix_rank_tolerance": "numpy default (S.max() * max(M.shape) * eps)",
        "aucs": {n: float(t) for n, t in zip(names, theta)},
        "auc_range": float(theta.max() - theta.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "lowcompute")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(args.predictions_dir / "y_test.npy").astype(int)
    names = list(MODELS)
    P = np.vstack([np.load(args.predictions_dir / f"proba_{MODELS[n]}.npy").astype(float).reshape(-1)
                   for n in names])
    order = np.argsort(-y)  # as in delong_test
    theta, S = fast_delong(P[:, order], int(y.sum()))

    full = omnibus(theta, S, names)
    idx6 = [names.index(m) for m in names if m in TOP_SIX]  # top six in MODELS order
    six = omnibus(theta[idx6], S[np.ix_(idx6, idx6)], [names[i] for i in idx6])

    s_eig = np.linalg.eigvalsh(S)
    result = {
        "protocol_section": "3A(c)",
        "h0": "all AUCs in the family are equal",
        "n_test_records": int(len(y)), "n_positive": int(y.sum()),
        "joint_covariance_S_15x15": {
            "numerical_rank": int(np.linalg.matrix_rank(S)),
            "smallest_eigenvalue": float(s_eig.min()),
            "condition_number": float(np.linalg.cond(S)),
        },
        "primary_15_models": full,
        "secondary_top_six": six,
        "prespecified_interpretation": (
            "The 15-model omnibus is expected to reject because low-tier models (e.g. k-NN) are far "
            "from the top tier; rejection says nothing about the top tier. The top-six omnibus is the "
            "test relevant to the non-resolvability claim. Non-rejection there is not evidence of equivalence."
        ),
    }
    with open(args.output_dir / "omnibus_roc_test.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    for label, r in (("PRIMARY: 15 models", full), ("SECONDARY: top six", six)):
        print(f"\n{label}")
        for key in ("chi2", "df", "p", "numerical_rank_LSLt", "contrast_rows", "pseudo_inverse_used",
                    "smallest_eigenvalue_LSLt", "condition_number_LSLt", "auc_range"):
            print(f"  {key:26s} {r[key]}")
    print("\njoint S:", result["joint_covariance_S_15x15"])


if __name__ == "__main__":
    main()
