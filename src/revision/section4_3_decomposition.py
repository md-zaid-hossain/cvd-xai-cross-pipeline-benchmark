#!/usr/bin/env python3
"""Protocol Section 4.3, Amendment 3: decomposition of the concordance change.

Amendment 3 (docs/ANALYSIS_PROTOCOL.md Section 8) was committed and pushed alone
(4fd545d) before this script ran. It separates the two reasons the common-sample
Kendall's W (0.836) differs from the submitted value (0.657):

  (i)   submitted ranks, as published (Table 7)                  -- expect W = 0.657
  (ii)  original unequal-sample ranks, SAINT labels corrected
  (iii) common sample, correct labels (Section 4.3 output)       -- W = 0.836

Label correction (verified, not assumed):
  * SAINT_CVD_Complete.ipynb cell 25 prints one line per matrix column,
    "[ k/13] <label> -> value +- sd", in matrix order. These lines are parsed.
  * <label> comes from `feature_cols` (cell 4, superseded order); the matrix the
    model used was built from `FEATURE_COLS` (cell 10, canonical order). Both
    lists are parsed from the notebook source.
  * The corrected vector assigns each value to FEATURE_COLS[k]. The script checks
    that (a) the printed label at every position equals feature_cols[k],
    (b) the corrected vector re-sorted is the same multiset of values as the
    originally reported sorted table -- only names move, and (c) Table 7's SAINT
    column equals the ranks implied by the mislabelled vector.
  * Independent check: the canonical-order model (deposited corrected checkpoint)
    is evaluated on the first 1,000 rows of the scaled test set, the rows cell 25
    used; its baseline AUC is compared with the notebook's printed 0.8120.

Ranks: scipy.stats.rankdata(-value, method='average'). Kendall's W tie-corrected,
as in section4_3_concordance.py. Reads only; modifies no existing output.

Run from the repository root:
    python src/revision/section4_3_decomposition.py --data-file <stage-1 CSV>
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np  # before torch (duplicate-OpenMP issue in the original environment)
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
XAI = ROOT / "revision-results" / "xai"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conc = _load("conc", ROOT / "src" / "revision" / "section4_3_concordance.py")
pipes = _load("pipes", ROOT / "src" / "revision" / "section4_2_pipelines.py")
s3 = _load("s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")

# Short names in matrix (canonical) order, as used throughout the revision outputs.
SHORT = {"Gender (0: female, 1: male)": "Gender", "age(years)": "Age", "BMI": "BMI", "SBP": "SBP",
         "DBP": "DBP", "CHOL_COL": "Cholesterol", "GLUC_COL": "Glucose",
         "cholesterol(1: normal, 2: above normal, 3: well above normal)": "Cholesterol",
         "Glucose(1: normal, 2: above normal, 3: well above normal)": "Glucose",
         "smoking": "Smoking", "alcohol": "Alcohol", "active": "Active",
         "PP": "PP", "MAP": "MAP", "RiskCount": "RiskCount"}


def parse_list(src: str, name: str) -> list[str]:
    body = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.S).group(1)
    items = [t.strip().strip(",").strip() for t in re.split(r",\s*\n|\n", body)]
    items = [t.strip().strip("'\"") for t in items if t.strip() and not t.strip().startswith("#")]
    return [SHORT[t] for t in items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    args = parser.parse_args()

    nb = json.loads((ROOT / "notebooks" / "core" / "SAINT_CVD_Complete.ipynb").read_text(encoding="utf-8"))
    src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
    printed_order = parse_list(src[4], "feature_cols")     # labels used for printing (cell 4)
    matrix_order = parse_list(src[10], "FEATURE_COLS")     # columns the model actually used (cell 10)
    out25 = "".join("".join(o.get("text", [])) for o in nb["cells"][25].get("outputs", []))
    per_pos = re.findall(r"\[\s*(\d+)/13\]\s+(.+?)\s+→\s+([+-]\d\.\d+)\s+±\s+(\d\.\d+)", out25)
    sorted_tbl = re.findall(r"^\d+\s+(.+?)\s+([+-]\d\.\d+)\s+±\s+(\d\.\d+)\s*$", out25, re.M)
    baseline_printed = float(re.search(r"Baseline AUC:\s*([\d.]+)", out25).group(1))
    assert len(per_pos) == 13 and len(sorted_tbl) == 13, (len(per_pos), len(sorted_tbl))

    rows, label_ok = [], True
    for k, (pos, label, val, sd) in enumerate(per_pos):
        assert int(pos) == k + 1
        label_short = SHORT.get(label.rstrip("."), None) or next(
            (SHORT[key] for key in SHORT if label.rstrip(".") and key.startswith(label.rstrip(". "))), label)
        label_ok &= label_short == printed_order[k]
        rows.append({"position": k + 1, "printed_label": printed_order[k], "actual_feature": matrix_order[k],
                     "label_correct": printed_order[k] == matrix_order[k],
                     "importance": float(val), "sd_printed": float(sd)})
    mapping = pd.DataFrame(rows)

    vals = mapping["importance"].values
    permutation_check = sorted(vals.tolist(), reverse=True) == sorted([float(v) for _, v, _ in sorted_tbl], reverse=True)

    wrong = pd.Series(vals, index=mapping["printed_label"])     # as published
    right = pd.Series(vals, index=mapping["actual_feature"])    # corrected
    FEAT = conc.FEAT
    rank_wrong = pd.Series(stats.rankdata(-wrong.loc[FEAT].values, method="average"), index=FEAT)
    rank_right = pd.Series(stats.rankdata(-right.loc[FEAT].values, method="average"), index=FEAT)
    table7 = pd.DataFrame(conc.ORIGINAL, index=conc.LABELS).T.loc[FEAT]
    table7_matches_mislabelled = bool((table7["SAINT-inspired Perm."].values == rank_wrong.values).all())

    # Independent check: canonical-order model, first 1,000 scaled test rows (cell 25's rows).
    import torch
    model, ck = pipes.saint_model()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    X, y = s3.build_cohort(args.data_file)
    parts = s3.primary_split(X, y)
    Xs = ck["scaler"].transform(parts[2].astype(np.float32)).astype(np.float32)[:1000]
    with torch.no_grad():
        p = np.concatenate([torch.softmax(model(torch.tensor(Xs[s:s + 256], device=dev)), 1)[:, 1].cpu().numpy()
                            for s in range(0, 1000, 256)])
    baseline_recomputed = float(roc_auc_score(parts[8][:1000], p))

    cases = {}
    r_i = table7.astype(float)
    r_ii = r_i.copy(); r_ii["SAINT-inspired Perm."] = rank_right.values
    r_iii = pd.read_csv(XAI / "rank_matrix_common_sample.csv").set_index("feature").loc[FEAT, conc.LABELS]
    for key, R in (("i_submitted", r_i), ("ii_original_saint_corrected", r_ii), ("iii_common_sample", r_iii)):
        kw = conc.kendall_w(R.values)
        sp = conc.spearman_matrix(R)
        cases[key] = {"kendall_w": kw,
                      "spearman_pairs": {f"{a} | {b}": float(sp.loc[a, b]) for a, b in itertools.combinations(conc.LABELS, 2)},
                      "ranks": {c: R[c].to_dict() for c in R.columns}}

    W = {k: v["kendall_w"]["W"] for k, v in cases.items()}
    total = W["iii_common_sample"] - W["i_submitted"]
    out = {
        "protocol_section": "4.3, Amendment 3 (committed and pushed alone in 4fd545d before this run)",
        "verification": {
            "printed_label_at_each_position_equals_cell4_feature_cols": label_ok,
            "corrected_vector_is_permutation_of_reported_values_only_names_moved": permutation_check,
            "submitted_table7_saint_column_equals_ranks_of_mislabelled_vector": table7_matches_mislabelled,
            "positions_mislabelled": mapping.loc[~mapping["label_correct"], "position"].tolist(),
            "baseline_auc_printed_in_cell25": baseline_printed,
            "baseline_auc_recomputed_canonical_model_first_1000_test_rows": baseline_recomputed,
        },
        "kendall_w": W,
        "decomposition_sequential_i_to_ii_to_iii": {
            "total_change": total,
            "label_correction_i_to_ii": W["ii_original_saint_corrected"] - W["i_submitted"],
            "common_sample_ii_to_iii": W["iii_common_sample"] - W["ii_original_saint_corrected"],
            "label_correction_share": (W["ii_original_saint_corrected"] - W["i_submitted"]) / total,
            "common_sample_share": (W["iii_common_sample"] - W["ii_original_saint_corrected"]) / total,
            "note": "Sequential decomposition (label correction first). The reverse order is undefined: the common-sample ranks were never mislabelled.",
        },
        "cases": cases,
    }
    with open(XAI / "concordance_decomposition.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    mapping["rank_as_published"] = mapping["printed_label"].map(rank_wrong)
    mapping["rank_corrected"] = mapping["actual_feature"].map(rank_right)
    mapping.to_csv(XAI / "saint_label_correction.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 200)
    print(mapping.to_string(index=False))
    print(json.dumps(out["verification"], indent=2))
    for k, v in cases.items():
        kw = v["kendall_w"]
        print(f"\n{k}: W = {kw['W']:.4f}, chi2({kw['df']}) = {kw['chi2']:.2f}, p = {kw['p']:.3g}")
        print(pd.Series(v["spearman_pairs"]).round(3).to_string())
    print("\n" + json.dumps(out["decomposition_sequential_i_to_ii_to_iii"], indent=2))


if __name__ == "__main__":
    main()
