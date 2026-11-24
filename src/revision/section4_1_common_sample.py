#!/usr/bin/env python3
"""Protocol Section 4.1: draw the fixed n = 1,000 common XAI sample.

Implements docs/ANALYSIS_PROTOCOL.md Section 4.1 exactly:

    idx_all = np.arange(len(y_test))           # positions in the seed-42 test set
    xai_idx, _ = train_test_split(idx_all, train_size=1000,
                                  stratify=y_test, random_state=42)
    xai_idx = np.sort(xai_idx)
    np.save('revision-results/xai/xai_common_sample_idx.npy', xai_idx)

y_test is the deposited artifacts/predictions/y_test.npy; it is also rebuilt
from the stage-1 CSV with the Section 3 functions and checked for identity.
The SHA-256 of the index file and the sample's class balance are written to
xai_common_sample_manifest.json BEFORE any explainer runs. Every Section 4.2
pipeline and Section 3A(h) read this one file; none draws its own sample.

Run from the repository root:
    python src/revision/section4_1_common_sample.py --data-file <stage-1 CSV>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys

sys.dont_write_bytecode = True

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "s3", ROOT / "src" / "revision" / "section3_retraining_fidelity.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

N_SAMPLE = 1000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revision-results" / "xai")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    idx_path = args.output_dir / "xai_common_sample_idx.npy"
    if idx_path.exists():
        raise SystemExit(f"{idx_path} already exists; the sample is drawn once and never redrawn.")

    y_test = np.load(args.predictions_dir / "y_test.npy").astype(int)
    X, y = s3.build_cohort(args.data_file)
    rebuilt = s3.primary_split(X, y)[8]
    if not np.array_equal(rebuilt, y_test):
        raise SystemExit("SPLIT IDENTITY CHECK FAILED (Section 1.2)")

    # Protocol Section 4.1, verbatim.
    idx_all = np.arange(len(y_test))
    xai_idx, _ = train_test_split(idx_all, train_size=N_SAMPLE, stratify=y_test, random_state=SEED)
    xai_idx = np.sort(xai_idx)
    np.save(idx_path, xai_idx)

    sha = hashlib.sha256(idx_path.read_bytes()).hexdigest()
    ys = y_test[xai_idx]
    manifest = {
        "protocol_section": "4.1",
        "file": "revision-results/xai/xai_common_sample_idx.npy",
        "sha256": sha,
        "n": int(len(xai_idx)),
        "sampling_seed": SEED,
        "method": "sklearn train_test_split(np.arange(len(y_test)), train_size=1000, stratify=y_test, random_state=42); sorted ascending",
        "index_dtype": str(xai_idx.dtype),
        "index_min": int(xai_idx.min()), "index_max": int(xai_idx.max()),
        "indices_unique": bool(len(np.unique(xai_idx)) == len(xai_idx)),
        "indices_sorted": bool(np.all(np.diff(xai_idx) > 0)),
        "positions_refer_to": "rows of the seed-42 primary test set (10,281 records), artifacts/predictions/y_test.npy order",
        "class_balance": {
            "n_positive": int(ys.sum()), "n_negative": int(len(ys) - ys.sum()),
            "prevalence_sample": float(ys.mean()),
            "prevalence_test_set": float(y_test.mean()),
            "test_set_n_positive": int(y_test.sum()), "test_set_n": int(len(y_test)),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scikit-learn": sklearn.__version__, "platform": platform.platform()},
        "rule": "Drawn once, before any Section 4.2 explainer or Section 3A(h) runs; every pipeline reads this file.",
    }
    with open(args.output_dir / "xai_common_sample_manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
