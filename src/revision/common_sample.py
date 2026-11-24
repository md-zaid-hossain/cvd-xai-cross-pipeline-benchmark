"""Shared loader for the Section 4.1 common sample (used by Section 3A(h) and Section 4.2).

Reads revision-results/xai/xai_common_sample_idx.npy and refuses to return it
unless its SHA-256 matches xai_common_sample_manifest.json (Section 4.1:
"All five pipelines read this one file. No pipeline draws its own sample.").
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XAI_DIR = ROOT / "revision-results" / "xai"


def load_common_sample() -> tuple[np.ndarray, str]:
    idx_path = XAI_DIR / "xai_common_sample_idx.npy"
    manifest = json.loads((XAI_DIR / "xai_common_sample_manifest.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(idx_path.read_bytes()).hexdigest()
    if sha != manifest["sha256"]:
        raise SystemExit(f"COMMON SAMPLE HASH MISMATCH: {sha} != manifest {manifest['sha256']}")
    idx = np.load(idx_path)
    if len(idx) != 1000 or not np.all(np.diff(idx) > 0):
        raise SystemExit("common sample is not 1,000 sorted unique positions")
    return idx, sha
