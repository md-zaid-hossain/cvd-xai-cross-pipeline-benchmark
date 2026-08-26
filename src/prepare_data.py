#!/usr/bin/env python3
"""Reproduce the two-stage cohort cleaning used by the CVD benchmark.

The source dataset is not redistributed in this repository. Download it from
the URL documented in ``data/README.md`` and provide the converted 11-column
CSV used by the executed notebooks.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


SBP = "ap_hi(Systolic blood pressure)"
DBP = "ap_lo(Diastolic blood pressure)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Converted 70,000-row CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/local"), help="Output directory"
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = {SBP, DBP, "BMI"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != 70_000:
        raise ValueError(f"Expected 70,000 source rows, found {len(frame):,}")

    stage1 = frame[frame[SBP].between(80, 200) & frame[DBP].between(40, 130)].copy()
    stage2 = stage1[stage1["BMI"].between(10, 60) & ((stage1[SBP] - stage1[DBP]) > 0)].copy()
    stage1 = stage1.reset_index(drop=True)
    stage2 = stage2.reset_index(drop=True)

    if len(stage1) != 68_653 or len(stage2) != 68_534:
        raise RuntimeError(
            "Cleaning counts do not match the executed analysis: "
            f"stage 1={len(stage1):,}, final={len(stage2):,}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate = args.output_dir / "Heart_disease_clean_stage1.csv"
    final = args.output_dir / "Heart_disease_clean_final.csv"
    stage1.to_csv(intermediate, index=False)
    stage2.to_csv(final, index=False)

    print(f"Source:       {len(frame):,} rows")
    print(f"Stage 1:      {len(stage1):,} rows -> {intermediate}")
    print(f"Final cohort: {len(stage2):,} rows -> {final}")
    print(f"Stage-1 SHA256: {sha256(intermediate)}")
    print(f"Final SHA256:   {sha256(final)}")


if __name__ == "__main__":
    main()
