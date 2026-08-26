# Data access and cohort construction

## Source

The study uses the Cardiovascular Disease Dataset published by Svetlana Ulianova on Kaggle:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Citation used in the manuscript:

> Ulianova, S. (2019). *Cardiovascular disease dataset* [Data set]. Kaggle.

## Why the CSVs are not included

The source page available during packaging identified the dataset but did not expose a redistribution licence that could be verified. To avoid granting or implying rights the authors may not hold, the public release contains the processing code, schema, expected counts, and hashes but not the patient-level CSV files. If the dataset owner or Kaggle record later confirms redistribution permission, the authors may add the files in a new version with the source licence preserved.

## Expected local files

Place local files under `data/local/` (ignored by Git):

- `Heart_disease.csv`: converted 70,000-row, 11-column study input.
- `Heart_disease_clean_stage1.csv`: 68,653 rows after SBP/DBP plausibility bounds.
- `Heart_disease_clean_final.csv`: 68,534 rows after BMI and positive pulse-pressure checks.

The executed notebooks load the stage-1 filename and then perform the second check internally. The portable scripts accept `--data-file` and default to the stage-1 file.

## Cleaning rules

1. Retain SBP from 80 to 200 mmHg, inclusive.
2. Retain DBP from 40 to 130 mmHg, inclusive.
3. Retain BMI from 10 to 60 kg/m2, inclusive.
4. Retain positive pulse pressure (`SBP - DBP > 0`).

Run `python src/prepare_data.py path/to/Heart_disease.csv` and verify the counts shown above.

## Known supplied-file SHA-256 values

These hashes describe the files used to build the attached analysis package; they allow the authors to confirm a local copy without redistributing it.

- Converted 70,000-row file: `7d0ada2eb8f6b6ca6c2cfc09c2c6ae0d6f53e89f66fcfc05c604198f402c0ef5`
- Stage-1 68,653-row file: `8b1322989836cf85444ba7b693c28cd9a43188730b380cbe301da75d6569eb5f`

Both values above were re-confirmed against the supplied source files on 2026-08-26.

### Expected hash behaviour when you re-run the cleaning

`src/prepare_data.py` prints the SHA-256 of the files it writes. **These will not match
the stage-1 hash above, and that is expected.** The supplied stage-1 CSV and a freshly
generated one contain identical data — verified row-for-row and column-for-column with
`pandas.testing.assert_frame_equal` — but pandas re-serializes floating-point and quoting
detail when it writes, so the bytes differ.

Verify the cleaning by the **row counts** (`70,000 -> 68,653 -> 68,534`), which
`prepare_data.py` asserts and fails on, rather than by comparing the output hash to the
value above. Use the hashes above only to confirm that a local copy of a *supplied* file
is the same file the authors used.
