#!/usr/bin/env python3
"""Generate the release SHA-256 manifest in deterministic path order."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance" / "SHA256SUMS.txt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT and ".git" not in path.parts
    )
    OUTPUT.write_text(
        "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    print(f"Wrote {len(paths)} checksums to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
