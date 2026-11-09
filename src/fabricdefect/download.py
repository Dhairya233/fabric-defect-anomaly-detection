"""Verify the official archives against known checksums and extract them."""

from __future__ import annotations

import hashlib
from pathlib import Path

import py7zr

from .dataset import ARCHIVES


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_archives(archive_dir: Path, expected: dict[str, str] | None = None) -> list[str]:
    """Return a list of problems (empty list means every archive is present and matches)."""
    expected = ARCHIVES if expected is None else expected
    problems = []
    for name, digest in expected.items():
        path = archive_dir / name
        if not path.is_file():
            problems.append(f"missing: {path}")
        elif sha256_of(path) != digest:
            problems.append(f"checksum mismatch: {path}")
    return problems


def extract_archives(archive_dir: Path, raw_dir: Path, names: list[str] | None = None) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names or list(ARCHIVES):
        with py7zr.SevenZipFile(archive_dir / name) as z:
            z.extractall(path=raw_dir)
