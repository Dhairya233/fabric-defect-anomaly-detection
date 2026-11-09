"""Small helpers shared by the command-line scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def require(path: str | Path, hint: str) -> Path:
    """Exit with a plain message (not a traceback) when a prerequisite file is missing."""
    path = Path(path)
    if not path.exists():
        print(f"error: {path} not found. {hint}", file=sys.stderr)
        raise SystemExit(1)
    return path
