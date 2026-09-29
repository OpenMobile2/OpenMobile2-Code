"""Thin wrapper: reuse MobileWorld/AndroidWorld process_refine.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE.parent / "MobileWorld" / "process_refine.py",
    _HERE.parent / "AndroidWorld" / "process_refine.py",
]


def main() -> None:
    for path in _CANDIDATES:
        if path.exists():
            sys.argv[0] = str(path)
            runpy.run_path(str(path), run_name="__main__")
            return
    raise SystemExit(
        "process_refine.py not found under MobileWorld/ or AndroidWorld/"
    )


if __name__ == "__main__":
    main()
