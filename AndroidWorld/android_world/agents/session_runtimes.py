"""Compatibility shim. Runtime names live in ``runtimes.registry``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtimes.registry import *  # noqa: F403
