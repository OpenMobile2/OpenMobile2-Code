"""Compatibility shim. Prompt text lives in ``runtimes.prompts``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtimes.prompts import *  # noqa: F403
