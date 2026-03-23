"""Earth-to-Jupiter search helpers for the JOI phase."""

from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))
