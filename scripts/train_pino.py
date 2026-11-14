"""Standalone launcher script for PINO training on UKEA 32x32 flood data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crestix_pino.training.train import main  # noqa: E402


if __name__ == "__main__":
    main()
