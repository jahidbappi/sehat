"""Pytest bootstrap.

The canonical setup is ``pip install -e .``; this path shim lets contributors
run the suite straight from a fresh checkout (``pytest`` with no install),
matching how the stdlib ``unittest`` runner is used in offline environments.
"""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
