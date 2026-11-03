"""Add monorepo root to PYTHONPATH for `focusfleet` imports."""

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent

if (_REPO_ROOT / "focusfleet").is_dir():
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

REPO_ROOT = _REPO_ROOT
WEB_DIR = Path(os.getenv("FOCUSFLEET_DATA_DIR", str(_REPO_ROOT / "web")))
STATIC_DIR = Path(os.getenv("FOCUSFLEET_STATIC_DIR", str(_REPO_ROOT / "out")))
