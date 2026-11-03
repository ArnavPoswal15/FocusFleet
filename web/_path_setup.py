"""Ensure repo-root or Space-root `focusfleet` package is importable."""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent

for candidate in (_APP_DIR, _REPO_ROOT):
    if (candidate / "focusfleet").is_dir():
        root = str(candidate)
        if root not in sys.path:
            sys.path.insert(0, root)
        break
