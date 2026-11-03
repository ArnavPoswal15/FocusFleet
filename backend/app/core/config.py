import os
from pathlib import Path

from app.path_setup import WEB_DIR

# Stable default for local dev so restarts do not invalidate session cookies.
# Set FOCUSFLEET_SECRET_KEY in production.
SECRET_KEY = os.getenv(
    "FOCUSFLEET_SECRET_KEY",
    "focusfleet-dev-secret-do-not-use-in-production",
)
SESSION_COOKIE_NAME = "focusfleet_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7
DATA_DIR = Path(os.getenv("FOCUSFLEET_DATA_DIR", str(WEB_DIR))).expanduser()
