"""
Shared runtime state.

The video stream callback runs at full webcam FPS; the status banner is
polled by a 1 Hz Timer. Isolating the shared, mutable state in one module
(instead of scattering `global` statements across files) makes the
data flow between those two loops explicit and easy to trace.
"""

import datetime
from dataclasses import dataclass, field

from . import config


@dataclass
class AppState:
    log_file: str | None = None
    session_id: str | None = None
    tracker: object | None = None  # DriverStateTracker, set on login

    drowsy_since: float | None = None
    current_status_key: str = config.STATUS_NEUTRAL


STATE = AppState()


def write_log(message: str) -> None:
    """Append a timestamped line to the active session's log file, if any."""
    if not STATE.log_file:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with open(STATE.log_file, "a") as f:
        f.write(line + "\n")
    print(line)
