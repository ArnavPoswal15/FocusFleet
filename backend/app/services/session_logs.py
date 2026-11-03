import os
import re

_RE_SESSION_DURATION = re.compile(r"Session Duration:\s*([\d.]+)")
_RE_DROWSY_EVENTS = re.compile(r"Total Drowsy Events:\s*(\d+)")
_RE_YAWN_EVENTS = re.compile(r"Total Yawn Events:\s*(\d+)")

SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")

MAX_LOG_BYTES = 512 * 1024


def parse_session_log(text: str) -> tuple[float | None, int | None, int | None]:
    """Parse summary lines from session_log.txt (handles timestamp prefixes)."""
    duration: float | None = None
    drowsy_events: int | None = None
    yawns: int | None = None
    for line in text.splitlines():
        if duration is None and (m := _RE_SESSION_DURATION.search(line)):
            duration = float(m.group(1))
        if drowsy_events is None and (m := _RE_DROWSY_EVENTS.search(line)):
            drowsy_events = int(m.group(1))
        if yawns is None and (m := _RE_YAWN_EVENTS.search(line)):
            yawns = int(m.group(1))
    return duration, drowsy_events, yawns


def driver_logs_base(driver_name: str) -> str:
    if not driver_name or ".." in driver_name or "/" in driver_name or "\\" in driver_name:
        raise ValueError("Invalid driver name")
    return os.path.join("logs", driver_name)


def session_log_path(driver_name: str, session_id: str) -> str:
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("Invalid session id")
    base = driver_logs_base(driver_name)
    path = os.path.join(base, session_id, "session_log.txt")
    resolved = os.path.realpath(path)
    logs_root = os.path.realpath("logs")
    if not resolved.startswith(logs_root + os.sep):
        raise ValueError("Invalid log path")
    return path


def list_driver_sessions(driver_name: str, active_session_id: str | None = None) -> list[dict]:
    base = driver_logs_base(driver_name)
    if not os.path.isdir(base):
        return []

    sessions: list[dict] = []
    for entry in sorted(os.listdir(base), reverse=True):
        path = os.path.join(base, entry)
        if not os.path.isdir(path):
            continue
        if not SESSION_ID_PATTERN.match(entry):
            continue
        log_path = os.path.join(path, "session_log.txt")
        sessions.append(
            {
                "session_id": entry,
                "log_file": log_path if os.path.isfile(log_path) else None,
                "active": entry == active_session_id,
            }
        )
    return sessions


def read_log_file(path: str, tail: int | None = None) -> str:
    if not os.path.isfile(path):
        return ""
    size = os.path.getsize(path)
    if tail is not None and tail > 0:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        return "".join(lines[-tail:])

    if size > MAX_LOG_BYTES:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            fh.seek(max(0, size - MAX_LOG_BYTES))
            if fh.tell() > 0:
                fh.readline()
            return fh.read()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def aggregate_driver_logs(driver_name: str) -> dict:
    base = driver_logs_base(driver_name)
    if not os.path.isdir(base):
        return {
            "session_count": 0,
            "total_drowsy_events": 0,
            "total_yawns": 0,
            "last_session_duration_s": None,
        }

    session_dirs = [
        d
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and SESSION_ID_PATTERN.match(d)
    ]
    total_drowsy = 0
    total_yawns = 0
    last_duration = None

    for session_id in sorted(session_dirs, reverse=True):
        log_path = os.path.join(base, session_id, "session_log.txt")
        if not os.path.isfile(log_path):
            continue
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        duration, drowsy_events, yawns = parse_session_log(text)
        if drowsy_events is not None:
            total_drowsy += drowsy_events
        if yawns is not None:
            total_yawns += yawns
        if last_duration is None and duration is not None:
            last_duration = duration
            break

    return {
        "session_count": len(session_dirs),
        "total_drowsy_events": total_drowsy,
        "total_yawns": total_yawns,
        "last_session_duration_s": last_duration,
    }
