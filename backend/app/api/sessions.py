import datetime
import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.deps import require_driver
from app.services.session_logs import list_driver_sessions
from focusfleet import config
from focusfleet.session_logger import DriverStateTracker
from focusfleet.state import STATE

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class StartSessionBody(BaseModel):
    ear_threshold: float = Field(default=config.DEFAULT_EAR_THRESHOLD, ge=config.EAR_THRESHOLD_MIN, le=config.EAR_THRESHOLD_MAX)


def _driver_name(driver_id: int) -> str:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute("SELECT Name FROM Driver WHERE DriverID=?", (driver_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return row[0]


@router.post("", status_code=status.HTTP_201_CREATED)
def start_session(body: StartSessionBody, user: dict = Depends(require_driver)) -> dict:
    if STATE.tracker is not None and STATE.session_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An active monitoring session already exists",
        )

    name = _driver_name(user["driver_id"])
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join("logs", name, session_id)
    os.makedirs(logs_dir, exist_ok=True)

    STATE.log_file = os.path.join(logs_dir, "session_log.txt")
    STATE.session_id = session_id
    STATE.tracker = DriverStateTracker(STATE.log_file)
    STATE.tracker.ear_threshold = body.ear_threshold

    return {
        "session_id": session_id,
        "log_file": STATE.log_file,
        "ear_threshold": body.ear_threshold,
    }


@router.post("/{session_id}/stop")
def stop_session(session_id: str, user: dict = Depends(require_driver)) -> dict:
    if STATE.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not active")

    if STATE.tracker is not None:
        STATE.tracker.end_session()
    ended = session_id
    STATE.tracker = None
    STATE.session_id = None
    STATE.log_file = None
    STATE.current_status_key = config.STATUS_NEUTRAL
    STATE.drowsy_since = None

    return {"session_id": ended, "status": "stopped"}


@router.get("")
def list_sessions(user: dict = Depends(require_driver)) -> dict:
    name = _driver_name(user["driver_id"])
    sessions = list_driver_sessions(name, active_session_id=STATE.session_id)
    return {"sessions": sessions}
