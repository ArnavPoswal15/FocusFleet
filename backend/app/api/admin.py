import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import require_admin
from app.services.session_logs import (
    MAX_LOG_BYTES,
    aggregate_driver_logs,
    list_driver_sessions,
    read_log_file,
    session_log_path,
)
from focusfleet import config
from focusfleet.database import ROLE_DRIVER, driver_role_from_row
from focusfleet.state import STATE

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _driver_row(driver_id: int) -> tuple:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute("SELECT * FROM Driver WHERE DriverID=?", (driver_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")
    if driver_role_from_row(row) != ROLE_DRIVER:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")
    return row


def _driver_public(row: tuple) -> dict:
    return {
        "driver_id": row[0],
        "name": row[1],
        "contact_info": row[3],
        "license_number": row[4],
    }


@router.get("/drivers")
def list_drivers(_admin: dict = Depends(require_admin)) -> dict:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        rows = conn.execute(
            "SELECT DriverID, Name, ContactInfo, LicenseNumber FROM Driver "
            "WHERE Role=? OR Role IS NULL ORDER BY Name",
            (ROLE_DRIVER,),
        ).fetchall()
    finally:
        conn.close()

    drivers = []
    for row in rows:
        stats = aggregate_driver_logs(row[1])
        drivers.append(
            {
                "driver_id": row[0],
                "name": row[1],
                "contact_info": row[2],
                "license_number": row[3],
                "session_count": stats["session_count"],
            }
        )
    return {"drivers": drivers}


@router.get("/drivers/{driver_id}")
def get_driver(driver_id: int, _admin: dict = Depends(require_admin)) -> dict:
    row = _driver_row(driver_id)
    stats = aggregate_driver_logs(row[1])
    return {**_driver_public(row), **stats}


@router.get("/drivers/{driver_id}/sessions")
def get_driver_sessions(driver_id: int, _admin: dict = Depends(require_admin)) -> dict:
    row = _driver_row(driver_id)
    sessions = list_driver_sessions(row[1], active_session_id=STATE.session_id)
    return {"sessions": sessions}


@router.get("/drivers/{driver_id}/sessions/{session_id}/log")
def get_session_log(
    driver_id: int,
    session_id: str,
    tail: int | None = Query(default=None, ge=1, le=5000),
    _admin: dict = Depends(require_admin),
) -> dict:
    row = _driver_row(driver_id)
    try:
        path = session_log_path(row[1], session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not os.path.isfile(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")

    content = read_log_file(path, tail=tail)
    truncated = tail is None and os.path.getsize(path) > MAX_LOG_BYTES
    return {
        "session_id": session_id,
        "content": content,
        "truncated": truncated,
    }


@router.get("/fleet/summary")
def fleet_summary(_admin: dict = Depends(require_admin)) -> dict:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        rows = conn.execute(
            "SELECT Name FROM Driver WHERE Role=? OR Role IS NULL",
            (ROLE_DRIVER,),
        ).fetchall()
    finally:
        conn.close()

    driver_count = len(rows)
    session_count = 0
    total_drowsy = 0
    total_yawns = 0

    for (name,) in rows:
        stats = aggregate_driver_logs(name)
        session_count += stats["session_count"]
        total_drowsy += stats["total_drowsy_events"]
        total_yawns += stats["total_yawns"]

    return {
        "driver_count": driver_count,
        "session_count": session_count,
        "total_drowsy_events": total_drowsy,
        "total_yawns": total_yawns,
    }
