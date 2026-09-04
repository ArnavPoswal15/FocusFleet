"""SQLite-backed driver accounts."""

import datetime
import os
import sqlite3

from . import config
from .state import STATE
from logger import DriverStateTracker  # logger.py at project root


def init_db() -> None:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Driver (
                DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Password TEXT NOT NULL,
                ContactInfo TEXT,
                LicenseNumber TEXT
            )
        ''')
        conn.commit()
    finally:
        conn.close()


def login_driver(name: str, password: str):
    """Authenticate a driver and start a new logging session.

    Returns (driver_info_dict | None, message).
    """
    if not name.strip() or not password.strip():
        return None, "Please enter both name and password."

    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute(
            "SELECT * FROM Driver WHERE Name=? AND Password=?", (name, password)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None, "Invalid credentials."

    driver_info = {
        "DriverID": row[0],
        "Name": row[1],
        "ContactInfo": row[3],
        "LicenseNumber": row[4],
    }

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join("logs", driver_info["Name"], session_id)
    os.makedirs(logs_dir, exist_ok=True)

    STATE.log_file = os.path.join(logs_dir, "session_log.txt")
    STATE.session_id = session_id
    STATE.tracker = DriverStateTracker(STATE.log_file)

    return driver_info, f"Welcome, {row[1]}!"


def register_driver(name: str, password: str, contact: str, license_num: str) -> str:
    if not name.strip() or not password.strip():
        return "Name and Password are required."

    conn = sqlite3.connect(config.DB_NAME)
    try:
        conn.execute(
            "INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber) "
            "VALUES (?, ?, ?, ?)",
            (name, password, contact, license_num),
        )
        conn.commit()
    finally:
        conn.close()

    return "Registration successful! You can now log in."
