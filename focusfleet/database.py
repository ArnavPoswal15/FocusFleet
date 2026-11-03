"""SQLite-backed driver accounts."""

import datetime
import os
import sqlite3

from . import config
from .state import STATE
from focusfleet.session_logger import DriverStateTracker

ROLE_DRIVER = "driver"
ROLE_ADMIN = "admin"


def _driver_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(Driver)").fetchall()
    return {row[1] for row in rows}


def _ensure_role_column(conn: sqlite3.Connection) -> None:
    if "Role" not in _driver_columns(conn):
        conn.execute(
            f"ALTER TABLE Driver ADD COLUMN Role TEXT NOT NULL DEFAULT '{ROLE_DRIVER}'"
        )


def driver_role_from_row(row: tuple) -> str:
    if len(row) >= 6 and row[5]:
        return str(row[5])
    return ROLE_DRIVER


def init_db() -> None:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Driver (
                DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Password TEXT NOT NULL,
                ContactInfo TEXT,
                LicenseNumber TEXT,
                Role TEXT NOT NULL DEFAULT 'driver'
            )
            """
        )
        _ensure_role_column(conn)
        conn.commit()
    finally:
        conn.close()


def ensure_admin_account(username: str, password_hash: str) -> bool:
    """Create or update the fleet admin from env (Space secrets). Returns True if synced."""
    name = username.strip()
    if not name or not password_hash:
        return False

    conn = sqlite3.connect(config.DB_NAME)
    try:
        _ensure_role_column(conn)
        row = conn.execute(
            "SELECT DriverID, Role FROM Driver WHERE Name=? LIMIT 1", (name,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE Driver SET Password=?, Role=? WHERE Name=?",
                (password_hash, ROLE_ADMIN, name),
            )
            conn.commit()
            return True

        conn.execute(
            "INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber, Role) "
            "VALUES (?, ?, '', '', ?)",
            (name, password_hash, ROLE_ADMIN),
        )
        conn.commit()
        return True
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

    if driver_role_from_row(row) == ROLE_ADMIN:
        return None, "Use the web console to sign in as admin."

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
        _ensure_role_column(conn)
        conn.execute(
            "INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber, Role) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, password, contact, license_num, ROLE_DRIVER),
        )
        conn.commit()
    finally:
        conn.close()

    return "Registration successful! You can now log in."
