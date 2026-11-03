import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.core.config import SESSION_COOKIE_NAME, SESSION_MAX_AGE_SECONDS
from app.core.deps import get_current_user
from app.core.security import create_session_token, hash_password, verify_password
from focusfleet import config
from focusfleet.database import ROLE_DRIVER, driver_role_from_row

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    name: str = Field(min_length=1)
    password: str = Field(min_length=1)
    contact: str = ""
    license_number: str = ""


class LoginBody(BaseModel):
    name: str = Field(min_length=1)
    password: str = Field(min_length=1)


def _user_from_row(row: tuple) -> dict:
    role = driver_role_from_row(row)
    return {
        "driver_id": row[0],
        "name": row[1],
        "contact_info": row[3],
        "license_number": row[4],
        "role": role,
    }


def _set_session_cookie(response: Response, driver_id: int, name: str, role: str) -> None:
    token = create_session_token({"driver_id": driver_id, "name": name, "role": role})
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=os.getenv("FOCUSFLEET_COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterBody) -> dict:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        existing = conn.execute("SELECT 1 FROM Driver WHERE Name=?", (body.name,)).fetchone()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Name already registered")
        hashed = hash_password(body.password)
        conn.execute(
            "INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber, Role) "
            "VALUES (?, ?, ?, ?, ?)",
            (body.name, hashed, body.contact, body.license_number, ROLE_DRIVER),
        )
        conn.commit()
    finally:
        conn.close()
    return {"message": "Registration successful"}


@router.post("/login")
def login(body: LoginBody, response: Response) -> dict:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute("SELECT * FROM Driver WHERE Name=?", (body.name,)).fetchone()
    finally:
        conn.close()

    if not row or not verify_password(body.password, row[2]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user = _user_from_row(row)
    _set_session_cookie(response, user["driver_id"], user["name"], user["role"])
    return {"user": user}


@router.get("/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute("SELECT * FROM Driver WHERE DriverID=?", (user["driver_id"],)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return _user_from_row(row)


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"message": "Logged out"}
