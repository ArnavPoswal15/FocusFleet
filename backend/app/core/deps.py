from fastapi import Cookie, Depends, HTTPException, status

from app.core.config import SESSION_COOKIE_NAME
from app.core.security import read_session_token
from focusfleet.database import ROLE_ADMIN, ROLE_DRIVER


def _normalize_role(payload: dict) -> dict:
    role = payload.get("role") or ROLE_DRIVER
    return {**payload, "role": role}


def get_current_user(session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict:
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = read_session_token(session)
    if not payload or "driver_id" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return _normalize_role(payload)


def get_optional_user(session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict | None:
    if not session:
        return None
    payload = read_session_token(session)
    if not payload:
        return None
    return _normalize_role(payload)


def require_driver(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != ROLE_DRIVER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Driver access required")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != ROLE_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
