from fastapi import Cookie, Depends, Header, HTTPException, status

from app.core.config import SESSION_COOKIE_NAME
from app.core.security import read_session_token
from focusfleet.database import ROLE_ADMIN, ROLE_DRIVER


def extract_session_token(
    session: str | None,
    authorization: str | None,
) -> str | None:
    if session:
        return session
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _normalize_role(payload: dict) -> dict:
    role = payload.get("role") or ROLE_DRIVER
    return {**payload, "role": role}


def user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    payload = read_session_token(token)
    if not payload or "driver_id" not in payload:
        return None
    return _normalize_role(payload)


def get_current_user(
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    authorization: str | None = Header(default=None),
) -> dict:
    resolved = extract_session_token(session, authorization)
    user = user_from_token(resolved)
    if not user:
        detail = "Invalid session" if resolved else "Not authenticated"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
    return user


def get_optional_user(
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    authorization: str | None = Header(default=None),
) -> dict | None:
    resolved = extract_session_token(session, authorization)
    return user_from_token(resolved)


def require_driver(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != ROLE_DRIVER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Driver access required")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != ROLE_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
