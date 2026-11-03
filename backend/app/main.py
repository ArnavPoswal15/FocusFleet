import os
from pathlib import Path

import app.path_setup  # noqa: F401 — repo root on sys.path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.api import admin, analytics, auth, sessions, websocket
from app.core.security import hash_password
from app.core.config import DATA_DIR
from app.path_setup import STATIC_DIR
from focusfleet.database import ensure_admin_account, init_db


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_DIR)


def _resolve_static_file(relative: str) -> Path | None:
    if not STATIC_DIR.is_dir():
        return None
    static_root = STATIC_DIR.resolve()
    candidates = [relative]
    if not relative.endswith(".html"):
        candidates.append(f"{relative.rstrip('/')}/index.html")
    for rel in candidates:
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(static_root)):
            continue
        if target.is_file():
            return target
        if target.is_dir() and (target / "index.html").is_file():
            return target / "index.html"
    return None


# Next `output: "export"` only pre-renders placeholder dynamic segments (see generateStaticParams).
_ADMIN_DRIVER_DETAIL_SHELL = "admin/drivers/0"
_ADMIN_SESSION_LOG_SHELL = "admin/drivers/0/sessions/00000000_000000"


def _dynamic_export_shell(relative: str) -> Path | None:
    """Serve pre-built shells for admin dynamic routes so the client router keeps the URL."""
    parts = relative.strip("/").split("/")
    if (
        len(parts) >= 5
        and parts[0] == "admin"
        and parts[1] == "drivers"
        and parts[3] == "sessions"
    ):
        return _resolve_static_file(_ADMIN_SESSION_LOG_SHELL)
    if (
        len(parts) == 3
        and parts[0] == "admin"
        and parts[1] == "drivers"
        and parts[2].isdigit()
    ):
        return _resolve_static_file(_ADMIN_DRIVER_DETAIL_SHELL)
    return None


class _StripApiTrailingSlashMiddleware(BaseHTTPMiddleware):
    """Frontend static export may call `/api/foo/`; FastAPI routes are `/api/foo`."""

    async def dispatch(self, request: Request, call_next):
        path = request.scope.get("path", "")
        if (
            (path.startswith("/api/") or path.startswith("/ws/"))
            and path.endswith("/")
            and len(path) > 1
        ):
            request.scope["path"] = path.rstrip("/")
        return await call_next(request)


app = FastAPI(title="FocusFleet API", version="0.1.0")

app.add_middleware(_StripApiTrailingSlashMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FOCUSFLEET_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(sessions.router)
app.include_router(analytics.router)
app.include_router(websocket.router)


@app.on_event("startup")
def on_startup() -> None:
    _ensure_data_dir()
    init_db()
    admin_user = os.getenv("FOCUSFLEET_ADMIN_USERNAME", "").strip()
    admin_pass = os.getenv("FOCUSFLEET_ADMIN_PASSWORD", "")
    if admin_user and admin_pass:
        try:
            if ensure_admin_account(admin_user, hash_password(admin_pass)):
                print(f"[FocusFleet] Admin account ready for login: {admin_user!r}")
            else:
                print("[FocusFleet] Admin bootstrap failed (empty username or hash).")
        except Exception as exc:
            print(f"[FocusFleet] Admin bootstrap error: {exc}")
    else:
        missing = []
        if not admin_user:
            missing.append("FOCUSFLEET_ADMIN_USERNAME")
        if not admin_pass:
            missing.append("FOCUSFLEET_ADMIN_PASSWORD")
        print(f"[FocusFleet] Admin bootstrap skipped (set Space secrets: {', '.join(missing)})")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str) -> FileResponse:
    if full_path.startswith(("api/", "api", "ws/", "ws")):
        raise HTTPException(status_code=404, detail="Not found")

    normalized = full_path.strip("/")
    if not normalized:
        normalized = "index.html"

    lower = normalized.lower()
    if lower.startswith(".") or lower.endswith(".env") or ".env" in lower:
        raise HTTPException(status_code=404, detail="Not found")

    resolved = _resolve_static_file(normalized)
    if resolved is None:
        resolved = _dynamic_export_shell(normalized)
    if resolved is not None:
        return FileResponse(resolved)

    not_found = STATIC_DIR / "404.html"
    if not_found.is_file():
        return FileResponse(not_found, status_code=404)

    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)

    raise HTTPException(status_code=404, detail="Not found")
