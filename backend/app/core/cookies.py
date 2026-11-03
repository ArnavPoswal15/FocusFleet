import os
from typing import Literal

from starlette.requests import Request

SameSite = Literal["lax", "none", "strict"]


def session_cookie_params(request: Request) -> dict[str, bool | SameSite]:
    """HF Spaces run in an HTTPS iframe — cookies need Secure + SameSite=None."""
    forced = os.getenv("FOCUSFLEET_COOKIE_SECURE", "").strip().lower()
    if forced == "true":
        secure = True
    elif forced == "false":
        secure = False
    else:
        proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        secure = proto == "https" or request.url.scheme == "https"

    samesite: SameSite = "none" if secure else "lax"
    return {"secure": secure, "samesite": samesite}
