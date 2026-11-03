from passlib.context import CryptContext
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.core.config import SECRET_KEY, SESSION_MAX_AGE_SECONDS

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_session_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="focusfleet-session")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, stored: str) -> bool:
    if stored.startswith("$2"):
        return pwd_context.verify(plain, stored)
    return plain == stored


def create_session_token(payload: dict) -> str:
    return _session_serializer.dumps(payload)


def read_session_token(token: str) -> dict | None:
    try:
        return _session_serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
