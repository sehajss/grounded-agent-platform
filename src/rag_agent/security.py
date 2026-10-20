from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from jwt import InvalidTokenError

from rag_agent.config import Settings, get_settings


_PASSWORD_ALGORITHM = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")

    salt = hashlib.sha256(str(datetime.now(UTC).timestamp()).encode()).digest()[:16]
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii")

    return f"{_PASSWORD_ALGORITHM}${_PASSWORD_ITERATIONS}${salt_text}${digest_text}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, expected_text = password_hash.split("$")
        iterations = int(iterations_text)
        if (
            algorithm != _PASSWORD_ALGORITHM
            or iterations < 100_000
            or iterations > 1_000_000
        ):
            return False

        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(expected_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: UUID, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_access_token_expire_minutes),
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def decode_access_token(token: str, settings: Settings | None = None) -> UUID:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"],
            options={"require": ["exp", "iat", "sub", "type"]},
        )
        if payload["type"] != "access":
            raise ValueError("Invalid token type")
        return UUID(payload["sub"])
    except (InvalidTokenError, ValueError, TypeError) as exc:
        raise ValueError("Invalid or expired access token") from exc