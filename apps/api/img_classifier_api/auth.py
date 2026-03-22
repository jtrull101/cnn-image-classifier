"""API key authentication dependency for FastAPI."""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER_NAME = "X-API-Key"
_API_KEY_ENV_VAR = "IMG_CLASSIFIER_API_KEY"

_api_key_header = APIKeyHeader(name=_API_KEY_HEADER_NAME, auto_error=False)


def _get_configured_api_key() -> str | None:
    """Return the configured API key, or None if authentication is disabled.

    Reads from the environment on every call (rather than caching at module
    load time) so that tests can change the key at runtime via
    ``monkeypatch.setenv`` without reloading the module.
    """
    value = os.environ.get(_API_KEY_ENV_VAR, "").strip()
    return value if value else None


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency that enforces API key authentication.

    If ``IMG_CLASSIFIER_API_KEY`` is set in the environment every protected
    request must supply a matching ``X-API-Key`` header.  If the variable is
    not set the check is skipped (development / local mode).

    Raises:
        HTTPException 401: when a key is configured and the header is
            absent or does not match.
    """
    expected = _get_configured_api_key()
    if expected is None:
        return  # Auth disabled — development mode

    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
