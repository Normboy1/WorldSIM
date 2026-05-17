"""CORS configuration helpers for the FastAPI server."""

from __future__ import annotations

import os


DEFAULT_CORS_ORIGINS = (
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
)


def cors_origins() -> list[str]:
    """Return the configured CORS origin allowlist."""
    raw = os.getenv("SIMLAB_CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def cors_allow_credentials(origins: list[str] | None = None) -> bool:
    """Return whether CORS credentials are enabled for a non-wildcard origin set."""
    allow = os.getenv("SIMLAB_CORS_ALLOW_CREDENTIALS", "false").lower()
    enabled = allow in {"1", "true", "yes", "on"}
    return enabled and "*" not in (origins or cors_origins())
