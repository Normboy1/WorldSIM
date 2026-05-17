"""Small cached HTTP client for external scientific data sources."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "proof" / "cache" / "http"
_DEFAULT_TTL_S = 24 * 60 * 60


@dataclass(frozen=True)
class CachedHTTPResponse:
    """HTTP response body with provenance for cache-aware callers."""

    text: str
    source: str
    status_code: int

    def json(self) -> Any:
        return json.loads(self.text)


def _configured_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env_dir = os.getenv("SIMLAB_DATA_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_CACHE_DIR


def _configured_ttl(ttl_s: float | None) -> float:
    if ttl_s is not None:
        return float(ttl_s)
    raw = os.getenv("SIMLAB_DATA_CACHE_TTL_S")
    if raw:
        return float(raw)
    return float(_DEFAULT_TTL_S)


class CachedHTTPClient:
    """GET-only HTTP client with TTL cache and stale-cache fallback."""

    def __init__(
        self,
        timeout: float = 15.0,
        cache_dir: str | Path | None = None,
        ttl_s: float | None = None,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.cache_dir = _configured_cache_dir(cache_dir)
        self.ttl_s = _configured_ttl(ttl_s)
        self.use_cache = use_cache
        self.transport = transport

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> CachedHTTPResponse:
        """GET a URL, using cached data when fresh and stale data on network failure."""
        cache_path = self._cache_path(url, params)
        cached = self._read_cache(cache_path)

        if self.use_cache and cached and self._is_fresh(cached):
            return CachedHTTPResponse(
                text=cached["text"],
                source="cache",
                status_code=int(cached.get("status_code", 200)),
            )

        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                text = response.text
                if self.use_cache:
                    self._write_cache(cache_path, url, params, response.status_code, text)
                return CachedHTTPResponse(
                    text=text,
                    source="network",
                    status_code=response.status_code,
                )
        except Exception:
            if self.use_cache and cached:
                return CachedHTTPResponse(
                    text=cached["text"],
                    source="stale_cache",
                    status_code=int(cached.get("status_code", 200)),
                )
            raise

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        """GET JSON and return `(decoded_json, source)`."""
        response = self.get_text(url, params=params)
        return response.json(), response.source

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key_payload = json.dumps(
            {"url": url, "params": params or {}},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(
        self,
        path: Path,
        url: str,
        params: dict[str, Any] | None,
        status_code: int,
        text: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": time.time(),
            "url": url,
            "params": params or {},
            "status_code": status_code,
            "text": text,
        }
        path.write_text(json.dumps(payload))

    def _is_fresh(self, cached: dict[str, Any]) -> bool:
        created_at = float(cached.get("created_at", 0.0))
        return (time.time() - created_at) <= self.ttl_s
