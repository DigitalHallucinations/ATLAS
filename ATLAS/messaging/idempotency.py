"""Idempotency helpers for message delivery."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class IdempotencyStore:
    """Track idempotency keys using Redis when available with an in-memory fallback."""

    def __init__(self, redis_client: Any | None = None, *, namespace: str = "atlas:messaging:idempotency") -> None:
        self._redis = redis_client
        self._namespace = namespace.rstrip(":")
        self._memory_store: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check_and_set(self, key: str, ttl_seconds: float | int | None) -> bool:
        """Return ``True`` on first-seen keys and ``False`` when a duplicate is detected."""

        normalized_key = (key or "").strip()
        ttl = self._normalize_ttl(ttl_seconds)
        if not normalized_key or ttl is None:
            return True

        if self._redis is not None:
            return bool(await self._redis.set(self._namespaced(normalized_key), "1", ex=ttl, nx=True))

        return await self._check_and_set_memory(normalized_key, ttl)

    async def _check_and_set_memory(self, key: str, ttl: float) -> bool:
        async with self._lock:
            now = time.monotonic()
            self._purge_expired(now)

            expires_at = now + ttl
            existing_expiry = self._memory_store.get(key)
            if existing_expiry and existing_expiry > now:
                return False

            self._memory_store[key] = expires_at
            return True

    def _purge_expired(self, now: float) -> None:
        expired_keys = [candidate for candidate, expiry in self._memory_store.items() if expiry <= now]
        for candidate in expired_keys:
            self._memory_store.pop(candidate, None)

    def _namespaced(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    @staticmethod
    def _normalize_ttl(ttl_seconds: float | int | None) -> float | None:
        if ttl_seconds is None:
            return None
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError):
            return None
        if ttl <= 0:
            return None
        return ttl
