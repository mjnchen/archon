"""Narrow retry helper for LLM calls.

Handles the errors providers universally classify as transient (rate limits,
connection blips, 5xx) and honors Retry-After headers when present. Never
retries on auth/bad-request/permission errors. Deliberately minimal — users
who want richer policies (deadlines, circuit breakers, tenacity) can wrap
``acompletion`` themselves.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


_TRANSIENT_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "APIStatusError",
}

_NON_RETRY_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "ConflictError",
}


def _is_transient(exc: Exception) -> bool:
    """True if *exc* is one of the universally-transient provider errors."""
    cls = type(exc).__name__
    if cls in _NON_RETRY_NAMES:
        return False
    if cls not in _TRANSIENT_NAMES:
        return False
    # APIStatusError is a broad catch-all; only retry on 429 / 5xx.
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or 500 <= status < 600
    return True


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract a Retry-After header value from the provider error, if any."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 2,
    initial_backoff: float = 1.0,
) -> T:
    """Call *fn*; retry on transient LLM errors up to *max_attempts* times.

    Backoff is Retry-After (when present) else exponential with jitter.
    Caller surfaces the final exception unchanged if all attempts fail.
    """
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == max_attempts - 1 or not _is_transient(exc):
                raise
            wait = _retry_after_seconds(exc)
            if wait is None:
                wait = initial_backoff * (2 ** attempt) + random.uniform(0, 0.3)
            log.warning(
                "LLM call failed with %s; retrying in %.2fs (%d/%d)",
                type(exc).__name__, wait, attempt + 2, max_attempts,
            )
            await asyncio.sleep(wait)
    # Unreachable — loop either returns or raises — but keeps type checkers happy.
    raise RuntimeError("with_retry: exhausted without raising")
