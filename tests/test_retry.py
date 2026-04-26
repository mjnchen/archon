"""Tests for the narrow retry helper."""

from __future__ import annotations

import pytest

from archon.retry import with_retry, _is_transient, _retry_after_seconds


# ---------------------------------------------------------------------------
# Synthetic exceptions matching provider class names
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Mimics openai.RateLimitError / anthropic.RateLimitError."""


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class InternalServerError(Exception):
    pass


class APIStatusError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code


class AuthenticationError(Exception):
    pass


class BadRequestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_rate_limit_is_transient(self):
        assert _is_transient(RateLimitError())

    def test_api_connection_is_transient(self):
        assert _is_transient(APIConnectionError())

    def test_internal_server_error_is_transient(self):
        assert _is_transient(InternalServerError())

    def test_timeout_is_transient(self):
        assert _is_transient(APITimeoutError())

    def test_5xx_status_is_transient(self):
        assert _is_transient(APIStatusError(503))

    def test_429_status_is_transient(self):
        assert _is_transient(APIStatusError(429))

    def test_400_status_is_not_transient(self):
        assert not _is_transient(APIStatusError(400))

    def test_auth_error_never_retries(self):
        assert not _is_transient(AuthenticationError())

    def test_bad_request_never_retries(self):
        assert not _is_transient(BadRequestError())

    def test_unknown_class_never_retries(self):
        assert not _is_transient(ValueError("nope"))


# ---------------------------------------------------------------------------
# Retry-After header parsing
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, headers: dict):
        self.headers = headers


class _ExcWithResponse(Exception):
    def __init__(self, headers: dict):
        self.response = _FakeResponse(headers)


class TestRetryAfter:
    def test_returns_seconds_when_header_present(self):
        exc = _ExcWithResponse({"retry-after": "7"})
        assert _retry_after_seconds(exc) == 7.0

    def test_returns_none_when_missing(self):
        exc = _ExcWithResponse({})
        assert _retry_after_seconds(exc) is None

    def test_returns_none_when_unparseable(self):
        exc = _ExcWithResponse({"retry-after": "Mon, 01 Jan 2030 00:00:00 GMT"})
        assert _retry_after_seconds(exc) is None

    def test_returns_none_when_no_response_attribute(self):
        assert _retry_after_seconds(ValueError("nope")) is None


# ---------------------------------------------------------------------------
# with_retry behavior
# ---------------------------------------------------------------------------

class TestWithRetry:
    @pytest.mark.asyncio
    async def test_returns_value_on_success_first_try(self):
        async def f():
            return 42

        assert await with_retry(f, max_attempts=2, initial_backoff=0) == 42

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        calls = {"n": 0}

        async def f():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RateLimitError()
            return "ok"

        result = await with_retry(f, max_attempts=2, initial_backoff=0)
        assert result == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_attempts(self):
        async def f():
            raise RateLimitError("still down")

        with pytest.raises(RateLimitError):
            await with_retry(f, max_attempts=2, initial_backoff=0)

    @pytest.mark.asyncio
    async def test_does_not_retry_non_transient(self):
        calls = {"n": 0}

        async def f():
            calls["n"] += 1
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            await with_retry(f, max_attempts=4, initial_backoff=0)
        assert calls["n"] == 1
