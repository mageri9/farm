from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from openai import APIConnectionError, APIStatusError

from ..runtime import logger, safe_error

T = TypeVar("T")

def is_transient(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(exc, (APIConnectionError, httpx.TimeoutException, httpx.NetworkError,
                            httpx.RemoteProtocolError, TimeoutError, ConnectionError))


async def complete_with_fallback(call: Callable[[str], Awaitable[T]], primary: str, fallback: str) -> T:
    """Retry transient failures on primary, then make one bounded fallback attempt."""
    for attempt in range(2):
        try:
            return await call(primary)
        except Exception as exc:
            if not is_transient(exc):
                raise
            if attempt == 0:
                logger.warning("Primary LLM request failed; retrying. Error: %s", safe_error(exc))
                await asyncio.sleep(2)
            else:
                logger.warning("Primary LLM model failed after retries. "
                               "Switching to fallback model: %s Error: %s", fallback, safe_error(exc))
    return await call(fallback)
