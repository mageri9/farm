from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import math
import asyncio
import socket
from uuid import uuid4

from .config import Settings
from .runtime import logger, safe_error


class TTSError(RuntimeError):
    """Raised when speech or word timings cannot be generated."""


@dataclass(frozen=True)
class WordBoundary:
    word: str
    start: float
    end: float


def _boundary_from_event(event: Any) -> WordBoundary | None:
    data = getattr(event, "__dict__", event)
    if not isinstance(data, dict):
        return None
    # edge-tts exposes offset/duration in 100 ns ticks.
    offset = data.get("offset")
    duration = data.get("duration")
    text = data.get("text") or data.get("word") or ""
    if offset is None or duration is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        start = max(0.0, float(offset) * 1e-7)
        end = max(start, start + float(duration) * 1e-7)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return WordBoundary(text, start, end)


def _retryable(exc: Exception) -> bool:
    import aiohttp
    from edge_tts.exceptions import EdgeTTSException

    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or 500 <= exc.status < 600
    if isinstance(exc, (aiohttp.InvalidURL, aiohttp.ClientSSLError, ValueError, TypeError)):
        return False
    return isinstance(exc, (EdgeTTSException, TTSError, aiohttp.ClientConnectionError,
                            aiohttp.ClientPayloadError, ConnectionError, socket.gaierror, TimeoutError))


async def generate_tts(text: str, audio_path: Path, voice: str, rate: str, *,
                       attempts: int = 3, retry_delay: float = 3.0,
                       timeout: float = 30.0) -> list[WordBoundary]:
    """Stream Edge TTS audio to disk and return word timings in seconds."""
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSError("edge-tts is not installed. Run: pip install -r requirements.txt") from exc
    # Reuse validation for the existing SHORTS retry/timeout settings.
    Settings(retry_attempts=attempts, retry_delay=retry_delay, tts_timeout=timeout)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    partial = audio_path.with_name(f".{audio_path.name}.{uuid4().hex}.partial")
    fallback_voice = "ru-RU-SvetlanaNeural"
    voices = [voice] * attempts
    if voice == "ru-RU-DmitryNeural":
        voices.append(fallback_voice)
    for attempt, selected_voice in enumerate(voices):
        if attempt == attempts:
            logger.warning("Primary TTS voice failed after retries. Switching to fallback voice: %s", fallback_voice)
        boundaries: list[WordBoundary] = []
        try:
            kwargs = {"rate": rate}
            if "boundary" in inspect.signature(edge_tts.Communicate).parameters:
                kwargs["boundary"] = "WordBoundary"
            communicate = edge_tts.Communicate(text, selected_voice, **kwargs)
            async with asyncio.timeout(timeout):
                with partial.open("wb") as output:
                    async for event in communicate.stream():
                        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                        if event_type == "audio":
                            data = event.get("data") if isinstance(event, dict) else getattr(event, "data", None)
                            if data:
                                output.write(data)
                        elif event_type == "WordBoundary":
                            boundary = _boundary_from_event(event)
                            if boundary:
                                boundaries.append(boundary)
            if not boundaries:
                raise TTSError("Edge TTS returned no WordBoundary events; cannot create timed subtitles.")
            if partial.stat().st_size == 0:
                raise TTSError("Edge TTS produced an empty audio file.")
            partial.replace(audio_path)
            return boundaries
        except Exception as exc:
            logger.warning("TTS error: %s (voice=%s, attempt=%s/%s)",
                           safe_error(exc), selected_voice, attempt + 1, len(voices))
            if not _retryable(exc) or attempt + 1 == len(voices):
                raise TTSError(f"Edge TTS failed: {safe_error(exc)}") from exc
        finally:
            partial.unlink(missing_ok=True)
        if attempt + 1 < attempts:
            await asyncio.sleep(min(60, retry_delay * 2 ** attempt))
    raise TTSError("Edge TTS failed after retries")
