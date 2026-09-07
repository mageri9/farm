from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import math


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


async def generate_tts(text: str, audio_path: Path, voice: str, rate: str) -> list[WordBoundary]:
    """Stream Edge TTS audio to disk and return word timings in seconds."""
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSError("edge-tts is not installed. Run: pip install -r requirements.txt") from exc
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    boundaries: list[WordBoundary] = []
    try:
        # В новых версиях edge-tts нужно явно запросить WordBoundary
        kwargs = {"rate": rate}
        if "boundary" in inspect.signature(edge_tts.Communicate).parameters:
            kwargs["boundary"] = "WordBoundary"

        communicate = edge_tts.Communicate(text, voice, **kwargs)
        with audio_path.open("wb") as output:
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
    except Exception as exc:
        raise TTSError(f"Edge TTS failed ({type(exc).__name__}); check network and voice settings") from exc
    if not boundaries:
        raise TTSError("Edge TTS returned no WordBoundary events; cannot create timed subtitles.")
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise TTSError("Edge TTS produced an empty audio file.")
    return boundaries
