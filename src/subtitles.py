from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from .tts import WordBoundary


def format_time_ass(seconds: float) -> str:
    """Format seconds as ASS H:MM:SS.cc, rounding to centiseconds."""
    total_cs = max(0, int(math.floor(float(seconds) * 100 + 0.5)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def group_words(words: Iterable[WordBoundary], words_per_subtitle: int = 2) -> list[list[WordBoundary]]:
    if words_per_subtitle < 1:
        raise ValueError("words_per_subtitle must be >= 1")
    clean = [w for w in words if w.word and w.end >= w.start]
    return [clean[i : i + words_per_subtitle] for i in range(0, len(clean), words_per_subtitle)]


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def choose_font(configured: str, fonts_dir: Path | None = None) -> str:
    if not configured:
        return "Arial"
    if fonts_dir:
        for path in fonts_dir.glob("*"):
            if path.suffix.lower() in {".ttf", ".otf"} and path.stem.lower() == configured.lower():
                return configured
    # ASS/FFmpeg will resolve installed fonts; Arial is a dependable fallback.
    return configured


def write_ass(words: Iterable[WordBoundary], output_path: Path, words_per_subtitle: int = 2,
              font_name: str = "Montserrat", font_size: int = 80, fonts_dir: Path | None = None) -> int:
    groups = group_words(words, words_per_subtitle)
    if not groups:
        raise ValueError("No valid words available for subtitles")
    font = choose_font(font_name, fonts_dir)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,2,5,40,40,40,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for group in groups:
        start, end = group[0].start, group[-1].end
        text = " ".join(_escape_ass_text(w.word) for w in group)
        lines.append(f"Dialogue: 0,{format_time_ass(start)},{format_time_ass(end)},Default,,0,0,0,,{text}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(groups)


def escape_subtitle_path(path: str | Path) -> str:
    """Escape a path for FFmpeg's subtitles= filter argument (Windows-safe)."""
    value = str(Path(path).resolve()).replace("\\", "/")
    return value.replace("'", r"\'").replace(":", r"\:").replace(",", r"\,")

