from __future__ import annotations

import math
import re
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
    groups: list[list[WordBoundary]] = []
    current: list[WordBoundary] = []
    for index, word in enumerate(clean):
        current.append(word)
        # Edge-TTS may strip punctuation; treat capitals as sentence starts.
        next_word = clean[index + 1].word if index + 1 < len(clean) else ""
        cleaned_next_word = next_word.lstrip(" \t\r\n\"'\u00ab\u00bb\u201c\u201d\u201e\u2018\u2019([{\u2039\u203a")
        next_sentence = bool(cleaned_next_word and cleaned_next_word[0].isupper())
        # Sentence-final punctuation may be followed by closing quotes/brackets.
        sentence_end = bool(re.search(r"[.!?\u2026][\"'\u00bb\u201d\u2019\u201c)}\]]*\s*$", word.word))
        if len(current) >= words_per_subtitle or sentence_end or next_sentence:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def choose_font(configured: str, fonts_dir: Path | None = None) -> str:
    requested = configured.strip() or "Forum"
    if requested.casefold() == "forum":
        if fonts_dir is not None:
            try:
                if not any(p.is_file() and "forum" in p.stem.casefold() for p in Path(fonts_dir).iterdir()):
                    return "Georgia"
            except OSError:
                return "Georgia"
    return requested


def write_ass(words: Iterable[WordBoundary], output_path: Path, words_per_subtitle: int = 2,
              font_name: str = "Forum", font_size: int = 74, fonts_dir: Path | None = None) -> int:
    groups = group_words(words, words_per_subtitle)
    if not groups:
        raise ValueError("No valid words available for subtitles")
    font = choose_font(font_name, fonts_dir)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{font_size},&H00E8E8E8,&H0061D0F5,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1.5,2,40,40,260,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hl_tag = "{\\c&H0061D0F5&}"
    reset_tag = "{\\c&H00E8E8E8&}"

    for group in groups:
        n_words = len(group)
        for active_idx in range(n_words):
            w_start = group[0].start if active_idx == 0 else group[active_idx].start
            if active_idx < n_words - 1:
                w_end = group[active_idx + 1].start
            else:
                w_end = group[-1].end

            if w_end <= w_start:
                w_end = w_start + 0.05

            parts = []
            for idx, w in enumerate(group):
                word_escaped = _escape_ass_text(w.word)
                if idx == active_idx:
                    parts.append(f"{hl_tag}{word_escaped}{reset_tag}")
                else:
                    parts.append(word_escaped)

            text = " ".join(parts)
            lines.append(f"Dialogue: 0,{format_time_ass(w_start)},{format_time_ass(w_end)},Default,,0,0,0,,{text}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(groups)


def escape_subtitle_path(path: str | Path) -> str:
    """Escape a path for FFmpeg's subtitles= filter argument (Windows-safe)."""
    raw = str(path)
    # Do not resolve a Windows path on a non-Windows host: Path.resolve() would
    # prepend the current working directory to an already absolute drive path.
    is_windows_absolute = len(raw) >= 3 and raw[1] == ":" and raw[2] in "\\/"
    is_unc = raw.startswith((r"\\", "//"))
    is_posix_absolute = raw.startswith("/")
    value = raw if is_windows_absolute or is_unc or is_posix_absolute else str(Path(path).resolve())
    value = value.replace("\\", "/")
    return value.replace("'", r"\'").replace(":", r"\:").replace(",", r"\,")
