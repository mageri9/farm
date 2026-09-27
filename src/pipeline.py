from __future__ import annotations

import os
import math
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from .config import Settings
from .runtime import event, file_lock, logger, safe_error
from .subtitles import write_ass
from .tts import TTSError, generate_tts
from .video import (
    check_executable,
    find_item_images,
    probe_duration,
    random_start,
    render_video,
    validate_output,
)


BACKGROUND_TARGET_CHUNK = 4.2


class ShortsPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.last_report: dict | None = None
        self.last_stage = "preflight"

    def preflight(self):
        s = self.settings
        check_executable(s.ffmpeg)
        check_executable(s.ffprobe)
        has_pool = bool(self._background_candidates())
        if not has_pool:
            raise ValueError(f"No MP4 or MOV in {s.assets_dir / 'backgrounds'}")

    def _background_candidates(self) -> list[Path]:
        directory = self.settings.assets_dir / "backgrounds"
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}) if directory.is_dir() else []

    def _select_backgrounds(self, total_duration: float, seed: int | None = None, target_clips: int = 4, story: dict | None = None) -> tuple[list[Path], list[float]]:
        s = self.settings
        if not math.isfinite(total_duration) or total_duration <= 0:
            raise ValueError("total_duration must be positive and finite")
        # Nearest count gives eight equal shots for 33-35 seconds of speech.
        num_clips = max(1, int(math.floor(total_duration / BACKGROUND_TARGET_CHUNK + 0.5)))
        self._clip_durations = [total_duration / num_clips] * num_clips
        print(f"[VIDEO] Равные тайминги планов (сек): {[round(t, 3) for t in self._clip_durations]}")

        videos = self._background_candidates()

        if videos:
            rng = random.Random(seed)
            selected = rng.sample(videos, num_clips) if len(videos) >= num_clips else rng.choices(videos, k=num_clips)
            offsets = []
            for idx, bg in enumerate(selected):
                dur = probe_duration(bg, s)
                offsets.append(random_start(dur, self._clip_durations[idx], None if seed is None else seed + idx))
            print(f"[VIDEO] Найдено {len(videos)} видео. Для ролика выбраны {num_clips} плана: " + " -> ".join(p.name for p in selected))
            return selected, offsets

        raise ValueError("No MP4 or MOV in assets/backgrounds")

    def _overlay_images(self, story: dict | None, overlays: Sequence[Path] | None) -> list[Path]:
        """Явно переданные картинки важнее, чем найденные по slug. Fail-soft: [] допустимо."""
        if overlays:
            return [Path(p) for p in overlays if Path(p).is_file()]
        slug = str((story or {}).get("item_slug") or "").strip()
        return find_item_images(self.settings.assets_dir, slug) if slug else []

    async def run(self, text: str | dict, output_path: str | Path | None = None, seed: int | None = None,
                  overlays: Sequence[Path] | None = None) -> Path:
        s = self.settings
        self.last_report = None
        self.last_stage = "preflight"
        story = text if isinstance(text, dict) else None
        if story is not None:
            text = story.get("text") or " ".join(str(story.get("beats", {}).get(k, "")) for k in ("establishing", "tension", "subject", "aftermath"))
        if not isinstance(text, str) or not 3 <= len(text.strip()) <= s.max_text_chars:
            raise ValueError(f"Text must contain 3-{s.max_text_chars} characters")
        text = text.strip()
        self.preflight()
        output = Path(output_path) if output_path else s.output_dir / f"final_{uuid4().hex}.mp4"
        if not output.is_absolute():
            output = s.root / output
        output = output.resolve()
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        output.parent.mkdir(parents=True, exist_ok=True)
        s.work_dir.mkdir(parents=True, exist_ok=True)

        partial = output.with_name(f".{output.stem}.{uuid4().hex}.partial.mp4")
        with file_lock(output.with_suffix(".mp4.lock")):
            run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=s.work_dir))
            started = time.monotonic()
            try:
                audio, ass = run_dir / "audio.mp3", run_dir / "subtitles.ass"
                self.last_stage = "tts"
                event("tts.started", output=output)
                boundaries = await generate_tts(text, audio, s.voice, s.rate,
                                                attempts=s.retry_attempts, retry_delay=s.retry_delay,
                                                timeout=s.tts_timeout, settings=s)
                duration = probe_duration(audio, s)
                if duration > s.max_speech_seconds:
                    event("tts.duration_rejected", duration=duration, limit=s.max_speech_seconds, output=output)
                    raise TTSError(
                        "Previous script exceeded the TTS duration limit. "
                        "Rewrite it shorter, simpler and more conversational."
                    )
                event("tts.completed", words=len(boundaries), seconds=round(time.monotonic() - started, 3))
                self.last_stage = "subtitles"
                write_ass(boundaries, ass, s.words_per_subtitle, s.font_name, s.font_size, s.assets_dir / "fonts")

                self.last_stage = "background"
                bg_list, start_offsets = self._select_backgrounds(duration, seed, story=story)

                self.last_stage = "render"
                overlay_images = self._overlay_images(story, overlays)
                if overlay_images:
                    print(f"[VIDEO] Оверлеи предмета ({len(overlay_images)}): "
                          + ", ".join(p.name for p in overlay_images))
                event("render.started", output=output, duration=duration, clips=len(bg_list),
                      overlays=len(overlay_images))
                render_video(bg_list, audio, ass, partial, duration, start_offsets, s,
                             clip_durations=getattr(self, "_clip_durations", None),
                             overlays=overlay_images)
                self.last_stage = "validation"
                report = validate_output(partial, s, expected_duration=duration)
                self.last_stage = "output"
                os.replace(partial, output)
                self.last_report = report
                event("render.completed", output=output, seconds=round(time.monotonic() - started, 3), report=self.last_report)
                return output
            finally:
                try:
                    partial.unlink(missing_ok=True)
                    if s.cleanup_work:
                        shutil.rmtree(run_dir)
                    else:
                        event("work.retained", path=run_dir)
                except OSError as exc:
                    logger.warning("Work cleanup failed for %s: %s", run_dir, safe_error(exc))
