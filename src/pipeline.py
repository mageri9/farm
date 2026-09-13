from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from .config import Settings
from .runtime import event, file_lock, safe_error
from .subtitles import write_ass
from .tts import TTSError, generate_tts
from .video import check_executable, probe_duration, random_start, render_video, validate_output


class ShortsPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.last_report: dict | None = None

    def preflight(self):
        s = self.settings
        check_executable(s.ffmpeg)
        check_executable(s.ffprobe)
        has_pool = bool(self._background_candidates())
        if not has_pool and not s.background_path.is_file():
            raise ValueError(f"No MP4 in assets/backgrounds, assets or {s.background_path}")

    def _background_candidates(self) -> list[Path]:
        for directory in (self.settings.assets_dir / "backgrounds", self.settings.assets_dir):
            videos = sorted(path for path in directory.glob("*.mp4") if path.is_file())
            if videos:
                return videos
        return []

    def _select_backgrounds(self, total_duration: float, seed: int | None = None, target_clips: int = 4) -> tuple[list[Path], list[float]]:
        s = self.settings
        if target_clips < 1:
            raise ValueError("target_clips must be positive")
        # Ищем mp4 в assets/backgrounds/, а если пусто — прямо в assets/
        videos = self._background_candidates()

        if videos:
            rng = random.Random(seed)
            selected = rng.sample(videos, target_clips) if len(videos) >= target_clips else rng.choices(videos, k=target_clips)
            clip_duration = total_duration / target_clips
            offsets = []
            for idx, bg in enumerate(selected):
                dur = probe_duration(bg, s)
                offsets.append(random_start(dur, clip_duration, None if seed is None else seed + idx))
            print(f"[VIDEO] Найдено {len(videos)} видео. Для ролика выбраны {target_clips} плана: " + " -> ".join(p.name for p in selected))
            return selected, offsets

        # Резерв на одиночный фон
        background = videos[0] if videos else s.background_path
        bg_dur = probe_duration(background, s)
        st = 0.0 if s.loop_background and bg_dur < total_duration else random_start(bg_dur, total_duration, seed)
        print(f"[VIDEO] Используется одиночный фон: {background.name}")
        return [background], [st]

    async def run(self, text: str, output_path: str | Path | None = None, seed: int | None = None) -> Path:
        s = self.settings
        self.last_report = None
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
                event("tts.started", output=output)
                for attempt in range(s.retry_attempts):
                    try:
                        async with asyncio.timeout(s.tts_timeout):
                            boundaries = await generate_tts(text, audio, s.voice, s.rate)
                        break
                    except (TTSError, TimeoutError) as exc:
                        event("tts.failed", attempt=attempt + 1, error=safe_error(exc), output=output)
                        if attempt + 1 == s.retry_attempts:
                            raise TTSError(f"TTS failed after {s.retry_attempts} attempts ({type(exc).__name__})") from exc
                        await asyncio.sleep(min(60, s.retry_delay * 2 ** attempt))
                event("tts.completed", words=len(boundaries), seconds=round(time.monotonic() - started, 3))
                write_ass(boundaries, ass, s.words_per_subtitle, s.font_name, s.font_size, s.assets_dir / "fonts")
                duration = probe_duration(audio, s)

                bg_list, start_offsets = self._select_backgrounds(duration, seed)

                event("render.started", output=output, duration=duration, clips=len(bg_list))
                render_video(bg_list, audio, ass, partial, duration, start_offsets, s)
                self.last_report = validate_output(partial, s, expected_duration=duration)
                os.replace(partial, output)
                event("render.completed", output=output, seconds=round(time.monotonic() - started, 3), report=self.last_report)
                return output
            finally:
                partial.unlink(missing_ok=True)
                if s.cleanup_work:
                    shutil.rmtree(run_dir)
                else:
                    event("work.retained", path=run_dir)
