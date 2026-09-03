from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import Settings
from .subtitles import write_ass
from .tts import TTSError, generate_tts
from .video import VideoError, check_executable, get_audio_duration, probe_duration, random_start, render_video, validate_output


class ShortsPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.last_report: dict | None = None

    async def run(self, text: str, output_path: str | Path | None = None, seed: int | None = None) -> Path:
        text = (text or "").strip()
        if len(text) < 3:
            raise ValueError("Text is empty or too short (minimum 3 characters).")
        print("[1/5] Validating environment...")
        check_executable("ffmpeg")
        print("[CHECK] ffmpeg OK")
        check_executable("ffprobe")
        print("[CHECK] ffprobe OK")
        s = self.settings
        s.work_dir.mkdir(parents=True, exist_ok=True)
        s.output_dir.mkdir(parents=True, exist_ok=True)
        output = Path(output_path) if output_path else s.output_dir / f"final_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        if not output.is_absolute():
            output = s.root / output
        if not s.background_path.exists():
            raise VideoError(f"Background video not found: {s.background_path}")
        audio = s.work_dir / "audio.mp3"
        ass = s.work_dir / "subtitles.ass"
        print("[2/5] Generating voice...")
        print(f"[TTS] Voice {s.voice}")
        boundaries = await generate_tts(text, audio, s.voice, s.rate)
        print(f"[TTS] Audio saved {audio}")
        print(f"[TTS] Word boundaries {len(boundaries)}")
        print("[3/5] Generating subtitles...")
        event_count = write_ass(boundaries, ass, s.words_per_subtitle, s.font_name, s.font_size, s.assets_dir / "fonts")
        print(f"[SUBTITLES] Events {event_count}")
        audio_duration = get_audio_duration(audio)
        background_duration = probe_duration(s.background_path)
        start = random_start(background_duration, audio_duration, seed)
        print(f"[VIDEO] Segment start {start:.2f} sec")
        print("[4/5] Rendering video...")
        render_video(s.background_path, audio, ass, output, audio_duration, start, s)
        print("[5/5] Validating output...")
        self.last_report = validate_output(output, s)
        if s.cleanup_work:
            for path in (audio, ass):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return output
