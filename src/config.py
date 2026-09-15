from dataclasses import dataclass
from pathlib import Path
import os
import math
from dataclasses import fields
from dotenv import load_dotenv

PRIMARY_MODEL = "ag/gemini-3.7-flash-medium"
FALLBACK_MODEL = "ag/claude-sonnet-4-6"


def llm_models() -> tuple[str, str]:
    return (os.getenv("PRIMARY_MODEL") or os.getenv("ANYMODEL_MODEL") or PRIMARY_MODEL,
            os.getenv("FALLBACK_MODEL") or FALLBACK_MODEL)


@dataclass
class Settings:
    voice: str = "ru-RU-DmitryNeural"
    rate: str = "+10%"
    words_per_subtitle: int = 2
    fps: int = 30
    video_width: int = 1080
    video_height: int = 1920
    crf: int = 22
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    font_name: str = "Montserrat"
    font_size: int = 80
    cleanup_work: bool = True
    root: Path = Path(__file__).resolve().parent.parent
    background: Path | None = None
    ambient_audio: Path | None = None
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    probe_timeout: float = 30.0
    render_timeout: float = 1800.0
    tts_timeout: float = 30.0
    request_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 3.0
    max_text_chars: int = 20000
    loop_background: bool = True
    anyvoice_url: str = "https://anymodel.org/api/anyvoice/tts"
    anyvoice_reference_id: str = "dcd50b837c724daf898bb375c6ae4537"
    anyvoice_cookie: str = ""
    tts_provider: str = "anyvoice"

    def __post_init__(self):
        self.root = Path(self.root)
        if self.background is not None:
            self.background = Path(self.background)
        if self.ambient_audio is not None:
            self.ambient_audio = Path(self.ambient_audio)
        for name in ("probe_timeout", "render_timeout", "tts_timeout", "request_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.retry_delay) or not 0 <= self.retry_delay <= 60:
            raise ValueError("retry_delay must be between 0 and 60")
        if not 1 <= self.retry_attempts <= 10:
            raise ValueError("retry_attempts must be between 1 and 10")
        if self.fps not in (30, 60):
            raise ValueError("fps must be 30 or 60")
        if any(v < 2 or v % 2 for v in (self.video_width, self.video_height)):
            raise ValueError("Video dimensions must be positive and even")
        if not 0 <= self.crf <= 51 or self.words_per_subtitle < 1 or self.font_size < 1:
            raise ValueError("Invalid encoding or subtitle settings")
        if self.max_text_chars < 3:
            raise ValueError("max_text_chars must be at least 3")

    @classmethod
    def from_env(cls, **overrides):
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        defaults = cls()
        values = {}
        for field in fields(cls):
            raw = os.getenv("SHORTS_" + field.name.upper())
            if raw is None and field.name in {"anyvoice_reference_id", "anyvoice_cookie"}:
                raw = os.getenv("ANYVOICE_" + field.name.removeprefix("anyvoice_").upper())
            if raw is None:
                continue
            default = getattr(defaults, field.name)
            if isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"SHORTS_{field.name.upper()} must be true or false")
                values[field.name] = raw.lower() in {"true", "1"}
            elif field.name in {"root", "background", "ambient_audio"}:
                values[field.name] = Path(raw)
            else:
                values[field.name] = type(default)(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def background_path(self) -> Path:
        if self.background is not None:
            return self.background if self.background.is_absolute() else self.root / self.background
        return self.assets_dir / "background.mp4"

    @property
    def ambient_path(self) -> Path:
        if self.ambient_audio is not None:
            return self.ambient_audio if self.ambient_audio.is_absolute() else self.root / self.ambient_audio
        return self.assets_dir / "audio" / "ambient.mp3"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def primary_model(self) -> str:
        return llm_models()[0]

    @property
    def fallback_model(self) -> str:
        return llm_models()[1]
