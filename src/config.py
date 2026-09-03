from dataclasses import dataclass
from pathlib import Path


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

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def background_path(self) -> Path:
        return self.assets_dir / "background.mp4"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

