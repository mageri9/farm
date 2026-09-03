from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .subtitles import escape_subtitle_path


class VideoError(RuntimeError):
    pass


def check_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        label = "FFmpeg" if name.lower() in {"ffmpeg", "ffprobe"} else name
        raise VideoError(f"{label} was not found. Install FFmpeg and add {name}.exe to PATH.")
    return path


def _run_probe(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise VideoError(f"ffprobe failed: {exc}") from exc


def probe_duration(path: Path) -> float:
    data = _run_probe([check_executable("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoError(f"Could not determine duration of {path}") from exc


def get_audio_duration(audio_path: Path) -> float:
    """Return duration via ffprobe using a descriptive public API."""
    return probe_duration(audio_path)


def random_start(background_duration: float, audio_duration: float, seed: int | None = None) -> float:
    if background_duration + 1e-6 < audio_duration:
        raise VideoError(f"Background video is too short. Required {audio_duration:.1f} sec; Available {background_duration:.1f} sec")
    rng = random.Random(seed)
    return rng.uniform(0.0, max(0.0, background_duration - audio_duration))


def calculate_random_start(background_duration: float, audio_duration: float, seed: int | None = None) -> float:
    return random_start(background_duration, audio_duration, seed)


def render_video(background: Path, audio: Path, subtitles: Path, output: Path, duration: float, start: float, settings: Settings) -> None:
    check_executable("ffmpeg")
    subtitle_filter = f"subtitles='{escape_subtitle_path(subtitles)}'"
    vf = f"crop=ih*9/16:ih:(iw-ow)/2:0,scale={settings.video_width}:{settings.video_height},{subtitle_filter}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
           "-i", str(background), "-i", str(audio), "-t", f"{duration:.3f}", "-vf", vf, "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf), "-r", str(settings.fps),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", settings.audio_bitrate, "-shortest", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()[-8:]
        raise VideoError("FFmpeg render failed:\n" + "\n".join(detail)) from exc


def validate_output(path: Path, settings: Settings) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise VideoError("Output validation failed: file is missing or empty.")
    data = _run_probe([check_executable("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or video.get("width") != settings.video_width or video.get("height") != settings.video_height:
        raise VideoError("Output validation failed: expected 1080x1920 video stream.")
    if not audio:
        raise VideoError("Output validation failed: audio stream is missing.")
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        raise VideoError("Output validation failed: duration is zero.")
    fps_text = video.get("r_frame_rate", "0/1")
    try:
        num, den = fps_text.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 0.0
    if abs(fps - settings.fps) > 0.5:
        raise VideoError(f"Output validation failed: expected {settings.fps} FPS, got {fps:.2f}.")
    return {"duration": duration, "fps": fps, "audio_codec": audio.get("codec_name", "unknown")}
