from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .config import Settings
from .subtitles import escape_subtitle_path


class VideoError(RuntimeError):
    pass


def check_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        local = Path(__file__).resolve().parent.parent / (name + (".exe" if not name.endswith(".exe") else ""))
        if local.is_file():
            path = str(local)
    if not path:
        label = "FFmpeg" if name.lower() in {"ffmpeg", "ffprobe"} else name
        raise VideoError(f"{label} was not found. Install FFmpeg and add {name}.exe to PATH.")
    return path


def _run_probe(args: list[str], timeout: float = 30.0) -> dict:
    try:
        result = subprocess.run(
            args, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        raise VideoError(f"ffprobe failed: {exc}") from exc


def probe_duration(path: Path, settings: Settings | None = None) -> float:
    s = settings or Settings()
    data = _run_probe(
        [check_executable(s.ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        s.probe_timeout,
    )
    try:
        value = float(data["format"]["duration"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid duration")
        return value
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoError(f"Could not determine duration of {path}") from exc


def random_start(background_duration: float, segment_duration: float, seed: int | None = None) -> float:
    if background_duration + 1e-6 < segment_duration:
        return 0.0
    rng = random.Random(seed)
    available = max(0.0, background_duration - segment_duration - 0.5)
    return rng.uniform(0.0, available)


def render_video(
    backgrounds: Sequence[Path] | Path,
    audio: Path,
    subtitles: Path,
    output: Path,
    duration: float,
    start_offsets: Sequence[float] | float,
    settings: Settings,
    target_clips: int = 4,
) -> None:
    executable = check_executable(settings.ffmpeg)
    subtitle_path = escape_subtitle_path(subtitles)
    fonts_dir = escape_subtitle_path(settings.assets_dir / "fonts")
    subtitle_filter = f"ass='{subtitle_path}':fontsdir='{fonts_dir}'"

    bg_list = [backgrounds] if isinstance(backgrounds, Path) else list(backgrounds)
    starts = [start_offsets] if isinstance(start_offsets, (int, float)) else list(start_offsets)

    cmd = [executable, "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]

    crop_scale = (
        f"scale={settings.video_width}:{settings.video_height}:force_original_aspect_ratio=increase,"
        f"crop={settings.video_width}:{settings.video_height}:(in_w-{settings.video_width})/2:"
        f"(in_h-{settings.video_height})/2,setsar=1,fps={settings.fps}"
    )

    if target_clips < 1:
        raise ValueError("target_clips must be positive")
    if len(bg_list) <= 1:
        # Одиночный фон (классический режим)
        bg = bg_list[0]
        st = starts[0] if starts else 0.0
        if settings.loop_background:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-ss", f"{st:.3f}", "-i", str(bg), "-i", str(audio)]
        vf = f"{crop_scale},setpts=PTS-STARTPTS,{subtitle_filter}"
        cmd += ["-vf", vf, "-map", "0:v:0", "-map", "1:a:0"]
    else:
        # Мульти-клип: жесткий стык планов через trim
        cut_duration = duration / len(bg_list)
        filter_complex = []
        concat_inputs = ""

        for idx, bg in enumerate(bg_list):
            st = starts[idx] if idx < len(starts) else 0.0
            dur = (
                cut_duration
                if idx < len(bg_list) - 1
                else (duration - cut_duration * idx)
            )
            cmd += ["-stream_loop", "-1", "-ss", f"{st:.3f}", "-i", str(bg)]
            filter_complex.append(
                f"[{idx}:v]trim=duration={dur:.3f},{crop_scale},setpts=PTS-STARTPTS[v{idx}];"
            )
            concat_inputs += f"[v{idx}]"

        audio_idx = len(bg_list)
        cmd += ["-i", str(audio)]
        # Keep concat and subtitles in one chain so the labelled output remains connected.
        filter_complex.append(f"{concat_inputs}concat=n={len(bg_list)}:v=1:a=0,{subtitle_filter}[vout]")

        cmd += [
            "-filter_complex",
            "".join(filter_complex),
            "-map",
            "[vout]",
            "-map",
            f"{audio_idx}:a:0",
        ]

    cmd += [
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-r", str(settings.fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", settings.audio_bitrate,
        "-movflags", "+faststart", "-shortest", str(output),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=settings.render_timeout
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()[-8:]
        raise VideoError("FFmpeg render failed:\n" + "\n".join(detail)) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"FFmpeg exceeded {settings.render_timeout:g} seconds and was terminated") from exc
    except OSError as exc:
        raise VideoError(f"Could not start FFmpeg: {type(exc).__name__}") from exc


def validate_output(path: Path, settings: Settings, expected_duration: float | None = None) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise VideoError("Output validation failed: file is missing or empty.")
    data = _run_probe(
        [check_executable(settings.ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        settings.probe_timeout,
    )
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or video.get("width") != settings.video_width or video.get("height") != settings.video_height:
        raise VideoError(f"Output validation failed: expected {settings.video_width}x{settings.video_height} video stream.")
    if not audio:
        raise VideoError("Output validation failed: audio stream is missing.")
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise VideoError("Output validation failed: duration is zero.")
    if expected_duration is not None and abs(duration - expected_duration) > 0.5:
        raise VideoError("Output validation failed: duration does not match speech")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise VideoError("Output validation failed: expected H.264 video and AAC audio")
    fps_text = video.get("r_frame_rate", "0/1")
    try:
        num, den = fps_text.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 0.0
    if abs(fps - settings.fps) > 0.5:
        raise VideoError(f"Output validation failed: expected {settings.fps} FPS, got {fps:.2f}.")
    return {"duration": duration, "fps": fps, "audio_codec": audio.get("codec_name", "unknown")}
