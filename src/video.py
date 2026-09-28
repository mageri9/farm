from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
import os
import time
import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .config import Settings
from .subtitles import escape_subtitle_path

logger = logging.getLogger("shorts")


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


OVERLAY_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@lru_cache(maxsize=1)
def _cutout_session():
    """Create rembg's ONNX session once per process."""
    from rembg import new_session
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return new_session(model_name="isnet-general-use", sess_opts=opts)


def find_item_images(assets_dir: Path, slug: str) -> list[Path]:
    """Ассеты предмета: assets/items/{slug}/*.ext по имени, иначе assets/items/{slug}.ext.

    Fail-soft: при отсутствии файлов возвращает пустой список, рендер идет на одних фонах.
    """
    slug = (slug or "").strip()
    if not slug or slug != Path(slug).name or slug in {".", ".."}:
        return []
    items = Path(assets_dir) / "items"
    folder = items / slug
    try:
        if folder.is_dir():
            pngs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
            raw = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".webp"}), key=lambda p: p.name.casefold())
            numbered = [p for p in pngs if p.stem.isdigit() and int(p.stem) > 0]
            if numbered:
                return sorted(numbered, key=lambda p: (int(p.stem), p.name))
            raw = [p for p in raw if p.stem.casefold() not in {png.stem.casefold() for png in pngs}]
            if raw and not numbered:
                try:
                    from rembg import remove
                    from PIL import Image
                    print(f"[AUTO-CUTOUT] Вырезаем фон для {len(raw)} изображений предмета...", flush=True)
                    # Publish only after the entire batch succeeds, so a failed cutout can be retried.
                    session = _cutout_session()
                    with tempfile.TemporaryDirectory(dir=folder) as staging:
                        for index, source in enumerate(raw, 1):
                            resize_started = time.perf_counter()
                            with Image.open(source) as image:
                                image.load()
                                max_dim = 1024
                                if max(image.size) > max_dim:
                                    image.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
                                resize_time = time.perf_counter() - resize_started
                                inference_started = time.perf_counter()
                                result = remove(image, session=session)
                                output_img = result if isinstance(result, Image.Image) else Image.open(result)
                                output_img = output_img.convert("RGBA")
                                bbox = output_img.getchannel("A").getbbox()
                                if bbox:
                                    pad = 8
                                    w, h = output_img.size
                                    padded_bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                                                   min(w, bbox[2] + pad), min(h, bbox[3] + pad))
                                    output_img = output_img.crop(padded_bbox)
                                inference_time = time.perf_counter() - inference_started
                                save_started = time.perf_counter()
                                output_img.save(Path(staging) / f"{index}.png")
                                save_time = time.perf_counter() - save_started
                            print(f"[CUTOUT] {source.name} -> {index}.png: resize={resize_time:.2f}s, "
                                  f"inference={inference_time:.2f}s, save={save_time:.2f}s", flush=True)
                        converted = [folder / f"{index}.png" for index in range(1, len(raw) + 1)]
                        for target in converted:
                            (Path(staging) / target.name).replace(target)
                    return converted
                except Exception as exc:
                    # Keep the original assets usable when optional cutout dependencies fail.
                    print(f"[AUTO-CUTOUT] Ошибка: {exc}; используем исходные изображения", flush=True)
                    return raw
            found = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in OVERLAY_SUFFIXES]
            if pngs:
                return sorted(pngs, key=lambda p: p.name.casefold())
            if found:
                return sorted(found, key=lambda p: p.name.casefold())
        return [p for suffix in OVERLAY_SUFFIXES if (p := items / f"{slug}{suffix}").is_file()]
    except OSError:
        return []


def overlay_spans(num_images: int, total_duration: float) -> list[tuple[float, float]]:
    """Return (start, end) windows, clipped to the video duration."""
    if num_images < 1 or total_duration <= 0 or not math.isfinite(total_duration):
        return []
    start_global = 0.0
    end_global = total_duration
    total_window = end_global - start_global
    if total_window <= 0:
        return []
    span_duration = total_window / num_images
    return [
        (start_global + i * span_duration, start_global + (i + 1) * span_duration)
        for i in range(num_images)
    ]


def _overlay_chain(base: str, images: Sequence[Path], spans: Sequence[tuple[float, float]],
                   first_input: int, settings: Settings) -> tuple[list[str], str]:
    """Строит filter_complex для наложения картинок поверх метки base с макро-проездом на 2-м ракурсе."""
    from PIL import Image

    parts: list[str] = []
    current = base
    default_pos_x = "(W-w)/2"
    default_pos_y = "(H-h)/2-140"

    for order, (img_path, (start, end)) in enumerate(zip(images, spans)):
        idx = first_input + order
        span = end - start
        fade = min(settings.overlay_fade, span / 2.0)
        label = f"ov{order}"

        # Проверяем габариты изображения
        try:
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
        except Exception:
            orig_w, orig_h = 1000, 2000

        aspect = orig_h / max(1, orig_w)
        # 2-й кадр (order == 1) при вертикальной ориентации (aspect >= 1.2) делает вертикальный макро-проезд
        is_vertical_macro = (order == 1 and aspect >= 1.2)

        if is_vertical_macro:
            target_h = int(settings.video_height * 1.45)
            target_w = int(orig_w * (target_h / orig_h))
            max_w = int(settings.video_width * 0.82)
            if target_w > max_w:
                target_w = max_w
                target_h = int(orig_h * (target_w / orig_w))
            target_w = target_w - (target_w % 2)
            target_h = target_h - (target_h % 2)

            # Кинематика: сверху вниз
            y_start = 70
            y_end = (settings.video_height - 280) - target_h

            pos_x_expr = "(W-w)/2"
            pos_y_expr = f"'{y_start}+({y_end}-({y_start}))*clip((t-{start:.3f})/{span:.3f},0,1)'"
            sh_x_expr = "(W-w)/2+14"
            sh_y_expr = f"'{y_start+18}+({y_end}-({y_start}))*clip((t-{start:.3f})/{span:.3f},0,1)'"

            chain = (f"[{idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,format=rgba,"
                     f"trim=duration={span:.3f},setpts=PTS-STARTPTS,setsar=1")
        else:
            pos_x_expr = default_pos_x
            pos_y_expr = default_pos_y
            sh_x_expr = f"{default_pos_x}+14"
            sh_y_expr = f"{default_pos_y}+18"

            max_overlay_w = int(settings.video_width * 0.78)
            max_overlay_h = int(settings.video_height * 0.72)
            chain = (f"[{idx}:v]scale='min({max_overlay_w},iw)':'min({max_overlay_h},ih)':"
                     f"force_original_aspect_ratio=decrease,format=rgba,"
                     f"trim=duration={span:.3f},setpts=PTS-STARTPTS,"
                     f"scale=eval=frame:w='max(2,trunc(iw*(1+0.04*clip(t/{span:.3f},0,1))/2)*2)':h=-1,setsar=1")

        if fade > 0:
            if order == 0:
                chain += f",fade=t=out:st={span - fade:.3f}:d={fade:.3f}:alpha=1"
            else:
                chain += (f",fade=t=in:st=0:d={fade:.3f}:alpha=1"
                          f",fade=t=out:st={span - fade:.3f}:d={fade:.3f}:alpha=1")

        if start > 0:
            chain += f",tpad=start_duration={start:.3f}:start_mode=add:color=0x00000000"

        shadow = f"sh{order}"
        parts.append(chain + f",split[{label}][{shadow}];")
        blurred = f"blursh{order}"
        parts.append(f"[{shadow}]scale=iw/2:ih/2,colorchannelmixer=rr=0:gg=0:bb=0:aa=0.55,boxblur=6:6,scale=iw*2:ih*2[{blurred}];")
        with_shadow = f"ovshadow{order}"
        parts.append(f"[{current}][{blurred}]overlay={sh_x_expr}:{sh_y_expr}:eval=frame:"
                     f"enable='between(t,{start:.3f},{end:.3f})':eof_action=pass[{with_shadow}];")
        nxt = f"ovout{order}"
        parts.append(f"[{with_shadow}][{label}]overlay={pos_x_expr}:{pos_y_expr}:eval=frame:"
                     f"enable='between(t,{start:.3f},{end:.3f})':eof_action=pass[{nxt}];")
        current = nxt

    return parts, current
def render_video(
    backgrounds: Sequence[Path] | Path,
    audio: Path,
    subtitles: Path,
    output: Path,
    duration: float,
    start_offsets: Sequence[float] | float,
    settings: Settings,
    target_clips: int = 4,
    clip_durations: Sequence[float] | float | None = None,
    overlays: Sequence[Path] | None = None,
) -> None:
    executable = check_executable(settings.ffmpeg)
    subtitle_path = escape_subtitle_path(subtitles)
    fonts_dir = escape_subtitle_path(settings.assets_dir / "fonts")
    subtitle_filter = f"ass=filename='{subtitle_path}':fontsdir='{fonts_dir}'"
    cinematic_grade = "eq=contrast=1.35:brightness=-0.16:saturation=0.32,vignette=PI/3"

    bg_list = [backgrounds] if isinstance(backgrounds, Path) else list(backgrounds)
    starts = [start_offsets] if isinstance(start_offsets, (int, float)) else list(start_offsets)
    ambient = settings.ambient_path
    has_ambient = ambient.is_file()
    impact = settings.assets_dir / "sfx" / "impact.wav"
    has_impact = impact.is_file()
    has_audio_mix = has_ambient
    # Fail-soft: use at most four existing images.
    overlay_images = [p for p in list(overlays or [])[:4] if Path(p).is_file()]

    cmd = [executable, "-nostdin", "-y", "-hide_banner", "-nostats", "-loglevel", "warning"]

    crop_scale = (
        f"scale={settings.video_width}:{settings.video_height}:force_original_aspect_ratio=increase,"
        f"crop={settings.video_width}:{settings.video_height}:(in_w-{settings.video_width})/2:"
        f"(in_h-{settings.video_height})/2,setsar=1,fps={settings.fps}"
    )

    if not bg_list:
        raise ValueError("At least one background is required")
    if target_clips < 1:
        raise ValueError("target_clips must be positive")
    if len(bg_list) == 1 and not overlay_images:
        # Одиночный фон (классический режим)
        bg = bg_list[0]
        st = starts[0] if starts else 0.0
        if settings.loop_background:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-ss", f"{st:.3f}", "-i", str(bg), "-i", str(audio)]
        if has_ambient:
            cmd += ["-stream_loop", "-1", "-i", str(ambient)]
        vf = f"{crop_scale},setpts=PTS-STARTPTS,{cinematic_grade},{subtitle_filter}"
        if has_ambient:
            cmd += [
                "-vf", vf, "-filter_complex",
                "[2:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.08[amb];"
                "[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[main];"
                "[main][amb]amix=inputs=2:duration=first:dropout_transition=0,"
                "loudnorm=I=-14:LRA=7:TP=-1.5[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        else:
            cmd += ["-vf", vf, "-map", "0:v:0", "-map", "1:a:0"]
    else:
        # Мульти-клип: жесткий стык планов через trim
        if clip_durations is None:
            durations = [duration / len(bg_list)] * len(bg_list)
        elif isinstance(clip_durations, (int, float)):
            durations = [float(clip_durations)] * len(bg_list)
        else:
            durations = list(clip_durations)
        if len(durations) != len(bg_list) or any(not math.isfinite(d) or d <= 0 for d in durations):
            raise ValueError("clip_durations must contain one positive finite duration per clip")
        filter_complex = []
        concat_inputs = ""

        for idx, bg in enumerate(bg_list):
            st = starts[idx] if idx < len(starts) else 0.0
            dur = durations[idx]
            cmd += ["-stream_loop", "-1", "-ss", f"{st:.3f}", "-i", str(bg)]
            filter_complex.append(
                f"[{idx}:v]trim=duration={dur},{crop_scale},setpts=PTS-STARTPTS[v{idx}];"
            )
            concat_inputs += f"[v{idx}]"

        audio_idx = len(bg_list)
        cmd += ["-i", str(audio)]
        ambient_idx = audio_idx + 1
        if has_ambient:
            cmd += ["-stream_loop", "-1", "-i", str(ambient)]

        # Картинки предмета идут последними, чтобы не сдвигать индексы аудио-входов.
        spans = overlay_spans(len(overlay_images), duration)
        overlay_images = overlay_images[:len(spans)]
        overlay_first_idx = (ambient_idx + 1) if has_ambient else (audio_idx + 1)
        for image, (start, end) in zip(overlay_images, spans):
            cmd += ["-loop", "1", "-t", f"{end - start:.3f}", "-i", str(image)]
        impact_idx = overlay_first_idx + len(overlay_images)
        if overlay_images and has_impact:
            cmd += ["-i", str(impact)]

        if overlay_images:
            # Картинка ложится на фон до прожига субтитров, чтобы не перекрывать караоке.
            filter_complex.append(f"{concat_inputs}concat=n={len(bg_list)}:v=1:a=0,{cinematic_grade}[vbg];")
            chain, last = _overlay_chain("vbg", overlay_images, spans, overlay_first_idx, settings)
            filter_complex += chain
            filter_complex.append(f"[{last}]{subtitle_filter}[vout]")
        else:
            # Keep concat and subtitles in one chain so the labelled output remains connected.
            filter_complex.append(f"{concat_inputs}concat=n={len(bg_list)}:v=1:a=0,{cinematic_grade},{subtitle_filter}[vout]")
        has_audio_mix = has_ambient or bool(overlay_images and has_impact)
        if has_audio_mix:
            filter_complex.append(
                f";[{audio_idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[main]"
            )
            mix_inputs = ["[main]"]
            if has_ambient:
                filter_complex.append(
                    f";[{ambient_idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.08[amb]"
                )
                mix_inputs.append("[amb]")
            if overlay_images and has_impact:
                # Split the single impact stream before applying independent delays.
                filter_complex.append(
                    f";[{impact_idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"asplit={len(spans)}" + "".join(f"[sfx{n}]" for n in range(len(spans)))
                )
                for order, (start, _end) in enumerate(spans[:len(overlay_images)]):
                    delay_ms = max(0, int(round(start * 1000)))
                    label = f"impact{order}"
                    filter_complex.append(
                        f";[sfx{order}]adelay={delay_ms}:all=1,volume=0.12[{label}]"
                    )
                    mix_inputs.append(f"[{label}]")
            filter_complex.append(
                ";" + "".join(mix_inputs) +
                f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0,"
                "loudnorm=I=-14:LRA=7:TP=-1.5[aout]"
            )

        cmd += [
            "-filter_complex",
            "".join(filter_complex),
            "-map",
            "[vout]",
            "-map",
            "[aout]" if has_audio_mix else f"{audio_idx}:a:0",
        ]

    if not has_audio_mix:
        cmd += ["-af", "loudnorm=I=-14:LRA=7:TP=-1.5"]

    cmd += [
        "-t", f"{duration:.3f}",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
        "-r", str(settings.fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", settings.audio_bitrate,
        "-movflags", "+faststart", "-shortest", str(output),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("FFmpeg command: %s", subprocess.list2cmdline(cmd))
    try:
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            try:
                subprocess.run(
                    cmd, check=True, stdout=subprocess.DEVNULL, stderr=stderr_file,
                    timeout=settings.render_timeout
                )
            except subprocess.CalledProcessError as exc:
                # Keep only a bounded diagnostic tail in memory, even after log floods.
                stderr_file.seek(0, os.SEEK_END)
                stderr_file.seek(max(0, stderr_file.tell() - 65536))
                exc.stderr = stderr_file.read(65536).decode("utf-8", errors="replace")
                raise
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

