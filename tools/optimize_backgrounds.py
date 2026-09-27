"""Normalize background videos for fast rendering.

Run from the repository root with ``python tools/optimize_backgrounds.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov"}


def find_binary(name: str, root: Path) -> str:
    """Prefer binaries shipped with the repository, then use PATH."""
    local_name = f"{name}.exe" if os.name == "nt" else name
    local = root / local_name
    return str(local) if local.exists() else (shutil.which(name) or name)


def probe(path: Path, ffprobe: str) -> tuple[int, int, float, bool]:
    command = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_streams", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    data = json.loads(result.stdout)
    all_streams = data.get("streams", [])
    streams = [s for s in all_streams if s.get("codec_type") == "video"]
    if not streams:
        raise RuntimeError("no video stream found")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    has_audio = any(s.get("codec_type") == "audio" for s in all_streams)
    return int(stream.get("width", 0)), int(stream.get("height", 0)), fps, has_audio


def mb(size: int) -> float:
    return size / (1024 * 1024)


def optimize(path: Path, ffmpeg: str) -> int:
    temp = path.with_name(f"temp_opt_{path.name}.mp4")
    # Exclusive creation avoids clobbering a stale file or another running job.
    with temp.open("xb"):
        pass
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path),
        "-map", "0:v:0",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-an", "-movflags", "+faststart", str(temp),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed"
            raise RuntimeError(detail)
        new_size = temp.stat().st_size
        if new_size == 0:
            raise RuntimeError("ffmpeg produced an empty file")
        os.replace(temp, path)
        return new_size
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="background directory (default: assets/backgrounds)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = (args.directory or root / "assets" / "backgrounds").resolve()
    if not directory.is_dir():
        print(f"Background directory does not exist: {directory}", file=sys.stderr)
        return 1
    ffmpeg = find_binary("ffmpeg", root)
    ffprobe = find_binary("ffprobe", root)
    files = sorted(p for p in directory.iterdir() if p.is_file()
                   and p.suffix.lower() in VIDEO_EXTENSIONS
                   and not p.name.lower().startswith("temp_opt_"))
    optimized = skipped = failed = 0
    saved = 0
    total = len(files)
    for index, path in enumerate(files, 1):
        started = time.monotonic()
        try:
            original_size = path.stat().st_size
            width, height, fps, has_audio = probe(path, ffprobe)
            if width == 1080 and height == 1920 and 0 < fps <= 30 and not has_audio:
                skipped += 1
                print(f"[SKIP {index}/{total}] {path.name} (already optimized)", flush=True)
                continue
            print(f"[OPT {index}/{total}] {path.name}: encoding...", flush=True)
            new_size = optimize(path, ffmpeg)
            optimized += 1
            saved += original_size - new_size
            print(f"[OPT {index}/{total}] {path.name} ({width}x{height}, {fps:.2f}fps) -> "
                  f"1080x1920 30fps: {mb(original_size):.1f}MB -> {mb(new_size):.1f}MB "
                  f"(Done in {time.monotonic() - started:.1f}s)", flush=True)
        except Exception as exc:  # keep processing the remaining pool
            failed += 1
            print(f"[ERROR {index}/{total}] {path.name}: {exc}", file=sys.stderr, flush=True)
    print(f"\nSummary: optimized {optimized}, skipped {skipped}, failed {failed}; "
          f"saved {mb(saved):.1f}MB.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
