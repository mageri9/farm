from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import Settings
from src.pipeline import ShortsPipeline
from src.video import VideoError
from src.tts import TTSError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a vertical Shorts video from text")
    parser.add_argument("text_positional", nargs="*", help="Story text")
    parser.add_argument("--text", help="Story text")
    parser.add_argument("--input", type=Path, help="Read story text from file")
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--voice", default="ru-RU-DmitryNeural")
    parser.add_argument("--rate", default="+10%")
    parser.add_argument("--fps", type=int, choices=(30, 60), default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep-work", action="store_true", help="Keep work/audio.mp3 and subtitles.ass")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    if args.input:
        try:
            text = args.input.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR Could not read input file: {exc}", file=sys.stderr)
            return 2
    else:
        text = args.text or " ".join(args.text_positional)
    settings = Settings(voice=args.voice, rate=args.rate, fps=args.fps, cleanup_work=not args.keep_work)
    try:
        print("[CHECK] Validating environment...")
        pipeline = ShortsPipeline(settings)
        result = await pipeline.run(text, args.output, args.seed)
        report = pipeline.last_report or {}
        print("\n================================\nSUCCESS\n================================")
        print(f"Output {result}")
        print(f"Resolution {settings.video_width}x{settings.video_height}")
        print(f"FPS {report.get('fps', settings.fps):g}")
        print(f"Duration {report.get('duration', 0):.2f} sec")
        print(f"Audio {str(report.get('audio_codec', 'unknown')).upper()}")
        print("Subtitles burned-in")
        print("================================")
        return 0
    except (ValueError, TTSError, VideoError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
