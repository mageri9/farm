from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Settings
from src.pipeline import ShortsPipeline
from src.runtime import atomic_json, atomic_text, batch_status, log_failure, log_to


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("input/stories.json")
DEFAULT_OUTPUT_DIR = Path("output")
SEPARATOR = "=" * 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пакетная генерация Shorts-видео из JSON-файла"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON-файл с историями (по умолчанию: input/stories.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Базовая папка результатов (по умолчанию: output)",
    )
    parser.add_argument(
        "--channel-tag",
        default="@reddit_vault",
        help="Тег Telegram-канала для плана публикаций",
    )
    parser.add_argument("--voice", default="ru-RU-DmitryNeural")
    parser.add_argument("--rate", default="+10%")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve CLI paths relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_stories(path: Path) -> list[Any]:
    if not path.exists():
        raise ValueError(f"Файл с историями не найден: {path}")

    try:
        contents = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать файл с историями: {exc}") from exc

    if not contents.strip():
        raise ValueError(f"Файл с историями пуст: {path}")

    try:
        stories = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Некорректный JSON в файле {path}: строка {exc.lineno}, столбец {exc.colno}"
        ) from exc

    if not isinstance(stories, list):
        raise ValueError("JSON должен содержать массив историй")
    if not stories:
        raise ValueError(f"Массив историй пуст: {path}")
    return stories


def posting_block(index: int, filename: str, story: dict[str, Any], channel_tag: str) -> str:
    title = str(story.get("title") or "Без названия").strip()
    raw_tags = story.get("tags", [])
    if isinstance(raw_tags, list):
        tags = " ".join(str(tag).strip() for tag in raw_tags if str(tag).strip())
    else:
        tags = str(raw_tags).strip()
    return "\n".join(
        (
            SEPARATOR,
            f"[Ролик {index:02d}] Файл: {filename}",
            f"Название: {title}",
            f"Теги: {tags}",
            "Закрепленный комментарий:",
            "Знали об этом факте до видео? Делитесь мнением в комментариях.",
            SEPARATOR,
        )
    )


async def async_main(args: argparse.Namespace) -> int:
    input_path = project_path(args.input)
    try:
        stories = load_stories(input_path)
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2

    batch_dir = project_path(args.output_dir) / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        while True:
            try:
                batch_dir.mkdir(parents=True)
                break
            except FileExistsError:
                await asyncio.sleep(1)
                batch_dir = project_path(args.output_dir) / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    except OSError as exc:
        print(f"ОШИБКА: не удалось создать папку пачки {batch_dir}: {exc}", file=sys.stderr)
        return 2

    settings = Settings.from_env(voice=args.voice, rate=args.rate, fps=args.fps)
    pipeline = ShortsPipeline(settings)
    posting_blocks: list[str] = []
    successful = 0
    total = len(stories)
    failures = []
    posting_plan_path = batch_dir / "posting_plan.txt"

    for index, story in enumerate(stories, start=1):
        title = (
            str(story.get("title") or "Без названия").strip()
            if isinstance(story, dict)
            else "Некорректная история"
        )
        print(f'\n[BATCH {index}/{total}] Рендерим: "{title}"')
        video_path = batch_dir / f"video_{index:02d}.mp4"
        stage = "input"
        try:
            if not isinstance(story, dict):
                raise ValueError("история должна быть JSON-объектом")
            text = story.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("поле 'text' отсутствует или пусто")

            stage = "video"
            with log_to(batch_dir / "render.jsonl"):
                await pipeline.run(text, output_path=video_path)
        except Exception as exc:
            with log_to(batch_dir / "render.jsonl"):
                failures.append(log_failure(getattr(pipeline, "last_stage", "video") if stage == "video" else stage, exc,
                                            item=index, filename=video_path.name))
            continue

        successful += 1
        posting_blocks.append(posting_block(index, video_path.name, story, args.channel_tag))
        try:
            atomic_text(posting_plan_path, "\n\n".join(posting_blocks) + "\n")
        except OSError as exc:
            log_failure("posting_plan", exc, item=index)
            return 2
        print(f"[BATCH {index}/{total}] Готово: {video_path.name}")

    try:
        plan_text = "\n\n".join(posting_blocks)
        if plan_text:
            plan_text += "\n"
        atomic_text(posting_plan_path, plan_text)
        atomic_json(batch_dir / "batch_result.json", {
            "status": batch_status(successful, total), "successful": successful,
            "total": total, "failures": failures,
        })
    except OSError as exc:
        print(f"ОШИБКА: не удалось записать план публикаций: {exc}", file=sys.stderr)
        return 2

    print("\n" + SEPARATOR)
    print(f"Успешно сгенерировано {successful} из {total} роликов")
    print(f"Batch status: {batch_status(successful, total)}")
    print(f"Папка пачки: {batch_dir}")
    print(f"План публикаций: {posting_plan_path}")
    print(SEPARATOR)
    return 0 if successful == total else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
