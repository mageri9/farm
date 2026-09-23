from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import ExitStack

from dotenv import load_dotenv

from src.config import Settings
from src.content.generator import UnifiedStoryGenerator
# Kept importable for callers that patched the legacy classes in older integrations.
from src.content.adapter import StoryAdapter
from src.content.researcher import FactResearcher
from src.pipeline import ShortsPipeline
from src.runtime import atomic_json, atomic_text, batch_status, log_failure, log_to, safe_error
from src.video import find_item_images

ROOT = Path(__file__).resolve().parent


def slugify(value: str) -> str:
    """Имя папки ассетов из темы: пробелы в дефисы, только безопасные символы."""
    slug = re.sub(r"[\s_]+", "-", (value or "").strip().casefold())
    slug = re.sub(r"[^\w\-]", "", slug, flags=re.UNICODE).strip("-")
    return slug


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-button factual Shorts generator")
    p.add_argument("topic_or_slug", nargs="*", default=[],
                   help="Тема/slug предмета, например: python generate.py клевец")
    p.add_argument("--count", type=positive_int, default=3)
    p.add_argument("--category", choices=["all", "systems", "science", "mind"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--voice", default="ru-RU-DmitryNeural")
    p.add_argument("--rate", default="+10%")
    p.add_argument("--fps", type=int, choices=[30, 60], default=30)
    p.add_argument("--story", help="Готовый текст сценария")
    p.add_argument("--story-file", help="Путь к TXT-файлу со сценарием")
    p.add_argument("--title", help="Заголовок истории")
    return p.parse_args(argv)


def posting(story: dict, filename: str, index: int, status: str) -> str:
    tags = " ".join(story.get("tags", []))
    return ("=" * 40 + f"\n[Ролик {index:02d}] Файл: {filename}\n"
            f"Статус: {status}\nНазвание: {story.get('title', 'Без названия')}\nТеги: {tags}\n"
            "Закрепленный комментарий:\nКакой вывод из этого факта кажется самым неожиданным?\n" + "=" * 40)


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    if args.count < 1:
        print("[ERROR] --count должен быть положительным")
        return 2
    started = time.monotonic()
    batch = None
    generator = None
    logs = ExitStack()
    successful = 0
    failures = []
    stage = "setup"

    def progress(stage: str, message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] [{stage}] {message}", flush=True)

    try:
        settings = Settings.from_env(root=ROOT, voice=args.voice, rate=args.rate, fps=args.fps)
        pipeline = None
        if not args.dry_run:
            pipeline = ShortsPipeline(settings)
            pipeline.preflight()
        topic = " ".join(args.topic_or_slug).strip()
        slug = slugify(topic)
        overlays = find_item_images(settings.assets_dir, slug) if slug else []
        if topic:
            progress("ASSETS", f"Предмет «{topic}» (slug: {slug}); картинок найдено: {len(overlays)}"
                     + (f" -> {', '.join(p.name for p in overlays)}" if overlays else " (рендер на одних фонах)"))
        custom_story = None
        if args.story_file:
            custom_story = Path(args.story_file).read_text(encoding="utf-8")
        elif args.story is not None:
            custom_story = args.story
        custom_mode = custom_story is not None
        if not custom_mode and not os.getenv("ANYMODEL_API_KEY"):
            raise ValueError("Не задан ANYMODEL_API_KEY в .env")
        if custom_mode:
            custom_story = custom_story.strip()
            if not custom_story:
                raise ValueError("Сценарий не может быть пустым")
            args.count = 1
        batch = ROOT / "output" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        # Never reuse a previous batch, including simultaneous starts in the same second.
        while True:
            try:
                batch.mkdir(parents=True)
                break
            except FileExistsError:
                await asyncio.sleep(1)
                batch = ROOT / "output" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        logs.enter_context(log_to(batch / "render.jsonl"))
        target = topic or args.category
        progress("START", f"Батч {batch.name}: {'готовый сценарий' if custom_mode else f'ищем {args.count} фактов ({target})'}")
        if custom_mode:
            facts = [{"title": args.title or topic or " ".join(custom_story.split()[:7]), "raw_data": custom_story,
                      "topic": args.title or topic or "custom", "category": args.category, "sources": []}]
        else:
            stage = "generation"
            generator = UnifiedStoryGenerator(history_path=ROOT / "work" / "facts_history.json")
            facts = await generator.generate(args.count, args.category, topic=topic or None)
            atomic_json(batch / "facts.json", facts)
            progress("GENERATION", f"Создано историй: {len(facts)}")
            if not facts:
                raise RuntimeError("Generator returned no stories")
        stories = []

        def save_plan() -> None:
            atomic_json(batch / "stories.json", stories)
            blocks = [posting(s, s["filename"], int(Path(s["filename"]).stem.split("_")[-1]), s["status"])
                      for s in stories if s["status"] in ("rendered", "dry-run")]
            atomic_text(batch / "posting_plan.txt", "\n\n".join(blocks) + "\n")

        save_plan()
        for i, fact in enumerate(facts, 1):
            story = None
            stage = "adaptation"
            try:
                progress(f"SCRIPT {i}/{args.count}", f"Processing item {i}")
                if custom_mode:
                    story = {"title": args.title or fact["title"], "text": custom_story,
                             "tags": ["#шортс", "#история", "#факты"], "topic": fact["topic"],
                             "category": fact["category"], "sources": []}
                else:
                    story = dict(fact)
                    story["sources"] = [{"url": fact.get("source_url", ""), "title": fact.get("topic", "")}]
                story.update(filename=f"video_{i:02d}.mp4", status="dry-run" if args.dry_run else "pending")
                if slug:
                    story["item_slug"] = slug
                stories.append(story)
                stage = "script.output"
                atomic_text(batch / f"story_{i:02d}.txt", story["text"] + "\n")
                atomic_json(batch / "stories.json", stories)
                progress(f"SCRIPT {i}/{args.count}", f"{len(story['text'].split())} слов")
                if pipeline is not None:
                    stage = "video"
                    progress(f"VIDEO {i}/{args.count}", f"Озвучка и рендер {story['filename']}")
                    await pipeline.run(story["text"], output_path=batch / story["filename"],
                                       overlays=overlays)
                    story.update(status="rendered", report=pipeline.last_report)
                    progress(f"VIDEO {i}/{args.count}", f"Готово: {story['filename']}; {pipeline.last_report}")
                successful += 1
            except Exception as exc:
                failed_stage = getattr(pipeline, "last_stage", "video") if stage == "video" else stage
                failures.append(log_failure(failed_stage, exc, item=i, filename=f"video_{i:02d}.mp4"))
                if story is not None:
                    story.update(status="failed", error=safe_error(exc))
            stage = "posting_plan"
            save_plan()
            atomic_json(batch / "batch_result.json", {
                "status": batch_status(successful, args.count), "successful": successful,
                "total": args.count, "failures": failures, "dry_run": args.dry_run,
            })
        status = batch_status(successful, args.count)
        progress("DONE", f"{status}: {successful}/{args.count} successful; {batch}")
        return 0 if status == "SUCCESS" else 1
    except Exception as exc:
        failure = log_failure(stage, exc)
        status = "PARTIAL_SUCCESS" if successful else "FAILED"
        if batch is not None and batch.exists():
            try:
                atomic_json(batch / "error.json", {"error": safe_error(exc)})
                atomic_json(batch / "batch_result.json", {
                    "status": status, "successful": successful, "total": args.count,
                    "failures": [*failures, failure], "dry_run": args.dry_run,
                })
            except OSError as report_error:
                progress("ERROR", f"Cannot save failure report: {safe_error(report_error)}")
        progress("ERROR", f"{status}: {safe_error(exc)}")
        return 1
    finally:
        for client in (generator,):
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    log_failure("client.close", exc)
        logs.close()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    try:
        return asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("[STOP] Генерация прервана; сохраненные результаты находятся в output.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
