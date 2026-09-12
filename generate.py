from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config import Settings
from src.content.adapter import AdaptedStory, StoryAdapter
from src.content.researcher import FactResearcher
from src.pipeline import ShortsPipeline
from src.runtime import atomic_json, atomic_text, log_to, safe_error

ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-button factual Shorts generator")
    p.add_argument("--count", type=positive_int, default=3)
    p.add_argument("--category", choices=["all", "systems", "science", "mind"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--voice", default="ru-RU-DmitryNeural")
    p.add_argument("--rate", default="+10%")
    p.add_argument("--fps", type=int, choices=[30, 60], default=30)
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
    researcher = adapter = None

    def progress(stage: str, message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] [{stage}] {message}", flush=True)

    try:
        settings = Settings.from_env(root=ROOT, voice=args.voice, rate=args.rate, fps=args.fps)
        pipeline = None
        if not args.dry_run:
            pipeline = ShortsPipeline(settings)
            pipeline.preflight()
        if not os.getenv("ANYMODEL_API_KEY"):
            raise ValueError("Не задан ANYMODEL_API_KEY в .env")
        batch = ROOT / "output" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        # Never reuse a previous batch, including simultaneous starts in the same second.
        while True:
            try:
                batch.mkdir(parents=True)
                break
            except FileExistsError:
                await asyncio.sleep(1)
                batch = ROOT / "output" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        progress("START", f"Батч {batch.name}: ищем {args.count} фактов ({args.category})")
        researcher = FactResearcher(history_path=ROOT / "work" / "facts_history.json")
        facts = await researcher.find_facts(args.count, args.category)
        atomic_json(batch / "facts.json", facts)
        progress("RESEARCH", f"Найдено и сохранено фактов: {len(facts)}")
        adapter = StoryAdapter(os.getenv("ANYMODEL_API_KEY", ""))
        stories = []
        for i, fact in enumerate(facts, 1):
            progress(f"SCRIPT {i}/{args.count}", fact["title"])
            story = await adapter.adapt_story(fact["title"], fact["raw_data"])
            if not story:
                raise RuntimeError(f"Не удалось адаптировать факт: {fact['topic']}")
            story = AdaptedStory.model_validate(story).model_dump()
            story.update(topic=fact["topic"], category=fact["category"], sources=fact["sources"],
                         filename=f"video_{i:02d}.mp4", status="dry-run" if args.dry_run else "pending")
            stories.append(story)
            atomic_text(batch / f"story_{i:02d}.txt", story["text"] + "\n")
            atomic_json(batch / "stories.json", stories)
            progress(f"SCRIPT {i}/{args.count}", f"{len(story['text'].split())} слов: {story['text']}")

        def save_plan() -> None:
            atomic_json(batch / "stories.json", stories)
            blocks = [posting(s, s["filename"], i, s["status"]) for i, s in enumerate(stories, 1)]
            atomic_text(batch / "posting_plan.txt", "\n\n".join(blocks) + "\n")

        save_plan()
        failures = 0
        if pipeline is not None:
            with log_to(batch / "render.jsonl"):
                for i, story in enumerate(stories, 1):
                    progress(f"VIDEO {i}/{args.count}", f"Озвучка и рендер {story['filename']}")
                    try:
                        await pipeline.run(story["text"], output_path=batch / story["filename"])
                        story.update(status="rendered", report=pipeline.last_report)
                        progress(f"VIDEO {i}/{args.count}", f"Готово: {story['filename']}; {pipeline.last_report}")
                    except Exception as exc:
                        failures += 1
                        story.update(status="failed", error=safe_error(exc))
                        progress("ERROR", safe_error(exc))
                    finally:
                        save_plan()
        progress("DONE", f"{batch}; сценариев: {len(stories)}, ошибок рендера: {failures}")
        return 1 if failures else 0
    except Exception as exc:
        progress("ERROR", safe_error(exc))
        if batch is not None and batch.exists():
            atomic_json(batch / "error.json", {"error": safe_error(exc)})
        return 1
    finally:
        if researcher is not None:
            await researcher.close()
        if adapter is not None:
            await adapter.client.close()


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
