from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from src.content.adapter import DEFAULT_MODEL, StoryAdapter
from src.runtime import atomic_text, batch_status, log_failure

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FACTS = PROJECT_ROOT / "seed_facts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "input" / "stories.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate documentary stories from facts")
    parser.add_argument(
        "--facts",
        type=Path,
        default=DEFAULT_FACTS,
        help="JSON-файл с фактами (по умолчанию: seed_facts.json)",
    )
    parser.add_argument("--model", default=None, help="LLM model (default: PRIMARY_MODEL / ANYMODEL_MODEL)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("ANYMODEL_API_KEY")
    if not api_key:
        print("[ERROR] Не задан ANYMODEL_API_KEY в .env")
        return 2

    if not args.facts.exists():
        print(f"[ERROR] Файл с фактами не найден: {args.facts}")
        return 2

    try:
        raw_facts = json.loads(args.facts.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log_failure("content", exc)
        return 2
    if not isinstance(raw_facts, list) or not raw_facts:
        print("[ERROR] seed_facts.json должен содержать непустой массив")
        return 2

    adapter = StoryAdapter(api_key, model=args.model)
    stories: list[dict] = []

    print(f"[START] Адаптируем {len(raw_facts)} фактов в документальные сценарии...\n")

    try:
        for i, fact in enumerate(raw_facts, start=1):
            print(f"[{i}/{len(raw_facts)}] Processing item {i}")
            try:
                if not isinstance(fact, dict):
                    raise ValueError("Fact must be a JSON object")
                adapted = await adapter.adapt_story(fact.get("title", f"Fact {i}"), fact.get("raw_data", ""))
                if not adapted:
                    error = getattr(adapter, "last_error", None)
                    raise RuntimeError("Fact adaptation failed") from (error if isinstance(error, Exception) else None)
                adapted["category"] = fact.get("category", "general")
            except Exception as exc:
                log_failure("adaptation", exc, item=i)
                continue
            stories.append(adapted)
            atomic_text(args.output, json.dumps(stories, ensure_ascii=False, indent=2))
        if not stories and not args.output.exists():
            atomic_text(args.output, "[]")
        status = batch_status(len(stories), len(raw_facts))
        print(f"\n[{status}] Saved {len(stories)}/{len(raw_facts)} stories: {args.output}")
        return 0 if status == "SUCCESS" else 1
    finally:
        try:
            await adapter.client.close()
        except Exception as exc:
            log_failure("client.close", exc)


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
