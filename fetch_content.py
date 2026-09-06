from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.content.adapter import DEFAULT_MODEL, StoryAdapter
from src.content.reddit import fetch_reddit_stories

AI_TOPICS = (
    "измена и предательство лучшего друга перед свадьбой",
    "наследство, из-за которого разругалась вся семья",
    "тайна мужа, раскрытая через скрытый аккаунт в соцсетях",
    "конфликт с токсичными родственниками на празднике",
    "случайно раскрытый обман на работе, подставивший коллегу",
    "сосед, который годами пользовался чужой парковкой",
    "подруга, скопировавшая свадьбу до мелочей",
    "родители, тайно набравшие кредитов на имя ребёнка",
    "неожиданный гость, сорвавший семейную помолвку",
    "ложь о дипломе, вскрывшаяся на собеседовании",
    "бывший партнёр, потребовавший вернуть дорогой подарок",
    "ребёнок, узнавший на семейном ужине правду о своём отце",
    "арендодатель, тайно входивший в квартиру жильца",
    "подмена подарка на юбилее, раскрывшая давнюю обиду",
    "невеста, запретившая сестре жениха приходить на свадьбу",
    "анонимная жалоба, из-за которой уволили невиновного сотрудника",
    "семейный рецепт, который бабушка завещала только одному из внуков",
    "друг, который годами выдавал чужие успехи за свои",
    "домашняя камера, снявшая неожиданную правду о няне",
    "ошибочно отправленное сообщение, разрушившее многолетнюю дружбу",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or generate stories for Shorts")
    parser.add_argument("--source", choices=["reddit", "ai"], default="ai", help="Источник: reddit или ai")
    parser.add_argument("--subreddits", default="AITAH,relationship_advice,tifu")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("input/stories.json"))
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("ANYMODEL_API_KEY")
    if not api_key:
        print("[ERROR] Не задан ANYMODEL_API_KEY в .env")
        return 2

    if args.count < 1:
        print("[ERROR] --count должен быть положительным")
        return 2

    adapter = StoryAdapter(api_key, model=args.model)
    stories: list[dict] = []

    if args.source == "reddit":
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
        batches = await asyncio.gather(
            *(fetch_reddit_stories(subreddit=s, limit=max(args.count * 3, 15)) for s in subreddits),
            return_exceptions=True,
        )
        posts: list[dict] = []
        for subreddit, batch in zip(subreddits, batches):
            if isinstance(batch, Exception):
                print(f"[REDDIT] Ошибка r/{subreddit}: {batch}")
                continue
            posts.extend(batch)

        unique: dict[str, dict] = {post["id"]: post for post in posts}
        candidates = list(unique.values())[: args.count * 3]
        print(f"[REDDIT] Найдено постов: {len(candidates)}")

        for post in candidates:
            if len(stories) >= args.count:
                break
            print(f'[AI {len(stories) + 1}/{args.count}] Адаптируем: "{post["title"]}"')
            adapted = await adapter.adapt_story(post["title"], post["selftext"])
            if adapted:
                stories.append(adapted)

    # Если выбран режим AI или Reddit ничего не вернул — генерируем напрямую
    if args.source == "ai" or (args.source == "reddit" and not stories):
        if args.source == "reddit":
            print("[WARN] Reddit заблокировал запросы, переключаемся на генерацию через AnyModel...")

        while len(stories) < args.count:
            idx = len(stories)
            topic = AI_TOPICS[idx % len(AI_TOPICS)]
            print(f"[AI {idx + 1}/{args.count}] Генерируем сюжет: '{topic}'")
            story = await adapter.generate_story_from_scratch(topic)
            if story:
                stories.append(story)
            else:
                print("[WARN] Повторная попытка генерации...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Успешно сохранено {len(stories)} историй в {args.output}")
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
