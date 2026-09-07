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
    "Свекровь втайне сделала дубликат ключей от нашей квартиры и пришла, пока нас не было",
    "Муж тайком взял кредит на 500 тысяч, чтобы помочь своей сестре сыграть свадьбу",
    "Бабушка перед смертью переписала квартиру на сиделку, оставив внуков на улице",
    "Сосед на парковке проколол колеса за то, что я встал на 'его законное' место во дворе",
    "Узнала, что будущий муж за месяц до свадьбы зарегистрировался в Тиндере",
    "Родители мужа требуют, чтобы мы продали мою добрачную однушку ради расширения",
    "Начальник заставил выйти в законный выходной, а премию выписал своей любовнице",
    "Сестра попросила посидеть с племянником на выходных и пропала на две недели",
    "Случайно нашла в телефоне мужа скрытую переписку с моей родной сестрой",
    "Купили дачу, а сосед через суд пытается отобрать половину участка из-за забора",
    "Тайная заначка мужа в гараже, о которой я узнала случайно от слесаря",
    "Подруга заняла 100 тысяч на операцию маме, а сама улетела отдыхать в Дубай",
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
        posts: list[dict] = []
        for i, subreddit in enumerate(subreddits):
            try:
                batch = await fetch_reddit_stories(subreddit=subreddit, limit=max(args.count * 3, 15))
                posts.extend(batch)
            except Exception as exc:
                print(f"[REDDIT] Ошибка r/{subreddit}: {exc}")
            if i < len(subreddits) - 1:
                await asyncio.sleep(2)  # пауза между сабреддитами, чтобы не словить burst-лимит

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
