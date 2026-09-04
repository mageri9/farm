from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.content.adapter import StoryAdapter
from src.content.reddit import fetch_reddit_stories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Reddit stories and adapt them for Shorts")
    parser.add_argument("--subreddits", default="AITAH,relationship_advice,tifu")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--output", type=Path, default=Path("input/stories.json"))
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("ANYMODEL_API_KEY")
    if not api_key:
        print("[ERROR] Не задан ANYMODEL_API_KEY")
        return 2
    if args.count < 1:
        print("[ERROR] --count должен быть положительным")
        return 2

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

    adapter = StoryAdapter(api_key, model=args.model)
    stories: list[dict] = []
    for post in candidates:
        if len(stories) >= args.count:
            break
        print(f'[AI {len(stories) + 1}/{args.count}] Адаптируем историю "{post["title"]}"')
        adapted = await adapter.adapt_story(post["title"], post["selftext"])
        if adapted:
            stories.append(adapted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Сохранено в {args.output}")
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
