from __future__ import annotations

import httpx


async def fetch_reddit_stories(
    subreddit: str = "AITAH", time_filter: str = "week", limit: int = 15
) -> list[dict]:
    """Fetch top posts from PullPush API (public Reddit mirror without auth blocks)."""
    if not subreddit or limit < 1:
        return []

    # PullPush API отдает посты из Reddit без Cloudflare и без логина
    url = f"https://api.pullpush.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=score&size={limit}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Reddit request failed for r/{subreddit}: {exc}") from exc

    posts_data = payload.get("data", [])
    stories: list[dict] = []

    for post in posts_data:
        text = post.get("selftext") or ""
        # Проверяем, что это текстовый пост, не NSFW
        if post.get("over_18", False):
            continue
        if not 350 <= len(text) <= 3500:
            continue
        if "[removed]" in text.lower() or "[deleted]" in text.lower():
            continue

        stories.append(
            {
                "id": str(post.get("id", "")),
                "title": str(post.get("title", "")).strip(),
                "selftext": text.strip(),
                "url": f"https://reddit.com{post.get('permalink', '')}",
            }
        )

    return [story for story in stories if story["id"] and story["title"]]