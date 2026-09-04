from __future__ import annotations

import httpx

REDDIT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


async def fetch_reddit_stories(
    subreddit: str = "AITAH", time_filter: str = "week", limit: int = 15
) -> list[dict]:
    """Fetch and filter self posts from Reddit's public JSON endpoint."""
    if not subreddit or limit < 1:
        return []
    url = f"https://old.reddit.com/r/{subreddit}/top.json?t={time_filter}&limit={limit}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Reddit request failed for r/{subreddit}: {exc}") from exc

    children = payload.get("data", {}).get("children", [])
    stories: list[dict] = []
    for child in children:
        post = child.get("data", {})
        text = post.get("selftext") or ""
        if not post.get("is_self", False) or post.get("over_18", False):
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
                "url": str(post.get("permalink", "")),
            }
        )
    return [story for story in stories if story["id"] and story["title"]]
