from __future__ import annotations
import asyncio
import html
import re
from xml.etree import ElementTree as ET
import httpx

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

def _apply_common_filters(raw_posts: list[dict]) -> list[dict]:
    stories: list[dict] = []
    for post in raw_posts:
        text = post.get("selftext") or ""
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
                "url": post.get("url", ""),
            }
        )

    return [story for story in stories if story["id"] and story["title"]]

async def _fetch_pullpush_raw(subreddit: str, limit: int) -> list[dict]:
    """Fetch from PullPush with retries that respect 429/Retry-After."""
    url = f"https://api.pullpush.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=score&size={limit}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        last_status = None
        for attempt in range(3):
            response = await client.get(url, headers=_HEADERS)
            last_status = response.status_code
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 5 * (attempt + 1)))
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            posts = payload.get("data", [])
            return [
                {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "selftext": p.get("selftext"),
                    "over_18": p.get("over_18", False),
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                }
                for p in posts
            ]
    raise RuntimeError(f"PullPush rate limited after retries (last status {last_status})")

def _parse_reddit_rss(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    posts: list[dict] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_el = entry.find("atom:title", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        link_el = entry.find("atom:link", _ATOM_NS)
        id_el = entry.find("atom:id", _ATOM_NS)
        title = (title_el.text or "").strip() if title_el is not None else ""
        raw_html = (content_el.text or "") if content_el is not None else ""
        # Strip HTML tags from the RSS content field; this is a rough text
        # extraction, not full HTML parsing, but is enough for our purposes.
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))
        text = re.sub(r"\s+", " ", text).strip()
        permalink = link_el.get("href", "") if link_el is not None else ""
        post_id = (id_el.text or "").rsplit("_", 1)[-1] if id_el is not None else ""
        posts.append({"id": post_id, "title": title, "selftext": text, "over_18": False, "url": permalink})
    return posts

async def _fetch_rss_raw(subreddit: str, limit: int) -> list[dict]:
    url = f"https://old.reddit.com/r/{subreddit}/top/.rss?t=week&limit={min(limit, 100)}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=True) as client:
        response = await client.get(url, headers=_HEADERS)
        response.raise_for_status()
        return _parse_reddit_rss(response.text)

async def fetch_reddit_stories(
    subreddit: str = "AITAH", time_filter: str = "week", limit: int = 15
) -> list[dict]:
    """Fetch top text posts from a subreddit. Tries PullPush first (with
    retries), falls back to Reddit's public RSS feed if PullPush is rate
    limited or unavailable."""
    if not subreddit or limit < 1:
        return []
    errors = []
    try:
        raw_posts = await _fetch_pullpush_raw(subreddit, limit)
        return _apply_common_filters(raw_posts)
    except (httpx.HTTPError, RuntimeError) as exc:
        errors.append(f"PullPush: {exc}")
    try:
        raw_posts = await _fetch_rss_raw(subreddit, limit)
        return _apply_common_filters(raw_posts)
    except (httpx.HTTPError, ET.ParseError) as exc:
        errors.append(f"RSS: {exc}")
    raise RuntimeError(f"Reddit request failed for r/{subreddit}: " + "; ".join(errors))