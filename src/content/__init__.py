"""Content acquisition and adaptation for Shorts stories."""

from .adapter import StoryAdapter
from .reddit import fetch_reddit_stories

__all__ = ["StoryAdapter", "fetch_reddit_stories"]
