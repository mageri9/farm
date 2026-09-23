from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import llm_models
from ..runtime import atomic_json, file_lock
from .llm import complete_with_fallback

CATEGORIES = ("systems", "science", "mind")

# Хронометраж 35-40 секунд озвучки при темпе ~2.1 слова/сек.
MIN_WORDS = 75
MAX_WORDS = 85

SYSTEM_PROMPT = """Ты — сценарист брутального научпопа об историческом оружии и доспехах. Твой стиль — холодная ирония и разговорная подача в духе нуарного детектива: ты объясняешь механику убийства так же спокойно, как инженер объясняет работу редуктора.

Твой голос — бархатный, спокойный, знающий финал наперед. Ты сидишь на ночной кухне с умным собеседником и без театральных криков разбираешь, почему железо работает именно так. Никакого тиктокерского визга.

Возьми заданный предмет (оружие или доспех) и напиши цельный, связный монолог для озвучки из ровно четырех beats. Это не хокку и не рубленые лозунги: биты должны читаться как единый абзац с живыми переходами, предложения полные и разговорные.

Драматургия по битам строго такая:
- establishing: миф, расхожее заблуждение об этом предмете. Начни утверждением, разрушающим иллюзию; риторические вопросы со словом «Как» запрещены.
- tension: реальная механика и геометрия — форма, масса, баланс, распределение силы, почему предмет устроен так.
- subject: физика разрушения. Что конкретно происходит с бронёй и костью: продавливание, скол, рычаг, инерция, передача ударной волны. Телесные образы точные, анатомические, без садистского смакования.
- aftermath: резкий payoff — вывод о том, чем на самом деле была эта вещь и почему миф оказался глупее реальности. Без морализаторства и призывов подписаться.

ЖЕСТКИЕ ПРАВИЛА: суммарный текст четырех beats строго от 75 до 85 слов; каждый бит непустой и содержит законченные предложения; все числа, даты и величины только словами; не используй омографы со спорным ударением; не используй фразы «сломал догму», «зеркало реальности», «фундаментально», «парадокс заключается в том»; только реальные, задокументированные сведения с существующей ссылкой.

Верни только JSON-массив объектов с полями topic, category, source_url, title, beats и tags. beats должен содержать establishing, tension, subject и aftermath. source_url — прямая HTTPS-ссылка на Википедию или отчет.
Формат объекта:
{"topic": "English Name", "category": "systems", "source_url": "https://...", "title": "Хлесткий заголовок", "beats": {"establishing": "...", "tension": "...", "subject": "...", "aftermath": "..."}, "tags": ["#шортс", "#факты"]}
"""


class StoryBeats(BaseModel):
    establishing: str
    tension: str
    subject: str
    aftermath: str

class GeneratedStory(BaseModel):
    topic: str = Field(min_length=3)
    category: Literal["systems", "science", "mind"]
    source_url: str = Field(min_length=8)
    title: str = Field(min_length=3, max_length=50)
    beats: StoryBeats
    text: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("source_url")
    @classmethod
    def valid_source_url(cls, value: str) -> str:
        value = value.strip()
        if not re.match(r"^https://[^\s]+$", value):
            raise ValueError("source_url must be a direct HTTPS URL")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        words = len(value.split())
        if not MIN_WORDS <= words <= MAX_WORDS:
            raise ValueError(
                f"Количество слов должно быть строго от {MIN_WORDS} до {MAX_WORDS}. Сейчас: {words}"
            )
        if re.search(r"\d", value):
            raise ValueError("Все числа, даты и величины должны быть написаны строго словами")
        return value

    @model_validator(mode="before")
    @classmethod
    def compose_text(cls, data: Any) -> Any:
        if isinstance(data, dict) and "beats" in data:
            beats = StoryBeats.model_validate(data["beats"])
            data = {**data, "text": f"{beats.establishing} {beats.tension} {beats.subject} {beats.aftermath}"}
        return data


class UnifiedStoryGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 history_path: str | Path | None = None) -> None:
        key = api_key or os.getenv("ANYMODEL_API_KEY")
        if not key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("ANYMODEL_BASE_URL", "https://api.aitunnel.ru/v1/"), timeout=90.0, max_retries=0)
        self.model, self.fallback_model = llm_models()
        if model:
            self.model = model
        self.history_path = Path(history_path) if history_path else Path(__file__).resolve().parents[2] / "work/facts_history.json"

    def _history(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read fact history: {self.history_path}") from exc
        if not isinstance(data, list):
            raise ValueError("Fact history must be a JSON list")
        return [item if isinstance(item, dict) else {"topic": str(item), "keywords": []} for item in data]

    @staticmethod
    def _parse(content: str) -> list[dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError("Ответ должен быть JSON-массивом объектов")
        return data

    async def generate(self, count: int, category: str = "all",
                       topic: str | None = None) -> list[dict[str, Any]]:
        if count < 1 or (category != "all" and category not in CATEGORIES):
            raise ValueError("Некорректные count или category")
        categories = [CATEGORIES[i % 3] if category == "all" else category for i in range(count)]
        with file_lock(self.history_path.with_suffix(".json.lock")):
            history = self._history()
        request_text = (
            f"Создай ровно {count} историй. Категории по порядку: {json.dumps(categories)}. "
            f"Не повторяй темы из истории: {json.dumps(history, ensure_ascii=False)}"
        )
        if topic and topic.strip():
            # Заданный предмет важнее истории тем: разбираем именно его.
            request_text = (
                f"Предмет для разбора: «{topic.strip()}». Разбери строго его, не подменяй другим.\n"
                + request_text
            )
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request_text}]
        last_error: Exception | None = None
        stories: list[GeneratedStory] | None = None
        for attempt in range(3):
            async def request(model: str):
                return await self.client.chat.completions.create(model=model, temperature=0.8,
                    messages=messages)
            completion = await complete_with_fallback(request, self.model, self.fallback_model)
            content = completion.choices[0].message.content or ""
            try:
                data = self._parse(content)
                if len(data) != count:
                    raise ValueError(f"Ожидалось {count} историй, получено {len(data)}")
                stories = [GeneratedStory.model_validate(item) for item in data]
                for story, requested in zip(stories, categories):
                    if story.category != requested:
                        raise ValueError(f"Требуется категория: {requested}")
                break
            except ValueError as exc:
                stories = None
                last_error = exc
                messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Исправь JSON с учетом ошибки: {exc}. Верни только корректный JSON."}])
        if stories is None:
            raise ValueError(f"Не удалось проверить ответ генератора: {last_error}") from last_error
        with file_lock(self.history_path.with_suffix(".json.lock")):
            history = self._history()
            seen = [{str(old.get("topic", "")).casefold(), *(str(k).casefold() for k in old.get("keywords", []))} for old in history]
            for story in stories:
                keys = {story.topic.casefold(), story.title.casefold()}
                # Явно заданный предмет разрешено разбирать повторно.
                if not topic and any(keys & old_keys for old_keys in seen):
                    raise ValueError(f"Already used: {story.topic}")
                history.append({"topic": story.topic, "category": story.category, "keywords": [story.topic, story.title]})
                seen.append(keys)
            atomic_json(self.history_path, history)
        return [story.model_dump() for story in stories]

    async def close(self) -> None:
        await self.client.close()
