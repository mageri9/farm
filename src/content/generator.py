from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from ..config import llm_models
from ..runtime import atomic_json, file_lock
from .llm import complete_with_fallback

CATEGORIES = ("systems", "science", "mind")

SYSTEM_PROMPT = """Ты — сценарист и фактологический исследователь коротких видео. Твой стиль — интеллектуальный фатализм, холодная ирония и разговорная подача в духе историка Бушвакера (Bushwacker) или нуарного детектива.

Твой голос — бархатный, спокойный, знающий финал наперед. Ты сидишь на ночной кухне с умным собеседником и с легкой усмешкой рассказываешь, как сложнейшие системы мира спотыкаются о человеческую спесь и банальную физику. Никаких криков и тиктокерского визга.

Выбери один реальный, строго задокументированный инцидент или эксперимент по заданной категории и сразу напиши готовый монолог для озвучки. Начни утверждением, разрушающим иллюзию контроля; запрещены риторические вопросы со словом «Как». Передай конфликт через понятные телесные образы, затем дай резкий payoff: грандиозная система погибает из-за смехотворно мелкой причины. Без морализаторства и призывов подписаться.

ЖЕСТКИЕ ПРАВИЛА: text строго от 42 до 48 слов; все числа, даты и величины только словами; не используй омографы со спорным ударением; не используй фразы «сломал догму», «зеркало реальности», «фундаментально», «парадокс заключается в том»; только реальные события с существующей ссылкой.

Верни только JSON-массив объектов с полями topic (каноническое английское имя), category, source_url, title, text, tags и keywords. source_url — прямая HTTPS-ссылка на Википедию или отчет.
"""


class GeneratedStory(BaseModel):
    topic: str = Field(min_length=3)
    category: Literal["systems", "science", "mind"]
    source_url: str = Field(min_length=8)
    title: str = Field(min_length=3, max_length=50)
    text: str
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
        if not 42 <= words <= 48:
            raise ValueError(f"Количество слов должно быть строго от 42 до 48. Сейчас: {words}")
        if re.search(r"\d", value):
            raise ValueError("Все числа, даты и величины должны быть написаны строго словами")
        return value


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

    async def generate(self, count: int, category: str = "all") -> list[dict[str, Any]]:
        if count < 1 or (category != "all" and category not in CATEGORIES):
            raise ValueError("Некорректные count или category")
        categories = [CATEGORIES[i % 3] if category == "all" else category for i in range(count)]
        with file_lock(self.history_path.with_suffix(".json.lock")):
            history = self._history()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": (
            f"Создай ровно {count} историй. Категории по порядку: {json.dumps(categories)}. "
            f"Не повторяй темы из истории: {json.dumps(history, ensure_ascii=False)}") }]
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
                if any(keys & old_keys for old_keys in seen):
                    raise ValueError(f"Already used: {story.topic}")
                history.append({"topic": story.topic, "category": story.category, "keywords": [story.topic, story.title]})
                seen.append(keys)
            atomic_json(self.history_path, history)
        return [story.model_dump() for story in stories]

    async def close(self) -> None:
        await self.client.close()
