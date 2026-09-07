from __future__ import annotations

import json
import os
import re
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

# Модель по умолчанию из .env или Claude Sonnet 5
DEFAULT_MODEL = os.getenv("ANYMODEL_MODEL", "cc/claude-sonnet-5")

SYSTEM_PROMPT = """Ты — сценарист вирусных коротких видео (Shorts/Reels) в стиле историй с Пикабу и «Подслушано».
Напиши монолог от первого лица о реальной бытовой ситуации в СНГ (семья, жилье, деньги, соседи).

Правила структуры:
1. Хук (0–3 сек): резкое признание без приветствий («В день свадьбы свекровь шепнула мне на ухо то, от чего земля ушла из-под ног...»).
2. Конфликт (30–40 сек): реалии жизни (ипотека, дубликат ключей, дача, чат дома, МФЦ). Разговорный живой язык, эмоции.
3. Развязка/Обрыв (последние 5 сек): этический тупик или открытый финал («Как бы вы поступили на моем месте?»).

Требования:
- Объем строго от 65 до 85 слов в поле text.
- Выводи только валидный JSON:
{
  "title": "Цепляющий заголовок до 6 слов",
  "text": "Текст диктора без ремарок",
  "tags": ["#историиизжизни", "#жиза", "#семья", "#отношения", "#шортс"]
}
"""


class AdaptedStory(BaseModel):
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value.split()) > 10:
            raise ValueError(f"Title too long ({len(value.split())} words)")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        words = len(value.split())
        if not 60 <= words <= 95:
            raise ValueError(f"Story length is {words} words, required 60-95")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def sanitize_tags(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [t.strip() for t in value.split(",")]
        if not isinstance(value, list):
            value = ["#историиизжизни", "#жиза", "#шортс"]

        clean_tags: list[str] = []
        for tag in value:
            t = str(tag).strip()
            if not t:
                continue
            if not t.startswith("#"):
                t = f"#{t}"
            clean_tags.append(t)

        # Берем максимум 5 тегов, дополняем если меньше 3
        result = clean_tags[:5]
        while len(result) < 3:
            result.append("#шортс")
        return result


class StoryAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://anymodel.org/v1")
        self.model = model

    async def adapt_story(self, original_title: str, original_text: str) -> dict | None:
        prompt = f"Исходный заголовок: {original_title}\n\nИсходная история:\n{original_text}"
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.85,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            data = self._parse_json(content)
            return AdaptedStory.model_validate(data).model_dump()
        except Exception as exc:
            print(f"[AI ERROR] {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned.strip())
        if not isinstance(parsed, dict):
            raise ValueError("Model response is not an object")
        return parsed

    async def generate_story_from_scratch(
        self, topic: str = "скандал из-за наследства или жилья"
    ) -> dict | None:
        """Синтезирует виральную историю с нуля под Рунет."""
        prompt = (
            f"Напиши жизненную, скандальную историю от первого лица на тему: '{topic}'. "
            "Используй реалии жизни в России/СНГ. Текст должен звучать искренне и вызывать бурную реакцию в комментариях."
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.85,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            data = self._parse_json(content)
            return AdaptedStory.model_validate(data).model_dump()
        except Exception as exc:
            print(f"[AI ERROR] {type(exc).__name__}: {exc}")
            return None