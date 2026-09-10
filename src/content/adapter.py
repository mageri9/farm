from __future__ import annotations

import json
import os
import re
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

# Модель по умолчанию из .env или Claude Sonnet 5
DEFAULT_MODEL = os.getenv("ANYMODEL_MODEL", "cc/claude-sonnet-5")

SYSTEM_PROMPT = """Ты — ведущий сценарист документальных мини-фильмов в концепции «Реальность страннее выдумки» (научные парадоксы, инженерные катастрофы, сбои систем, когнитивные иллюзии).

Твоя задача — взять реальный проверенный факт и превратить его в напряженную 40-секундную историю для взрослого диктора-документалиста.

СТРОГАЯ СТРУКТУРА (всего 65–80 слов!):
1. ХУК (0-3 сек / 1 предложение): Парадокс или ломка интуиции. Никаких «Знали ли вы», сразу факт, ломающий шаблон.
2. КОНТЕКСТ (3-10 сек / 1-2 предложения): Кто, где, масштаб системы или замысла.
3. МЕХАНИЗМ (10-25 сек / 2-3 предложения): В чем именно крылась неочевидная ошибка или закон природы. Суть сбоя/явления простыми словами.
4. НАРАСТАНИЕ (25-35 сек / 1-2 предложения): Как ошибка накапливалась незаметно, пока не стало поздно.
5. ФИНАЛ / ПАНЧ (35-40 сек / 1 предложение): Смысловой итог. Реальность беспощадна к небрежности, либо законы физики не обязаны быть интуитивными.

ПРАВИЛА:
- Тон холодный, строгий, интеллектуальный (как в документалках Netflix/BBC).
- Пиши только от третьего лица (нейтральный наблюдатель). Никакого первого лица!
- Никаких призывов перейти в Telegram или подписаться. Ролик должен быть законченным шедевром.
- ВСЕ ЧИСЛА ПИШИ СЛОВАМИ для диктора (например: «триста миллионов долларов», «в девяносто девятом году»).
- Общий объем поля text: СТРОГО от 65 до 80 слов.

ФОРМАТ ВЫВОДА (только валидный JSON):
{
  "title": "Цепляющий заголовок до 6 слов",
  "text": "Текст диктора без ремарок (65-80 слов)",
  "tags": ["#наука", "#история", "#технологии", "#факты", "#шортс"]
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
        # Исправлено: честный надежный диапазон 60-85 слов
        if not 60 <= words <= 85:
            raise ValueError(f"Story length is {words} words, required 60-85")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def sanitize_tags(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [t.strip() for t in value.split(",")]
        if not isinstance(value, list):
            value = ["#наука", "#технологии", "#шортс"]

        clean_tags: list[str] = []
        for tag in value:
            t = str(tag).strip()
            if not t:
                continue
            if not t.startswith("#"):
                t = f"#{t}"
            clean_tags.append(t)

        result = clean_tags[:5]
        while len(result) < 3:
            result.append("#факты")
        return result


class StoryAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://anymodel.org/v1")
        self.model = model

    async def adapt_story(self, original_title: str, original_text: str) -> dict | None:
        prompt = f"Тема/Фактура: {original_title}\n\nСырые данные о событии/парадоксе:\n{original_text}"
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
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
        self, topic: str = "Катастрофа Mars Climate Orbiter из-за единиц измерения"
    ) -> dict | None:
        """Генерирует документальный сценарий по реальному факту."""
        prompt = (
            f"Напиши документальный сценарий по 5-ступенчатой структуре на тему: '{topic}'. "
            "Опирайся только на строгие факты. Покажи драму сложной системы или контринтуитивность законов Вселенной."
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
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