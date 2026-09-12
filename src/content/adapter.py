from __future__ import annotations

import json
import os
import re
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

DEFAULT_MODEL = os.getenv("ANYMODEL_MODEL", "ag/gemini-3.7-flash-medium")

SYSTEM_PROMPT = """Ты — сценарист ультра-динамичных документальных расследований (Shorts/Reels) в формате «Реальность страннее выдумки».
Твоя задача — превратить сухой факт в напряженную историю ровно на 25 секунд для строгого голоса диктора.

СТРОГИЕ ПРАВИЛА:
1. ОБЪЕМ: строго от 45 до 55 слов! Каждое слово на счету. Никакой воды.
2. СТИЛЬ: короткие, рубленые фразы. Телеграфный ритм. Никаких сложных причастных оборотов и придаточных предложений.
3. ЧИСЛА: все числительные обязательно пиши словами («триста двадцать семь», «в девяносто девятом году»), иначе синтезатор речи споткнется.
4. ПОВЕСТВОВАНИЕ: строго от третьего лица (нейтральный хроникер). Никаких «мы», «я», «подпишитесь».
5. ДРАМАТУРГИЯ (4 ФАЗЫ):
   - Фаза 1 (1 предложение): Парадокс или шокирующий факт в лоб.
   - Фаза 2 (1-2 предложения): Исходные данные и масштаб системы.
   - Фаза 3 (1-2 предложения): Нелепая деталь сбоя или закона физики.
   - Фаза 4 (1 предложение): Холодный смысловой итог.

ФОРМАТ ВЫВОДА (ТОЛЬКО ЧИСТЫЙ JSON):
{
  "title": "Емкий заголовок до 5 слов",
  "text": "Текст диктора строго от 45 до 55 слов",
  "tags": ["#наука", "#технологии", "#факты", "#история", "#шортс"]
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
        # Коридор для динамичного ролика на 22-28 секунд
        if not 42 <= words <= 60:
            raise ValueError(f"Story length is {words} words, required 42-60 words for 25s format")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def sanitize_tags(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [t.strip() for t in value.split(",")]
        if not isinstance(value, list):
            value = ["#наука", "#факты", "#шортс"]

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
            result.append("#шортс")
        return result


class StoryAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://anymodel.org/v1")
        self.model = model

    async def adapt_story(self, original_title: str, original_text: str) -> dict | None:
        prompt = (
            f"Тема: {original_title}\n\n"
            f"Фактура: {original_text}\n\n"
            "Напиши ультра-динамичный сценарий строго на 45-55 слов короткими фразами."
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.6,
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