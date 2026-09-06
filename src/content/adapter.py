from __future__ import annotations

import json
import re
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_MODEL = "am/free"

SYSTEM_PROMPT = """Ты — опытный сценарист виральных Shorts/Reels на русском языке в нише бытовых драм и Reddit-историй.
Адаптируй исходную историю в монолог для озвучки:
1. Хук (0–3 сек): шокирующее признание от первого лица. Запрещены клише «Привет, реддит», «Сегодня я расскажу», «Пользователь поделился».
2. Тело (30–40 сек): динамичное раскрытие конфликта, никакой воды.
3. Клиффхэнгер / открытый финал (последние 5 сек): интригующий вопрос к зрителям, провоцирующий комментарий («Как бы вы поступили?», «Я прав в этой ситуации?»).
СТРОГО 70–85 слов в поле text. Выводи только валидный JSON с ключами:
title — цепляющий заголовок на русском, до 6 слов;
text — только текст диктора, без ремарок в скобках;
tags — список из 4–5 хэштегов.
"""


class AdaptedStory(BaseModel):
    title: str
    text: str
    tags: list[str] = Field(min_length=4, max_length=5)

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
        if not 70 <= words <= 85:
            raise ValueError(f"Story length is {words} words, required 70-85")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        tags = [tag.strip() for tag in value]
        if any(not tag.startswith("#") or len(tag) == 1 for tag in tags):
            raise ValueError("Every tag must be a non-empty hashtag")
        return tags


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
        self, topic: str = "бытовая драма, измена или конфликт в семье"
    ) -> dict | None:
        """Синтезирует виральную историю с нуля в стиле Reddit r/AITAH без парсинга."""
        prompt = (
            f"Придумай реалистичную, вирусную историю из жизни в стиле Reddit r/AITAH или r/tifu на тему: '{topic}'. "
            "История должна звучать максимально искренне и правдиво от первого лица, вызывая бурю споров в комментариях."
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.9,
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
