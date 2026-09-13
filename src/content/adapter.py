from __future__ import annotations

import json
import re
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from ..runtime import logger, safe_error
from ..config import llm_models
from .llm import complete_with_fallback

DEFAULT_MODEL = llm_models()[0]

SYSTEM_PROMPT = """Ты пишешь короткий сценарий для вертикального видео на естественном русском. Это разговор умного человека с другом, без лекционного или документального тона. Структура: хук → конкретное событие → неожиданный механизм → короткий payoff.

СТРОГИЕ ПРАВИЛА:
1. ОБЪЕМ: строго от 42 до 50 слов в поле text, без учета заголовка и тегов. Целевая длительность TTS: 21–25 секунд; аудио длиннее 26,5 секунды отклоняется.
2. ХУК ПЕРВЫХ ДВУХ СЕКУНД: первая фраза сразу лично цепляет зрителя или ломает базовую логику. Примеры интонации: «Как вырубить полстраны одной строчкой кода?», «Ваш мозг можно обмануть обычной перчаткой». В хуке запрещены даты и годы. Не начинай с объяснений.
3. ЗАПРЕТ НА КАНЦЕЛЯРИТ: никогда не начинай ролик с даты или года («В тысяча девятьсот девяносто восьмом году», «Четырнадцатого августа»). Даты допустимы только в середине и только если нужны. Не оставляй заумные термины без образного перевода: вместо «конфликта потоков» скажи «программа споткнулась сама о себя и замолчала», вместо «тактильной синхронизации» — «мозг поверил глазам и отказался от собственной руки». Каждый технический факт показывай через действие, предмет или телесное ощущение.
4. NATURAL SPOKEN RHYTHM: обычно четыре-десять слов в предложении; допускаются естественные полные предложения. Не дроби грамматически связную мысль на отдельные слова ради псевдодраматизма (нельзя: «Влажность. И микротрещины. Ослабили металл.»; нужно: «Морская влажность и микротрещины постепенно ослабили фюзеляж.»). Не расставляй тире или многоточия для искусственных пауз: пунктуация служит только естественному дыханию речи. Текст должен звучать естественно при чтении вслух без субтитров, как будто умный человек делится поразительным фактом со знакомым.
5. ЧИСЛА: все числительные, даты, годы, версии и величины пиши только словами.
6. Пиши как доверительный разговор. Не используй тире или многоточия ради драматизма.
7. СТРУКТУРА: хук-вызов; конкретная деталь события; поворот или открытие; короткий payoff. ФАКТЫ: опирайся только на переданную фактуру. Не выдумывай причины, последствия или детали ради драматизма. Обращайся к зрителю на «вы», если это усиливает хук. Никаких призывов подписаться.
8. ФОНЕТИЧЕСКАЯ ОДНОЗНАЧНОСТЬ ДЛЯ TTS: не используй русские омографы и формы с двусмысленным ударением или значением, зависящим от падежа/контекста (например, «шара», «стоит», «замок», «орган», «пропасть»). Всегда заменяй их однозначными научными или литературными синонимами: «четыре сферы» вместо «четыре шара», «обходится в миллионы» вместо «стоит миллионы», «рецептор» или «часть тела» вместо «орган восприятия». Проверь весь текст перед ответом.

ФОРМАТ ВЫВОДА (ТОЛЬКО ЧИСТЫЙ JSON):
{
  "title": "Емкий заголовок до 5 слов",
  "text": "Текст диктора строго от 42 до 50 слов",
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
        if not 42 <= words <= 50:
            raise ValueError(f"Story length is {words} words, required 42-50 words for short-form TTS")
        if re.search(r"\d", value):
            raise ValueError("Write all numbers in words, including dates and system names")
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
    def __init__(self, api_key: str, model: str | None = None) -> None:
        if not api_key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://anymodel.org/v1", timeout=90.0, max_retries=0)
        primary, self.fallback_model = llm_models()
        self.model = model or primary
        self.last_error: Exception | None = None

    async def adapt_story(self, original_title: str, original_text: str) -> dict | None:
        self.last_error = None
        prompt = (
            f"Тема: {original_title}\n\n"
            f"Фактура: {original_text}\n\n"
            "Напиши разговорный сценарий на 42-50 слов, максимум семь предложений. "
            "Структура: хук, событие, механизм, короткий payoff. Хук без даты. "
            "Пиши простыми устными фразами, без пафоса и лишних терминов; числа только словами. "
            "Не добавляй фактов и не используй тире или многоточия ради драматизма."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            content = ""
            try:
                async def request(model: str):
                    return await self.client.chat.completions.create(
                        model=model, temperature=0.6,
                        response_format={"type": "json_object"}, messages=messages,
                    )
                completion = await complete_with_fallback(request, self.model, self.fallback_model)
            except Exception as exc:
                self.last_error = exc
                logger.error("Adaptation request failed: %s", safe_error(exc))
                return None
            try:
                content = completion.choices[0].message.content or ""
                story = AdaptedStory.model_validate(self._parse_json(content)).model_dump()
                self.last_error = None
                return story
            except ValueError as exc:
                self.last_error = exc
                print(f"[SCRIPT {attempt + 1}/3] Ответ не прошел проверку; исправляем длину/формат.", flush=True)
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Исправь JSON: {exc}. Строго 42-50 слов, без цифр. Не добавляй фактов."},
                ])
            except Exception as exc:
                self.last_error = exc
                logger.error("Adaptation response failed: %s", safe_error(exc))
                return None
        logger.error("Adaptation validation failed: %s", safe_error(self.last_error))
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
