from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from ..runtime import atomic_json, file_lock

DEFAULT_MODEL = os.getenv("ANYMODEL_MODEL", "ag/gemini-3.7-flash-medium")
CATEGORIES = ("systems", "science", "mind")


def normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


class Source(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    title: str = Field(min_length=3)
    url: HttpUrl


class ResearchedFact(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    topic: str = Field(min_length=3)
    category: Literal["systems", "science", "mind"]
    title: str = Field(min_length=3)
    raw_data: str = Field(min_length=30)
    date: str = Field(min_length=4)
    location: str = Field(min_length=2)
    systems: list[str] = Field(min_length=1)
    figures: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1, max_length=3)

    @field_validator("systems", "figures", "keywords")
    @classmethod
    def nonempty_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("List entries must not be empty")
        return [item.strip() for item in value]

    @field_validator("date")
    @classmethod
    def concrete_date(cls, value: str) -> str:
        if not re.search(r"\b\d{4}\b", value):
            raise ValueError("date must include a four-digit year")
        return value

    @field_validator("figures")
    @classmethod
    def measurable_figures(cls, value: list[str]) -> list[str]:
        if not all(re.search(r"\d", item) for item in value):
            raise ValueError("Each figure must contain a measurable number")
        return value


RESEARCH_PROMPT = """Ты — строгий фактологический исследователь. Верни реальный, строго
задокументированный факт сразу в виде ровно одного JSON-объекта, без Markdown и пояснений.
Нужны только проверяемые, задокументированные исторические инциденты, сбои сложных систем,
контринтуитивная физика или эксперименты над восприятием. Обязательны конкретные дата,
имена систем/участников, локация и измеримые цифры. Запрещены общеизвестные факты из пабликов,
нейровысер, общая мотивация и неподтвержденные байки (например, «мозг использует десять процентов»).
Запрещены также «зрачки расширяются», общие советы и недоказанные квантовые чудеса.
Категории: systems — аварии инфраструктуры, software/hardware, авиация, инженерные ошибки;
science — космос, гравитация/время, квантовые и природные аномалии;
mind — когнитивные слепые зоны, иллюзии восприятия, воспроизводимые эксперименты.
Не повторяй темы из списка исключений даже под другим названием.
Поля JSON: topic (каноническое английское имя конкретного инцидента/эксперимента), category,
title, raw_data, date (с годом цифрами), location, systems (непустой массив имен),
figures (непустой массив чисел с единицами), keywords (синонимы имени события на русском и английском,
без общих слов вроде NASA/космос), sources (1-3 объекта с title и url).
Источники: конкретные первичные отчеты, публикации университетов, агентств или
авторов эксперимента. Только существующие прямые HTTPS-ссылки, не главные страницы.
Не выдумывай ссылки. Выбирай другой факт, если не знаешь документального источника.
raw_data — 60-120 слов на русском. Все детали должны подтверждаться указанными источниками.
Отделяй оценки от точных измерений. Не добавляй драматические детали без подтверждения."""


class FactResearcher:
    def __init__(self, api_key: str | None = None, model: str | None = None, history_path: str | Path | None = None):
        key = api_key or os.getenv("ANYMODEL_API_KEY")
        if not key:
            raise ValueError("ANYMODEL_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=key, base_url="https://anymodel.org/v1", timeout=90.0, max_retries=0)
        self.model = model or os.getenv("ANYMODEL_MODEL") or DEFAULT_MODEL
        self.history_path = Path(history_path) if history_path is not None else Path(__file__).resolve().parents[2] / "work/facts_history.json"

    def _history(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read fact history: {self.history_path}") from exc
        if not isinstance(data, list):
            raise ValueError("Fact history must be a JSON list")
        result = []
        for item in data:
            if isinstance(item, str):
                item = {"topic": item, "keywords": []}
            if (not isinstance(item, dict) or not isinstance(item.get("topic"), str)
                    or not item["topic"].strip() or not isinstance(item.get("keywords", []), list)
                    or any(not isinstance(k, str) for k in item.get("keywords", []))):
                raise ValueError("Invalid fact history entry; history was not overwritten")
            result.append(item)
        return result

    @staticmethod
    def _keys(item: dict[str, Any]) -> set[str]:
        return {normalize(s) for s in [item["topic"], *item.get("keywords", [])] if normalize(s)}

    def _remember(self, fact: ResearchedFact) -> bool:
        with file_lock(self.history_path.with_suffix(".json.lock")):
            items = self._history()
            entry = {"topic": fact.topic, "keywords": fact.keywords, "category": fact.category}
            if any(self._keys(entry) & self._keys(old) for old in items):
                return False
            atomic_json(self.history_path, [*items, entry])
            return True

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("fact response must be an object")
        return data

    async def find_facts(self, count: int, category: str = "all") -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be positive")
        if category != "all" and category not in CATEGORIES:
            raise ValueError(f"unsupported category: {category}")
        result: list[dict[str, Any]] = []
        for index in range(count):
            requested = CATEGORIES[index % len(CATEGORIES)] if category == "all" else category
            history = self._history()
            prompt = f"Категория: {requested}. Исключения: {json.dumps(history, ensure_ascii=False)}."
            print(f"[RESEARCH {index + 1}/{count}] {requested}", flush=True)
            completion = await self.client.chat.completions.create(
                model=self.model, temperature=0.8, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": RESEARCH_PROMPT}, {"role": "user", "content": prompt}],
            )
            fact = ResearchedFact.model_validate(self._parse(completion.choices[0].message.content or ""))
            if fact.category != requested:
                raise ValueError(f"Required category: {requested}")
            if not self._remember(fact):
                raise ValueError(f"Already used: {fact.topic}")
            result.append(fact.model_dump(mode="json"))
            print(f"[FACT] {fact.title}\n{fact.raw_data}", flush=True)
        return result

    async def close(self) -> None:
        await self.client.close()
