import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

from src.content.researcher import FactResearcher, ResearchedFact


FACT = {
    "topic": "Patriot clock drift",
    "category": "systems",
    "title": "Ошибка часов Patriot",
    "raw_data": "В тысяча девятьсот девяносто первом году в Дахране программная ошибка накопила измеримый сдвиг часов и повлияла на работу батареи Patriot.",
    "date": "1991-02-25",
    "location": "Дахран, Саудовская Аравия",
    "systems": ["Patriot missile system"],
    "figures": ["100 microseconds", "28 soldiers"],
    "keywords": ["Patriot clock drift"],
    "sources": [{"title": "GAO report", "url": "https://example.com/report"}],
}


class ResearcherTests(unittest.TestCase):
    def setUp(self):
        self.sdk_patch = patch("src.content.researcher.AsyncOpenAI")
        self.sdk = self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def test_fact_structure_validation(self):
        fact = ResearchedFact.model_validate(FACT)
        self.assertEqual(fact.category, "systems")
        with self.assertRaises(ValueError):
            ResearchedFact.model_validate({**FACT, "figures": ["unknown"]})
        for key in ("topic", "date", "location", "sources", "systems", "figures", "keywords"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ResearchedFact.model_validate({k: v for k, v in FACT.items() if k != key})
        for change in ({"title": "   "}, {"category": "all"}, {"date": "unknown"},
                       {"sources": []}, {"sources": [{"title": "Source", "url": "not-a-url"}]},
                       {"keywords": [" "]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ResearchedFact.model_validate({**FACT, **change})

    def test_history_deduplicates_topic_and_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts_history.json"
            path.write_text(json.dumps([{"topic": "Patriot clock drift", "keywords": ["Patriot"]}]), encoding="utf-8")
            researcher = FactResearcher(api_key="test", history_path=path)
            with self.assertRaisesRegex(ValueError, "Already used"):
                researcher._remember([ResearchedFact.model_validate({**FACT, "topic": "Different", "keywords": ["Patriot"]})])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["topic"], "Patriot clock drift")

    def test_find_facts_rejects_duplicate_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            researcher.history_path.write_text(json.dumps([FACT["topic"], *[f"Older topic {i}" for i in range(35)]]), encoding="utf-8")
            def completion(fact):
                response = MagicMock()
                response.choices[0].message.content = json.dumps([fact])
                return response
            researcher.client.chat.completions.create = AsyncMock(return_value=completion(FACT))
            with self.assertRaisesRegex(ValueError, "Already used"):
                asyncio.run(researcher.find_facts(1, "systems"))
            researcher.client.chat.completions.create.assert_awaited_once()
            first_prompt = researcher.client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
            self.assertIn(FACT["topic"], first_prompt)
            self.assertIn("Older topic 34", first_prompt)
            self.assertEqual(len(json.loads(researcher.history_path.read_text(encoding="utf-8"))), 36)

    def test_find_facts_uses_one_json_request_without_fetching_sources(self):
        with tempfile.TemporaryDirectory() as tmp, patch("httpx.AsyncClient") as http:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            response = MagicMock()
            response.choices[0].message.content = json.dumps([FACT])
            researcher.client.chat.completions.create = AsyncMock(return_value=response)

            facts = asyncio.run(researcher.find_facts(1, "systems"))

            self.assertEqual(facts, [FACT])
            researcher.client.chat.completions.create.assert_awaited_once()
            request = researcher.client.chat.completions.create.call_args.kwargs
            self.assertNotEqual(request.get("response_format"), {"type": "json_object"})
            self.assertEqual(request["model"], researcher.model)
            self.assertEqual(self.sdk.call_args.kwargs["max_retries"], 0)
            http.assert_not_called()
            self.assertEqual(json.loads(researcher.history_path.read_text(encoding="utf-8")), [
                {"topic": FACT["topic"], "keywords": FACT["keywords"], "category": FACT["category"]},
            ])

    def test_corrupt_history_is_not_silently_erased(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text("broken json", encoding="utf-8")
            researcher = FactResearcher(api_key="test", history_path=path)
            with self.assertRaises(RuntimeError):
                asyncio.run(researcher.find_facts(1))
            self.assertEqual(path.read_text(), "broken json")
            researcher.client.chat.completions.create.assert_not_called()

    def test_all_balances_categories_in_one_batch_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            batch = []
            for category in ("systems", "science", "mind"):
                batch.append({**FACT, "category": category, "topic": category + " event", "keywords": [category + " experiment"]})
            response = MagicMock()
            response.choices[0].message.content = json.dumps(batch)
            researcher.client.chat.completions.create = AsyncMock(return_value=response)
            facts = asyncio.run(researcher.find_facts(3, "all"))
            self.assertEqual([f["category"] for f in facts], ["systems", "science", "mind"])
            researcher.client.chat.completions.create.assert_awaited_once()
            prompt = researcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
            self.assertIn("ровно 3", prompt)
            self.assertIn(json.dumps(["systems", "science", "mind"]), prompt)
            self.assertEqual(len(json.loads(researcher.history_path.read_text(encoding="utf-8"))), 3)

    def test_bad_api_response_fails_without_retry_or_recording_fact(self):
        for content in ('[{"topic":"missing fields"}]', 'not json', '[]', '[1]',
                        json.dumps(FACT), json.dumps([FACT, FACT]),
                        json.dumps([{**FACT, "category": "science"}])):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
                response = MagicMock()
                response.choices[0].message.content = content
                researcher.client.chat.completions.create = AsyncMock(return_value=response)
                with self.assertRaises(ValueError):
                    asyncio.run(researcher.find_facts(1, "systems"))
                self.assertFalse(researcher.history_path.exists())
                researcher.client.chat.completions.create.assert_awaited_once()

    def test_api_error_is_propagated_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            researcher.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API unavailable"))
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                asyncio.run(researcher.find_facts(1, "systems"))
            self.assertFalse(researcher.history_path.exists())
            researcher.client.chat.completions.create.assert_awaited_once()

    def test_invalid_batch_does_not_partially_update_history(self):
        fresh = {**FACT, "topic": "Fresh event", "keywords": ["Fresh event"]}
        for second in (FACT, fresh, {"topic": "missing fields"}):
            with self.subTest(second=second), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history.json"
                original = json.dumps([FACT["topic"]])
                path.write_text(original, encoding="utf-8")
                researcher = FactResearcher(api_key="test", history_path=path)
                response = MagicMock()
                response.choices[0].message.content = json.dumps([fresh, second])
                researcher.client.chat.completions.create = AsyncMock(return_value=response)
                with self.assertRaises(ValueError):
                    asyncio.run(researcher.find_facts(2, "systems"))
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                researcher.client.chat.completions.create.assert_awaited_once()

    @staticmethod
    def api_error(error_type):
        request = httpx.Request("POST", "https://anymodel.org/v1/chat/completions")
        if error_type is APIConnectionError:
            return error_type(request=request)
        status = {InternalServerError: 500, RateLimitError: 429, BadRequestError: 400}[error_type]
        return error_type("API error", response=httpx.Response(status, request=request), body=None)

    def test_transient_errors_retry_batch_and_recover_on_third_attempt(self):
        for error_type in (InternalServerError, APIConnectionError, RateLimitError):
            with self.subTest(error_type=error_type), tempfile.TemporaryDirectory() as tmp, patch(
                "src.content.researcher.asyncio.sleep", new_callable=AsyncMock
            ) as sleep:
                researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
                batch = [FACT, {**FACT, "topic": "Fresh event", "keywords": ["Fresh event"]}]
                response = MagicMock()
                response.choices[0].message.content = json.dumps(batch)
                researcher.client.chat.completions.create = AsyncMock(side_effect=[
                    self.api_error(error_type), self.api_error(error_type), response,
                ])
                self.assertEqual(asyncio.run(researcher.find_facts(2, "systems")), batch)
                calls = researcher.client.chat.completions.create.call_args_list
                self.assertEqual(len(calls), 3)
                self.assertEqual(calls[0], calls[1])
                self.assertEqual(calls[1], calls[2])
                self.assertEqual([call.args for call in sleep.await_args_list], [(2,), (2,)])
                self.assertEqual(len(json.loads(researcher.history_path.read_text(encoding="utf-8"))), 2)

    def test_transient_errors_stop_after_three_attempts(self):
        for error_type in (InternalServerError, APIConnectionError, RateLimitError):
            with self.subTest(error_type=error_type), tempfile.TemporaryDirectory() as tmp, patch(
                "src.content.researcher.asyncio.sleep", new_callable=AsyncMock
            ) as sleep:
                researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
                error = self.api_error(error_type)
                researcher.client.chat.completions.create = AsyncMock(side_effect=error)
                with self.assertRaises(error_type) as caught:
                    asyncio.run(researcher.find_facts(1, "systems"))
                self.assertIs(caught.exception, error)
                self.assertEqual(researcher.client.chat.completions.create.await_count, 3)
                self.assertEqual([call.args for call in sleep.await_args_list], [(2,), (2,)])
                self.assertFalse(researcher.history_path.exists())

    def test_nontransient_api_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.content.researcher.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            researcher.client.chat.completions.create = AsyncMock(side_effect=self.api_error(BadRequestError))
            with self.assertRaises(BadRequestError):
                asyncio.run(researcher.find_facts(1, "systems"))
            researcher.client.chat.completions.create.assert_awaited_once()
            sleep.assert_not_awaited()
            self.assertFalse(researcher.history_path.exists())


if __name__ == "__main__":
    unittest.main()
