import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
            self.assertFalse(researcher._remember(ResearchedFact.model_validate({**FACT, "topic": "Different", "keywords": ["Patriot"]})))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["topic"], "Patriot clock drift")

    def test_find_facts_retries_duplicate_from_api(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(FactResearcher, "_verify_sources", new=AsyncMock(return_value=True)):
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            researcher.history_path.write_text(json.dumps([FACT["topic"], *[f"Older topic {i}" for i in range(35)]]), encoding="utf-8")
            def completion(fact):
                response = MagicMock()
                response.choices[0].message.content = json.dumps(fact)
                return response
            researcher.client.chat.completions.create = AsyncMock(side_effect=[completion(FACT), completion({**FACT, "topic": "Mars Climate Orbiter", "keywords": ["MCO"], "title": "Ошибка Mars Climate Orbiter"})])
            facts = asyncio.run(researcher.find_facts(1, "systems"))
            self.assertEqual(facts[0]["topic"], "Mars Climate Orbiter")
            self.assertEqual(researcher.client.chat.completions.create.call_count, 2)
            first_prompt = researcher.client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
            self.assertNotIn(FACT["topic"], first_prompt)
            self.assertIn("Older topic 34", first_prompt)
            self.assertEqual(len(json.loads(researcher.history_path.read_text(encoding="utf-8"))), 37)

    def test_corrupt_history_is_not_silently_erased(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text("broken json", encoding="utf-8")
            researcher = FactResearcher(api_key="test", history_path=path)
            with self.assertRaises(RuntimeError):
                asyncio.run(researcher.find_facts(1))
            self.assertEqual(path.read_text(), "broken json")
            researcher.client.chat.completions.create.assert_not_called()

    def test_all_balances_categories_and_excludes_current_batch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(FactResearcher, "_verify_sources", new=AsyncMock(return_value=True)):
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            responses = []
            for category in ("systems", "science", "mind"):
                response = MagicMock()
                response.choices[0].message.content = json.dumps({**FACT, "category": category, "topic": category + " event", "keywords": [category + " experiment"]})
                responses.append(response)
            researcher.client.chat.completions.create = AsyncMock(side_effect=responses)
            facts = asyncio.run(researcher.find_facts(3, "all"))
            self.assertEqual([f["category"] for f in facts], ["systems", "science", "mind"])
            prompt = researcher.client.chat.completions.create.call_args_list[1].kwargs["messages"][1]["content"]
            self.assertIn("systems event", prompt)

    def test_bad_api_response_retries_without_recording_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            researcher = FactResearcher(api_key="test", history_path=Path(tmp) / "history.json")
            response = MagicMock()
            response.choices[0].message.content = '{"topic":"missing fields"}'
            researcher.client.chat.completions.create = AsyncMock(return_value=response)
            with self.assertRaises(RuntimeError):
                asyncio.run(researcher.find_facts(1, "systems"))
            self.assertFalse(researcher.history_path.exists())
            self.assertEqual(researcher.client.chat.completions.create.call_count, 5)


if __name__ == "__main__":
    unittest.main()
