import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.content.generator import GeneratedStory, UnifiedStoryGenerator, load_prompt


def story(words=75):
    return {
        "topic": "War hammer", "category": "systems",
        "source_url": "https://example.org/hammer", "title": "Боевой молот",
        "beats": {
            "establishing": "Молот.", "tension": "Рычаг.",
            "subject": "Механика.", "aftermath": " ".join(["слово"] * (words - 3)) + ".",
        },
        "tags": ["#история"],
    }


class ForensicTests(unittest.IsolatedAsyncioTestCase):
    def test_word_boundaries(self):
        for count in (70, 88):
            self.assertEqual(len(GeneratedStory.model_validate(story(count)).text.split()), count)
        for count in (69, 89):
            with self.subTest(count=count), self.assertRaises(ValueError):
                GeneratedStory.model_validate(story(count))

    def test_narration_restrictions_and_empty_beats(self):
        for token in ("1", ":", ";", "—", "«", "»", "(", ")", "*", "!", "?", "-", '"', "Latin", ""):
            value = story()
            value["beats"]["establishing"] = token
            with self.subTest(token=token), self.assertRaises(ValueError):
                GeneratedStory.model_validate(value)

    def test_template_is_loaded_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.txt"
            path.write_text("Экспонат {item}", encoding="utf-8")
            with patch("src.content.generator.PROMPT_PATH", path):
                self.assertTrue(load_prompt("клевец").startswith("Экспонат клевец"))
                path.write_text("Разбор {item}", encoding="utf-8")
                self.assertTrue(load_prompt("кираса").startswith("Разбор кираса"))

    async def test_json_contract_and_retry(self):
        def response(value):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([value])))])

        with tempfile.TemporaryDirectory() as tmp, patch("src.content.generator.AsyncOpenAI") as client:
            create = AsyncMock(side_effect=[response(story(84)), response(story())])
            client.return_value.chat.completions.create = create
            generator = UnifiedStoryGenerator(api_key="test", history_path=Path(tmp) / "history.json")
            result = await generator.generate(1, topic="клевец")
            self.assertEqual(create.await_count, 2)
            self.assertEqual(len(result[0]["text"].split()), 75)
            self.assertEqual(result[0]["beats"], story()["beats"])
            prompt = create.call_args.kwargs["messages"][0]["content"]
            self.assertIn("экспоната: клевец", prompt)
            self.assertNotIn("{item}", prompt)
            self.assertIn("subject", prompt)
            self.assertTrue((Path(tmp) / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
