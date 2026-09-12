import unittest

from src.content.adapter import DEFAULT_MODEL, AdaptedStory, StoryAdapter


class AdapterTests(unittest.TestCase):
    def test_parse_json_removes_markdown_fence(self) -> None:
        data = StoryAdapter._parse_json('```json\n{"title": "Тест"}\n```')
        self.assertEqual(data, {"title": "Тест"})

    def test_adapted_story_accepts_required_shape(self) -> None:
        words = " ".join("слово" for i in range(50))
        story = AdaptedStory(
            title="Семейная тайна раскрылась",
            text=words,
            tags=["#истории", "#реддит", "#драма", "#шортс"],
        )
        self.assertEqual(len(story.text.split()), 50)

    def test_adapted_story_rejects_wrong_word_count(self) -> None:
        for word_count in (44, 56):
            with self.subTest(word_count=word_count), self.assertRaises(ValueError):
                AdaptedStory(
                    title="Неверная длина",
                    text=" ".join("слово" for i in range(word_count)),
                    tags=["#истории", "#реддит", "#драма", "#шортс"],
                )

    def test_story_adapter_uses_cli_default_model(self) -> None:
        adapter = StoryAdapter("test-key")
        self.assertEqual(adapter.model, DEFAULT_MODEL)

    def test_rejects_digits_in_narration(self):
        with self.assertRaises(ValueError):
            AdaptedStory(title="Тест", text="слово " * 49 + "1991")


if __name__ == "__main__":
    unittest.main()
