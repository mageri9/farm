import unittest

from src.content.adapter import DEFAULT_MODEL, AdaptedStory, StoryAdapter


class AdapterTests(unittest.TestCase):
    def test_parse_json_removes_markdown_fence(self) -> None:
        data = StoryAdapter._parse_json('```json\n{"title": "Тест"}\n```')
        self.assertEqual(data, {"title": "Тест"})

    def test_adapted_story_accepts_required_shape(self) -> None:
        words = " ".join(f"слово{i}" for i in range(70))
        story = AdaptedStory(
            title="Семейная тайна раскрылась",
            text=words,
            tags=["#истории", "#реддит", "#драма", "#шортс"],
        )
        self.assertEqual(len(story.text.split()), 70)

    def test_adapted_story_rejects_wrong_word_count(self) -> None:
        for word_count in (69, 86):
            with self.subTest(word_count=word_count), self.assertRaises(ValueError):
                AdaptedStory(
                    title="Неверная длина",
                    text=" ".join(f"слово{i}" for i in range(word_count)),
                    tags=["#истории", "#реддит", "#драма", "#шортс"],
                )

    def test_story_adapter_uses_cli_default_model(self) -> None:
        adapter = StoryAdapter("test-key")
        self.assertEqual(adapter.model, DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
