import unittest

from src.content.adapter import AdaptedStory, StoryAdapter


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
        with self.assertRaises(ValueError):
            AdaptedStory(
                title="Слишком коротко",
                text="мало слов",
                tags=["#истории", "#реддит", "#драма", "#шортс"],
            )


if __name__ == "__main__":
    unittest.main()
