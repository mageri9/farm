import unittest

from fetch_content import AI_TOPICS


class FetchContentTests(unittest.TestCase):
    def test_ai_topics_cover_twenty_story_batch_without_repeats(self) -> None:
        self.assertGreaterEqual(len(AI_TOPICS), 20)
        self.assertEqual(len(AI_TOPICS), len(set(AI_TOPICS)))


if __name__ == "__main__":
    unittest.main()
