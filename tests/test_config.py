import unittest
from pathlib import Path

from src.config import Settings


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        settings = Settings(root=Path("project"))
        self.assertEqual(settings.voice, "ru-RU-DmitryNeural")
        self.assertEqual(settings.fps, 30)
        self.assertEqual(settings.words_per_subtitle, 2)
        self.assertEqual(settings.background_path, Path("project/assets/background.mp4"))

    def test_overrides(self):
        settings = Settings(fps=60, words_per_subtitle=3)
        self.assertEqual((settings.fps, settings.words_per_subtitle), (60, 3))


if __name__ == "__main__":
    unittest.main()

