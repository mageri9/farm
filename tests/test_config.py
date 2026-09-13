import unittest
import os
from unittest.mock import patch
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

    def test_ambient_paths(self):
        root = Path.cwd()
        self.assertEqual(Settings(root=root).ambient_path, root / "assets/audio/ambient.mp3")
        self.assertEqual(Settings(root=root, ambient_audio="music/bed.mp3").ambient_path,
                         root / "music/bed.mp3")
        self.assertEqual(Settings(root=root, ambient_audio=root / "bed.mp3").ambient_path,
                         root / "bed.mp3")

    @patch("src.config.load_dotenv")
    def test_ambient_path_from_env(self, _load_dotenv):
        with patch.dict(os.environ, {"SHORTS_AMBIENT_AUDIO": "music/bed.mp3"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.ambient_path, settings.root / "music/bed.mp3")


if __name__ == "__main__":
    unittest.main()
