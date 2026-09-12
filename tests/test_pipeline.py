import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.pipeline import ShortsPipeline


class BackgroundSelectionTests(unittest.TestCase):
    @patch("src.pipeline.check_executable", return_value="ffmpeg")
    @patch("src.pipeline.probe_duration", return_value=40.0)
    def test_selection_and_preflight_use_the_same_pool(self, probe, _check):
        for pool_count, root_count in ((2, 2), (0, 2), (1, 2), (0, 1), (0, 0)):
            with self.subTest(pool=pool_count, assets=root_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                assets = root / "assets"
                pool = assets / "backgrounds"
                pool.mkdir(parents=True)
                for idx in range(pool_count):
                    (pool / f"clip{idx}.mp4").touch()
                for idx in range(root_count):
                    (assets / f"clip{idx}.mp4").touch()
                # Directories with a video suffix must not count as clips.
                (pool / "invalid.mp4").mkdir()
                fallback = root / "fallback.mp4"
                fallback.touch()
                pipeline = ShortsPipeline(Settings(root=root, background=fallback))
                pipeline.preflight()
                backgrounds, offsets = pipeline._select_backgrounds(24.0, seed=7)
                expected = pool_count if pool_count else root_count
                if expected >= 2:
                    self.assertEqual(len(set(backgrounds)), 2)
                    expected_parent = pool if pool_count else assets
                    self.assertTrue(all(path.parent == expected_parent for path in backgrounds))
                    fallback.unlink()
                    pipeline.preflight()
                else:
                    expected_background = (pool if pool_count else assets) / "clip0.mp4" if expected else fallback
                    self.assertEqual(backgrounds, [expected_background])
                    if expected:
                        fallback.unlink()
                        pipeline.preflight()
                self.assertEqual(len(backgrounds), len(offsets))
                self.assertTrue(all(0.0 <= offset <= 40.0 - 24.0 / len(backgrounds) - 0.5 for offset in offsets))
                self.assertEqual((backgrounds, offsets), pipeline._select_backgrounds(24.0, seed=7))
