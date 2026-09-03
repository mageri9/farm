import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.video import VideoError, random_start, render_video


class VideoTests(unittest.TestCase):
    def test_crop_width_is_even_for_common_source_heights(self):
        crop_widths = [int(height * 9 / 16 / 2) * 2 for height in (720, 1080, 1440, 2160)]
        self.assertEqual(crop_widths[1], 606)
        self.assertTrue(all(width % 2 == 0 for width in crop_widths))

    def test_random_start_is_seeded_and_in_range(self):
        first = random_start(100.0, 20.0, seed=123)
        second = random_start(100.0, 20.0, seed=123)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 79.5)

    def test_equal_durations_start_at_zero(self):
        self.assertEqual(random_start(20.0, 20.0), 0.0)

    def test_nearly_equal_durations_start_at_zero(self):
        self.assertEqual(random_start(20.4, 20.0), 0.0)

    def test_random_start_keeps_half_second_tail_buffer(self):
        self.assertEqual(random_start(20.5, 20.0), 0.0)

    def test_short_background_fails(self):
        with self.assertRaisesRegex(VideoError, "too short"):
            random_start(19.0, 20.0)

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    def test_render_filter_has_even_crop_pts_and_fontsdir(self, _check, run):
        settings = Settings(root=Path("D:/project"))
        render_video(Path("D:/bg.mp4"), Path("D:/audio.mp3"), Path("D:/subtitles.ass"),
                     Path("D:/out.mp4"), 2.0, 1.0, settings)
        command = run.call_args.args[0]
        vf = command[command.index("-vf") + 1]
        self.assertEqual(
            vf,
            "crop=trunc(ih*9/16/2)*2:ih:(iw-ow)/2:0,"
            "scale=1080:1920,setpts=PTS-STARTPTS,"
            "subtitles='D\\:/subtitles.ass':fontsdir='D\\:/project/assets/fonts'",
        )


if __name__ == "__main__":
    unittest.main()
