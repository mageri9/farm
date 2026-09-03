import unittest

from src.video import VideoError, random_start


class VideoTests(unittest.TestCase):
    def test_random_start_is_seeded_and_in_range(self):
        first = random_start(100.0, 20.0, seed=123)
        second = random_start(100.0, 20.0, seed=123)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 80.0)

    def test_equal_durations_start_at_zero(self):
        self.assertEqual(random_start(20.0, 20.0), 0.0)

    def test_short_background_fails(self):
        with self.assertRaisesRegex(VideoError, "too short"):
            random_start(19.0, 20.0)


if __name__ == "__main__":
    unittest.main()

