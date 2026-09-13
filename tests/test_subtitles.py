import tempfile
import unittest
from pathlib import Path

from src.subtitles import choose_font, escape_subtitle_path, format_time_ass, group_words, write_ass
from src.tts import WordBoundary


class SubtitleTests(unittest.TestCase):
    def test_choose_font_uses_configured_name_or_arial(self):
        self.assertEqual(choose_font(" Montserrat "), "Montserrat")
        self.assertEqual(choose_font(""), "Arial")

    def test_format_time_ass_rounds_centiseconds(self):
        self.assertEqual(format_time_ass(1.234), "0:00:01.23")
        self.assertEqual(format_time_ass(1.235), "0:00:01.24")
        self.assertEqual(format_time_ass(3661.999), "1:01:02.00")
        self.assertEqual(format_time_ass(-1), "0:00:00.00")

    def test_group_words_uses_requested_size(self):
        words = [WordBoundary(str(i), i, i + 0.5) for i in range(5)]
        groups = group_words(words, 2)
        self.assertEqual([len(group) for group in groups], [2, 2, 1])
        self.assertEqual((groups[0][0].start, groups[0][-1].end), (0, 1.5))

    def test_write_ass_uses_group_boundaries(self):
        words = [WordBoundary("hello", 1.0, 1.2), WordBoundary("world", 1.3, 1.8)]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.ass"
            self.assertEqual(write_ass(words, target), 1)
            text = target.read_text(encoding="utf-8")
        self.assertIn("0:00:01.00,0:00:01.80", text)
        self.assertIn("hello world", text)

    def test_capitalized_next_word_starts_a_new_chunk_without_punctuation(self):
        for prefix in ("", '"', "'", "\u00ab", "(", "[", "{", "\u201e", " \u201c("):
            for size in (2, 4):
                with self.subTest(prefix=prefix, size=size):
                    words = [WordBoundary("\u043a\u0440\u044b\u0448\u0443", 1.0, 1.2),
                             WordBoundary(prefix + "\u041d\u0430", 1.5, 1.7),
                             WordBoundary("\u0431\u043e\u0440\u0442\u0443", 1.8, 2.0)]
                    self.assertEqual(group_words(iter(words), size), [words[:1], words[1:]])

    def test_terminal_punctuation_still_closes_chunks(self):
        for ending in (".", "!", "?", '."', "!\u00bb", "?)"):
            with self.subTest(ending=ending):
                words = [WordBoundary("end" + ending, 0, 0.5), WordBoundary("next", 0.7, 1)]
                self.assertEqual(group_words(words, 4), [[words[0]], [words[1]]])

    def test_capitalized_words_never_follow_another_word_in_a_chunk(self):
        words = [WordBoundary(word, i, i + 0.5) for i, word in
                 enumerate(["first", "Second", "Third", "last", "Fourth"])]
        groups = group_words(words, 5)
        self.assertEqual(groups, [words[:1], words[1:2], words[2:4], words[4:]])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.ass"
            write_ass(words, target, words_per_subtitle=5)
            text = target.read_text(encoding="utf-8")
        self.assertIn("0:00:02.00,0:00:03.50,Default,,0,0,0,,Third last", text)

    def test_escape_subtitle_path_for_windows_filter(self):
        escaped = escape_subtitle_path(Path("D:/folder with space/sub,one.ass"))
        self.assertIn(r"D\:/", escaped)
        self.assertIn(r"sub\,one.ass", escaped)

    def test_escape_windows_backslash_path_without_resolving_it(self):
        self.assertEqual(escape_subtitle_path(r"C:\project\subtitles.ass"), r"C\:/project/subtitles.ass")

    def test_escape_path_with_apostrophe(self):
        escaped = escape_subtitle_path(r"C:\folder\John's subtitles.ass")
        self.assertIn(r"John\'s subtitles.ass", escaped)

    def test_escape_normalized_posix_path(self):
        self.assertEqual(escape_subtitle_path("/tmp/subtitles.ass"), "/tmp/subtitles.ass")


if __name__ == "__main__":
    unittest.main()
