import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.video import VideoError, find_item_images, overlay_spans, random_start, render_video


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

    def test_short_background_starts_at_zero_for_looping(self):
        self.assertEqual(random_start(19.0, 20.0), 0.0)

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    @patch("src.video.Path.mkdir")
    @patch("src.video.Path.is_file", return_value=False)
    def test_render_filter_has_even_crop_pts_and_fontsdir(self, _is_file, _mkdir, _check, run):
        settings = Settings(root=Path("D:/project"))
        render_video(Path("D:/bg.mp4"), Path("D:/audio.mp3"), Path("D:/subtitles.ass"),
                     Path("D:/out.mp4"), 2.0, 1.0, settings)
        command = run.call_args.args[0]
        vf = command[command.index("-vf") + 1]
        self.assertEqual(
            vf,
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,"
            "setsar=1,fps=30,setpts=PTS-STARTPTS,"
            "ass='D\\:/subtitles.ass':fontsdir='D\\:/project/assets/fonts'",
        )

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    @patch("src.video.Path.mkdir")
    @patch("src.video.Path.is_file", return_value=False)
    def test_render_maps_streams_to_an_encoded_output(self, _is_file, _mkdir, _check, run):
        for count in (1, 2, 3, 4):
            with self.subTest(clips=count):
                output = Path("D:/render/output.mp4")
                render_video(
                    [Path(f"D:/bg{idx}.mp4") for idx in range(count)],
                    Path("D:/audio.mp3"), Path("D:/subtitles.ass"), output,
                    24.0, [0.0] * count, Settings(root=Path("D:/project")),
                )
                command = run.call_args.args[0]
                self.assertEqual(command[-1], str(output))
                maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
                self.assertEqual(maps, ["0:v:0" if count == 1 else "[vout]", f"{count}:a:0"])
                self.assertEqual(command[command.index("-t") + 1], "24.000")
                self.assertEqual(command[command.index("-c:v") + 1], "libx264")
                self.assertEqual(command[command.index("-c:a") + 1], "aac")
                if count == 1:
                    self.assertIn("-vf", command)
                    self.assertNotIn("-filter_complex", command)
                else:
                    graph = command[command.index("-filter_complex") + 1]
                    self.assertIn(f"concat=n={count}:v=1:a=0,ass=", graph)
                    self.assertTrue(graph.endswith("[vout]"))
                    self.assertNotIn("[vbg]", graph)

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    def test_ambient_audio_is_optional_and_looped_after_narration(self, _check, run):
        for count in (1, 4):
            for present in (False, True):
                with self.subTest(clips=count, ambient=present), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    settings = Settings(root=root)
                    ambient = settings.ambient_path
                    if present:
                        ambient.parent.mkdir(parents=True)
                        ambient.touch()
                    render_video([root / f"clip{i}.mp4" for i in range(count)],
                                 root / "voice.wav", root / "subtitles.ass", root / "out.mp4",
                                 24.0, [0.0] * count, settings)
                    command = run.call_args.args[0]
                    inputs = [command[i + 1] for i, arg in enumerate(command) if arg == "-i"]
                    self.assertEqual(len(inputs), count + 1 + int(present))
                    maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
                    self.assertEqual(maps[-1], "[aout]" if present else f"{count}:a:0")
                    if present:
                        ambient_index = command.index(str(ambient))
                        self.assertEqual(command[ambient_index - 3:ambient_index], ["-stream_loop", "-1", "-i"])
                        graph = command[command.index("-filter_complex") + 1]
                        self.assertIn(f"[{count + 1}:a]volume=0.08[amb];[{count}:a][amb]", graph)
                        self.assertIn("amix=inputs=2:duration=first:dropout_transition=2[aout]", graph)
                        if count == 4:
                            self.assertIn("[vout];", graph)


class OverlaySpanTests(unittest.TestCase):
    DURATIONS = [4.0, 6.0, 8.0, 2.0]  # биты 1-4, границы: 4, 10, 18, 20

    def test_single_image_covers_beats_two_and_three(self):
        self.assertEqual(overlay_spans(self.DURATIONS, 1), [(4.0, 18.0)])

    def test_two_images_map_to_beat_two_and_beat_three(self):
        self.assertEqual(overlay_spans(self.DURATIONS, 2), [(4.0, 10.0), (10.0, 18.0)])

    def test_beat_one_and_four_are_never_covered(self):
        for count in (1, 2):
            for start, end in overlay_spans(self.DURATIONS, count):
                self.assertGreaterEqual(start, 4.0)
                self.assertLessEqual(end, 18.0)

    def test_no_spans_without_images_or_four_beats(self):
        self.assertEqual(overlay_spans(self.DURATIONS, 0), [])
        self.assertEqual(overlay_spans([5.0, 5.0], 2), [])
        self.assertEqual(overlay_spans(None, 1), [])
        self.assertEqual(overlay_spans([4.0, 0.0, 8.0, 2.0], 1), [])


class ItemAssetTests(unittest.TestCase):
    def test_folder_images_are_sorted_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            folder = assets / "items" / "klevec"
            folder.mkdir(parents=True)
            for name in ("2.png", "1.png", "3.webp", "notes.txt"):
                (folder / name).touch()
            self.assertEqual([p.name for p in find_item_images(assets, "klevec")],
                             ["1.png", "2.png", "3.webp"])

    def test_flat_file_is_used_when_folder_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            (assets / "items").mkdir(parents=True)
            (assets / "items" / "klevec.jpg").touch()
            self.assertEqual([p.name for p in find_item_images(assets, "klevec")], ["klevec.jpg"])

    def test_missing_assets_are_fail_soft(self):
        with tempfile.TemporaryDirectory() as directory:
            for slug in ("klevec", "", "../escape", "."):
                self.assertEqual(find_item_images(Path(directory), slug), [])


class OverlayRenderTests(unittest.TestCase):
    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    def test_overlay_inputs_and_graph_are_wired_after_audio(self, _check, run):
        for count in (1, 2):
            with self.subTest(images=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = Settings(root=root)
                images = [root / f"{i}.png" for i in range(count)]
                for image in images:
                    image.touch()
                render_video([root / f"clip{i}.mp4" for i in range(4)], root / "voice.wav",
                             root / "subtitles.ass", root / "out.mp4", 20.0, [0.0] * 4,
                             settings, clip_durations=[4.0, 6.0, 8.0, 2.0], overlays=images)
                command = run.call_args.args[0]
                inputs = [command[i + 1] for i, arg in enumerate(command) if arg == "-i"]
                # 4 фона + озвучка + картинки; картинки строго последними.
                self.assertEqual(len(inputs), 5 + count)
                self.assertEqual(inputs[5:], [str(p) for p in images])
                maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
                self.assertEqual(maps, ["[vout]", "4:a:0"])
                graph = command[command.index("-filter_complex") + 1]
                self.assertIn("concat=n=4:v=1:a=0[vbg];", graph)
                self.assertTrue(graph.endswith("[vout]"))
                # Субтитры прожигаются последними, поверх картинки.
                self.assertLess(graph.index("overlay="), graph.index("ass="))
                for order in range(count):
                    self.assertIn(f"scale=850:-1,format=rgba", graph)
                    self.assertIn(f"[ov{order}]", graph)
                    self.assertIn("(W-w)/2:(H-h)/2-120", graph)
                if count == 1:
                    self.assertIn("enable='between(t,4.000,18.000)'", graph)
                else:
                    self.assertIn("enable='between(t,4.000,10.000)'", graph)
                    self.assertIn("enable='between(t,10.000,18.000)'", graph)

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    def test_missing_images_fall_back_to_backgrounds_only(self, _check, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_video([root / f"clip{i}.mp4" for i in range(4)], root / "voice.wav",
                         root / "subtitles.ass", root / "out.mp4", 20.0, [0.0] * 4,
                         Settings(root=root), clip_durations=[4.0, 6.0, 8.0, 2.0],
                         overlays=[root / "absent.png"])
            command = run.call_args.args[0]
            graph = command[command.index("-filter_complex") + 1]
            self.assertNotIn("overlay=", graph)
            self.assertIn("concat=n=4:v=1:a=0,ass=", graph)

    @patch("src.video.subprocess.run")
    @patch("src.video.check_executable", return_value="ffmpeg")
    def test_overlays_coexist_with_ambient_audio_mix(self, _check, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(root=root)
            settings.ambient_path.parent.mkdir(parents=True)
            settings.ambient_path.touch()
            image = root / "1.png"
            image.touch()
            render_video([root / f"clip{i}.mp4" for i in range(4)], root / "voice.wav",
                         root / "subtitles.ass", root / "out.mp4", 20.0, [0.0] * 4,
                         settings, clip_durations=[4.0, 6.0, 8.0, 2.0], overlays=[image])
            command = run.call_args.args[0]
            # Картинка идет после ambient, поэтому звуковые индексы не смещаются.
            inputs = [command[i + 1] for i, arg in enumerate(command) if arg == "-i"]
            self.assertEqual(inputs[-1], str(image))
            graph = command[command.index("-filter_complex") + 1]
            self.assertIn("[5:a]volume=0.08[amb];[4:a][amb]", graph)
            self.assertIn("amix=inputs=2:duration=first:dropout_transition=2[aout]", graph)
            self.assertIn("[6:v]scale=850:-1", graph)
            maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
            self.assertEqual(maps, ["[vout]", "[aout]"])


if __name__ == "__main__":
    unittest.main()
