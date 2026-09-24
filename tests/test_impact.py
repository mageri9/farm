import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.subtitles import write_ass
from src.tts import WordBoundary
from src.video import render_video


class ImpactTests(unittest.TestCase):
    def test_optional_impact_tracks_every_overlay_before_master(self):
        for ambient in (False, True):
            for present in (False, True):
                for count in (0, 1, 3, 4):
                    with self.subTest(ambient=ambient, impact=present, images=count), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        settings = Settings(root=root)
                        impact = settings.assets_dir / "sfx" / "impact.wav"
                        impact.parent.mkdir(parents=True)
                        if present:
                            impact.touch()
                        if ambient:
                            settings.ambient_path.parent.mkdir(parents=True, exist_ok=True)
                            settings.ambient_path.touch()
                        images = [root / f"{i}.png" for i in range(count)]
                        for image in images:
                            image.touch()
                        with patch("src.video.check_executable", return_value="ffmpeg"), patch("src.video.subprocess.run") as run:
                            render_video(root / "bg.mp4", root / "voice.wav", root / "sub.ass",
                                         root / "out.mp4", 13.3, 0.0, settings, overlays=images)
                        command = run.call_args.args[0]
                        graph = command[command.index("-filter_complex") + 1] if "-filter_complex" in command else ""
                        enabled = present and count > 0
                        self.assertEqual(str(impact) in command, enabled)
                        self.assertEqual(graph.count("adelay="), count if enabled else 0)
                        self.assertEqual("-af" in command, not (ambient or enabled))
                        self.assertEqual(" ".join(command).count("loudnorm=I=-14:LRA=7:TP=-1.5"), 1)
                        if enabled:
                            for i in range(count):
                                delay = round((0.8 + i * 12 / count) * 1000)
                                self.assertIn(f"adelay={delay}:all=1,volume=0.28[impact{i}]", graph)
                            self.assertIn(f"amix=inputs={1 + int(ambient) + count}:duration=first", graph)
                            self.assertLess(graph.rindex("volume=0.28"), graph.index("loudnorm="))

    def test_default_subtitle_style_is_readable_and_safe(self):
        self.assertEqual(Settings().font_size, 74)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub.ass"
            write_ass([WordBoundary("hello", 0, 1)], path)
            style = next(line for line in path.read_text().splitlines() if line.startswith("Style:"))
            fields = style.split(",")
            self.assertEqual(fields[2], "74")
            self.assertEqual(fields[-2], "260")
