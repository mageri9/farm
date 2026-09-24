import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.video import VideoError, render_video


class FFmpegLoggingTests(unittest.TestCase):
    def test_log_flood_uses_file_and_preserves_bounded_error_tail(self):
        def fail(command, **kwargs):
            self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
            self.assertNotIn('capture_output', kwargs)
            self.assertIn('-nostats', command)
            stream = kwargs['stderr']
            for _ in range(128):
                stream.write(b'x' * 65536)
            stream.write(b'\nfinal diagnostic\n')
            raise subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('src.video.check_executable', return_value='ffmpeg'), \
                    patch('src.video.subprocess.run', side_effect=fail), \
                    self.assertLogs('shorts', level='INFO') as logs:
                with self.assertRaises(VideoError) as error:
                    render_video(root / 'bg.mp4', root / 'voice.wav', root / 'sub.ass',
                                 root / 'out.mp4', 2.0, 0.0, Settings(root=root))
            self.assertIn('final diagnostic', str(error.exception))
            self.assertLessEqual(len(error.exception.__cause__.stderr), 65536)
            self.assertEqual(len(logs.output), 1)
            self.assertIn('FFmpeg command:', logs.output[0])
