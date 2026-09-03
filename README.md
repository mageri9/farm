# Shorts Generator

A small Windows-friendly CLI that turns text into a vertical `1080x1920` MP4. It uses Edge TTS word boundaries for timing, creates burned-in ASS subtitles, and delegates video processing to FFmpeg without loading video into Python memory.

## Requirements

- Python 3.11+
- FFmpeg and ffprobe available on `PATH`
- Internet access for Microsoft Edge TTS
- A sufficiently long background video

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install FFmpeg using a Windows package manager or a build from [ffmpeg.org](https://ffmpeg.org/download.html), add its `bin` directory to `PATH`, then verify:

```powershell
ffmpeg -version
ffprobe -version
```

## Background

Place a landscape gameplay video at:

```text
assets\background.mp4
```

It must be at least as long as the generated voice track. The generator selects a random valid segment and crops its center to `9:16`.

Optional `.ttf` or `.otf` fonts can be placed in `assets\fonts`. The default style requests Montserrat; if it is not installed, FFmpeg/libass selects an available fallback. Use a font installed in Windows by changing `font_name` in `src/config.py`.

## Run

```powershell
python main.py "Тестовый текст для генерации короткого видео."
python main.py --text "Однажды разработчик решил автоматизировать создание Shorts..."
python main.py --input story.txt --output output\story.mp4 --seed 123
python main.py --text "Текст" --voice ru-RU-SvetlanaNeural --rate +5% --fps 60
```

By default, the finished file is written to `output\final_YYYYMMDD_HHMMSS.mp4`. Successful renders remove `work\audio.mp3` and `work\subtitles.ass`; pass `--keep-work` to retain them.

## Tests

The unit tests do not call Edge TTS or render video:

```powershell
python -m unittest discover -s tests -v
```

## Common errors

- `ffmpeg was not found`: install FFmpeg and add both executables to `PATH`.
- `Background video not found`: add `assets\background.mp4`.
- `Background video is too short`: use a longer source video.
- `Edge TTS returned no WordBoundary`: retry and verify network/TTS availability; timed subtitles cannot be generated safely without these events.

