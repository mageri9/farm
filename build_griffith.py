import os
import sys
import random
import subprocess
from pathlib import Path
from src.config import Settings
from src.video import render_video

AUDIO_PATH = Path("test_voice.mp3")
OUTPUT_DIR = Path("output/griffith_cloudflare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "cloudflare_2019.mp4"
SUBTITLES_PATH = OUTPUT_DIR / "subtitles.ass"

SCRIPT_TEXT = (
    "Как одна строчка может выключить сайты по всему миру? "
    "Второго июля две тысячи девятнадцатого Cloudflare разослала правило защиты на все серверы. "
    "Оно проверяло каждый запрос так тяжело, что процессоры захлебнулись работой. "
    "В тринадцать сорок две возникли ошибки. "
    "Через двадцать семь минут правило отменили. "
    "Это был не взлом, а одна команда."
)

def get_duration(audio_file: Path, settings: Settings) -> float:
    cmd = [
        settings.ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_file)
    ]
    res = subprocess.check_output(cmd).decode().strip()
    return float(res)

def build_ass(words, ass_path: Path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,60,60,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    chunk = []
    for i, w in enumerate(words):
        chunk.append(w)
        text = w["text"].strip()
        is_terminal = text.endswith(('.', '!', '?'))
        is_next_cap = (i + 1 < len(words)) and words[i + 1]["text"].strip() and words[i + 1]["text"].strip()[0].isupper()
        
        if len(chunk) >= 2 or is_terminal or is_next_cap:
            start_s = chunk[0]["start"]
            end_s = chunk[-1]["end"]
            line_text = " ".join(c["text"] for c in chunk)
            events.append(f"Dialogue: 0,{fmt(start_s)},{fmt(end_s)},Default,,0,0,0,,{line_text}")
            chunk = []
    
    if chunk:
        start_s = chunk[0]["start"]
        end_s = chunk[-1]["end"]
        line_text = " ".join(c["text"] for c in chunk)
        events.append(f"Dialogue: 0,{fmt(start_s)},{fmt(end_s)},Default,,0,0,0,,{line_text}")

    ass_path.write_text(header + "\n".join(events), encoding="utf-8")

def main():
    if not AUDIO_PATH.exists():
        print(f"[ОШИБКА] Файл {AUDIO_PATH} не найден в папке проекта! Положите его туда.")
        return

    settings = Settings.from_env()

    print(f"[1/3] Читаем аудио: {AUDIO_PATH}")
    duration = get_duration(AUDIO_PATH, settings)
    print(f"Длительность: {duration:.2f} сек.")

    print("[2/3] Генерируем караоке-субтитры...")
    raw_words = SCRIPT_TEXT.split()
    total_len = sum(len(w) for w in raw_words)
    words = []
    curr = 0.0
    for w in raw_words:
        w_dur = (len(w) / total_len) * duration
        words.append({"text": w, "start": curr, "end": curr + w_dur})
        curr += w_dur
    build_ass(words, SUBTITLES_PATH)

    print("[3/3] Подбираем 4 фона и рендерим видео...")
    bg_dir = Path("assets/backgrounds")
    bgs = list(bg_dir.glob("*.mp4"))
    if not bgs:
        bgs = list(Path("assets").glob("*.mp4"))
    if not bgs:
        print("[ОШИБКА] Нет видеофайлов .mp4 в assets/backgrounds/!")
        return

    if len(bgs) >= 4:
        bg_list = random.sample(bgs, 4)
    else:
        bg_list = (bgs * 4)[:4]

    start_offsets = [0.0] * len(bg_list)

    print(f"Выбраны фоны: {[b.name for b in bg_list]}")
    render_video(
        backgrounds=bg_list,
        audio=AUDIO_PATH,
        subtitles=SUBTITLES_PATH,
        output=OUTPUT_VIDEO,
        duration=duration,
        start_offsets=start_offsets,
        settings=settings,
        target_clips=4
    )
    print(f"\n[ГОТОВО!] Видео успешно собрано: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
