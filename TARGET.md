# Задача автономный генератор вертикальных Shorts из текста

## Цель

Нужно реализовать независимый Python CLI-инструмент, который принимает текст и автоматически создаёт готовый вертикальный `.mp4` для YouTube Shorts  TikTok  Reels.

Pipeline

```text
TEXT
 │
 ├── Edge TTS
 │      ├── audio.mp3
 │      └── word boundaries
 │
 ├── ASS subtitle generator
 │      └── subtitles.ass
 │
 └── FFmpeg
        ├── background gameplay
        ├── voice
        └── burned-in subtitles
              ↓
          output.mp4
```

На первом этапе НЕ нужны Telegram, YouTube API, база данных, веб-интерфейс или сложная архитектура.

Нужен рабочий локальный MVP.

---

# 1. Стек

Python 3.11+

Основные зависимости

 `edge-tts`
 стандартная библиотека Python
 FFmpeg  ffprobe как внешние executable

Не использовать

 Whisper
 MoviePy
 OpenCV
 тяжелые video-processing библиотеки

Основная обработка видео должна выполняться через FFmpeg subprocess.

---

# 2. Интерфейс CLI

Минимальный сценарий

```bash
python main.py Текст истории, который нужно озвучить и превратить в Shorts.
```

Также желательно поддержать

```bash
python main.py --text ...
python main.py --input story.txt
python main.py --output output.mp4
```

Пример

```bash
python main.py --text Однажды разработчик решил автоматизировать создание Shorts...
```

Результат

```text
output
    final_20260903_173000.mp4
```

При необходимости временные файлы можно сохранять в

```text
work
    audio.mp3
    subtitles.ass
```

Должна быть возможность автоматически очищать временные файлы после успешного рендера.

---

# 3. Структура проекта

Сделать примерно такую структуру

```text
shorts_generator
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src
│   ├── __init__.py
│   ├── tts.py
│   ├── subtitles.py
│   ├── video.py
│   ├── pipeline.py
│   └── config.py
│
├── assets
│   ├── background.mp4
│   └── fonts
│
├── output
└── work
```

Не надо создавать 25 абстрактных классов ради трёх функций.

Архитектура должна оставаться простой.

---

# 4. TTS

Использовать `edge-tts`.

По умолчанию

```python
VOICE = ru-RU-DmitryNeural
RATE = +10%
```

Предусмотреть конфигурацию голоса.

Например

```bash
python main.py --voice ru-RU-DmitryNeural
```

или через `config.py`.

TTS-функция должна

1. отправить текст в Edge TTS;
2. сохранить аудио;
3. получить WordBoundary;
4. преобразовать timestamps из ticks в секунды.

Важно

```text
1 tick = 100 nanoseconds = 1e-7 seconds
```

Пример структуры

```python
{
    word ...,
    start 1.23,
    end 1.57
}
```

Использовать streaming API

```python
communicate.stream()
```

Не использовать Whisper для получения таймингов.

---

# 5. Обработка WordBoundary

Edge TTS может возвращать неидеальные данные.

Нужно аккуратно обработать

 пустые слова;
 странные символы;
 пунктуацию;
 отсутствие WordBoundary;
 неожиданные timestamps.

Если WordBoundary отсутствуют полностью, pipeline должен завершиться понятной ошибкой, а не молча создать кривой ролик.

Логи должны показывать

```text
[TTS] Generating speech...
[TTS] Voice ru-RU-DmitryNeural
[TTS] Audio saved ...
[TTS] Word boundaries 37
```

---

# 6. Динамические субтитры

Использовать ASS.

Размер canvas

```text
PlayResX 1080
PlayResY 1920
```

Стиль

 крупный жирный шрифт;
 жёлтый  белый основной текст;
 толстая чёрная обводка;
 центр экрана;
 хорошая читаемость на мобильном.

Например

```text
Fontname Montserrat
Fontsize 80
Bold -1
Outline 8
Alignment 5
```

Но код НЕ должен сломаться, если Montserrat отсутствует.

Нужно предусмотреть конфигурацию шрифта и понятную проверку.

Если используется конкретный `.ttf`, лучше использовать локальный файл из

```text
assetsfonts
```

При необходимости добавить fallback на системный шрифт.

---

# 7. Группировка слов

По умолчанию показывать

```text
2 слова
```

на один subtitle event.

Например

```text
Однажды разработчик
решил автоматизировать
создание коротких
видео
```

Каждая группа должна использовать реальные timestamps

```text
start = first_word.start
end = last_word.end
```

Не использовать фиксированное время для каждой группы.

Добавить параметр

```python
WORDS_PER_SUBTITLE = 2
```

чтобы впоследствии можно было переключить на 3.

---

# 8. ASS timestamps

Реализовать корректное преобразование

```text
seconds
    ↓
HMMSS.cc
```

ASS использует centiseconds.

Например

```text
1.234
↓
00001.23
```

Учитывать корректное округление  обрезку сотых секунды.

---

# 9. FFmpeg

FFmpeg должен выполнять основной video pipeline.

Исходное видео

```text
assetsbackground.mp4
```

Это горизонтальный gameplay-фон, например GTA  Minecraft  parkour.

Предполагается, что оно достаточно длинное.

Pipeline

```text
background.mp4
       ↓
random start
       ↓
crop 916
       ↓
scale 1080x1920
       ↓
burn subtitles
       ↓
output.mp4
```

Audio

```text
audio.mp3
       ↓
AAC
       ↓
output.mp4
```

---

# 10. Crop

Для горизонтального видео сделать центральный crop 916.

Принцип

```text
crop=ih916ih
```

После этого

```text
scale=10801920
```

Но желательно сделать это через корректный FFmpeg filter chain.

Итоговый video stream обязан быть

```text
1080x1920
```

---

# 11. FPS

Добавить настройку

```python
FPS = 30
```

Поддержать

```bash
--fps 30
--fps 60
```

И использовать соответствующий FFmpeg filteroutput settings.

По умолчанию 30 FPS.

---

# 12. Длительность

Длительность результата должна соответствовать длительности озвучки.

Получить duration через

```bash
ffprobe
```

Например

```python
get_audio_duration(audio_path)
```

Если background короче необходимой длительности, не допускать зависания или создания чёрного видео.

Лучше на первом этапе явно проверить

```text
background duration = audio duration
```

и завершиться понятной ошибкой

```text
Background video is too short.
Required 37.4 sec
Available 21.8 sec
```

Позже можно добавить loop.

---

# 13. Random background segment

Если background длиннее audio, выбирать случайный старт.

Например

```python
max_start = background_duration - audio_duration
start_time = random.uniform(0, max_start)
```

НЕ использовать жёстко

```python
random.uniform(10, 500)
```

потому что это ломается на коротком фоне.

Добавить seed для воспроизводимости

```bash
--seed 123
```

Если seed не задан, использовать случайность.

---

# 14. FFmpeg command

Ориентировочная команда

```bash
ffmpeg -y
-ss START
-t DURATION
-i background.mp4
-i audio.mp3
-filter_complex [0v]crop=ih916ih,scale=10801920,subtitles=SUBTITLES[v]
-map [v]
-map 1a
-cv libx264
-preset veryfast
-crf 22
-r 30
-ca aac
-ba 192k
-shortest
output.mp4
```

Но не копировать её слепо.

Нужно корректно реализовать

 Windows paths;
 escaping paths для FFmpeg filters;
 пробелы в именах файлов;
 Unicode paths;
 ASS path;
 subprocess без shell=True.

Особенно важно `subtitles=...` является FFmpeg filter syntax, поэтому обычное quoting Python path недостаточно.

Нужно реализовать безопасное преобразование пути для FFmpeg subtitle filter.

---

# 15. Windows

Основная целевая среда

```text
Windows 1011
```

Код должен работать без Bash.

Не использовать Unix-only команды.

Проверять наличие

```text
ffmpeg
ffprobe
```

При старте

```text
[CHECK] ffmpeg OK
[CHECK] ffprobe OK
```

Если executable отсутствует

```text
ERROR FFmpeg was not found.

Install FFmpeg and add ffmpeg.exe  ffprobe.exe to PATH.
```

Не выдавать stack trace пользователю при обычной ошибке конфигурации.

---

# 16. FFmpeg subprocess

Использовать

```python
subprocess.run(
    cmd,
    check=True,
    ...
)
```

Не использовать

```python
shell=True
```

При ошибке FFmpeg сохранить stderr или вывести последние строки ошибки.

Например

```text
[VIDEO] Rendering...
[VIDEO] FFmpeg failed.

useful stderr
```

---

# 17. Pipeline class

Сделать простой orchestration layer

```python
class ShortsPipeline
    async def run(self, text str, output_path str) - Path
        ...
```

Логика

```text
validate environment
       ↓
validate text
       ↓
generate TTS
       ↓
generate ASS
       ↓
validate background
       ↓
render FFmpeg
       ↓
validate output
       ↓
return output path
```

Pipeline не должен содержать низкоуровневую реализацию TTSASSFFmpeg.

---

# 18. Config

Сделать конфигурацию через dataclass или простой config module.

Например

```python
VOICE = ru-RU-DmitryNeural
RATE = +10%
WORDS_PER_SUBTITLE = 2
FPS = 30
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
CRF = 22
PRESET = veryfast
AUDIO_BITRATE = 192k
```

CLI arguments должны иметь возможность переопределять defaults.

---

# 19. Валидация результата

После FFmpeg проверить

1. файл существует;
2. размер файла  0;
3. FFprobe показывает

    width = 1080
    height = 1920
    expected FPS
    audio stream exists;
    duration  0.

В конце

```text
================================
SUCCESS
================================

Output
outputfinal_20260903_173000.mp4

Resolution 1080x1920
FPS 30
Duration 37.42 sec
Audio AAC
Subtitles burned-in
================================
```

---

# 20. Error handling

Ошибки должны быть понятными.

Обработать минимум

 пустой текст;
 слишком короткий текст;
 Edge TTS failure;
 отсутствие WordBoundary;
 ffmpeg отсутствует;
 ffprobe отсутствует;
 background.mp4 отсутствует;
 background слишком короткий;
 FFmpeg render failure;
 subtitle generation failure;
 output validation failure.

Не скрывать реальные ошибки.

---

# 21. Logging

Сделать простой readable logging.

Пример

```text
[15] Validating environment...
[25] Generating voice...
[35] Generating subtitles...
[45] Rendering video...
[55] Validating output...

SUCCESS
```

Не нужен сложный logging framework.

---

# 22. Requirements

Создать

```text
requirements.txt
```

Минимум

```text
edge-tts
```

FFmpeg НЕ добавлять в requirements.txt, поскольку это внешний executable.

README должен объяснять установку FFmpeg в Windows.

---

# 23. README

README должен содержать

### Installation

```bash
python -m venv .venv
.venvScriptsactivate
pip install -r requirements.txt
```

### FFmpeg

Как проверить

```bash
ffmpeg -version
ffprobe -version
```

### Background

Куда положить

```text
assetsbackground.mp4
```

### Run

```bash
python main.py Тестовый текст для генерации короткого видео.
```

### Result

Где искать готовый MP4.

---

# 24. Tests

Добавить хотя бы unit tests для

 `format_time_ass()`;
 группировки слов;
 random start calculation;
 FFmpeg subtitle path escaping;
 конфигурации.

Не надо пытаться тестировать сам Edge TTS через интернет в каждом unit test.

---

# 25. Важное ограничение MVP

НЕ реализовывать сейчас

 Telegram;
 YouTube API;
 TikTok API;
 планировщик публикаций;
 генерацию историй;
 LLM;
 базы данных;
 Docker;
 Celery;
 Redis;
 web UI;
 multiprocessing;
 микросервисную архитектуру.

Это будет следующий этап.

Сейчас задача одна

```text
STRING
  ↓
VOICE
  ↓
WORD TIMINGS
  ↓
ASS
  ↓
BACKGROUND VIDEO
  ↓
FFMPEG
  ↓
1080x1920 MP4
```

---

# 26. Definition of Done

Считать задачу выполненной только если на чистой Windows-среде можно сделать

```bash
python main.py Это тестовая история для проверки генератора коротких видео.
```

и получить

```text
output.mp4
```

который

 открывается обычным видеоплеером;
 имеет 1080x1920;
 имеет 30 FPS;
 содержит голос;
 содержит динамические субтитры;
 субтитры уже burned-in в видео;
 длительность соответствует озвучке;
 фон выбран из `assetsbackground.mp4`;
 используется случайный сегмент background;
 Python не держит видео целиком в RAM;
 обработка выполняется потоково через FFmpeg;
 временные файлы можно удалить после завершения.

---

# 27. Порядок работы

Сначала изучи существующее состояние репозитория.

Затем

1. предложи краткий план реализации;
2. создай структуру проекта;
3. реализуй код;
4. установи зависимости;
5. запусти unit tests;
6. если в окружении есть FFmpeg и background.mp4, выполни реальный end-to-end тест;
7. исправь найденные ошибки;
8. обнови README.

Не останавливайся после написания кода.

Мне нужен именно рабочий MVP, а не набор красиво выглядящих функций.

При реализации приоритет

```text
correctness

reliability

simplicity

performance

abstraction
```

Не добавляй архитектуру, которая не нужна текущей задаче.
Если какое-то техническое решение неоднозначно, выбирай наиболее простой production-приемлемый вариант и кратко объясни его в финальном отчёте.
