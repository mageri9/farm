from pathlib import Path
import os
import time
from functools import lru_cache
from PIL import Image
from rembg import remove, new_session

@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return new_session(sess_opts=opts)

src_dir = Path("assets/items")
dst_dir = Path("assets/items/фламберг")
dst_dir.mkdir(parents=True, exist_ok=True)

# Берем картинки строго из корня items (без захода в подпапки типа "клевец")
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

if not files:
    print("В папке assets/items не найдено файлов картинок!")
else:
    print(f"Найдено картинок: {len(files)}. Начинаем обработку...")
    session = _session()
    for idx, f in enumerate(sorted(files), start=1):
        out_path = dst_dir / f"{idx}.png"
        print(f"[{idx}/{len(files)}] Вырезаем фон: {f.name} -> фламберг/{idx}.png")
        with Image.open(f) as img:
            t = time.perf_counter()
            img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
            resize_time = time.perf_counter() - t
            t = time.perf_counter()
            result = remove(img, session=session)
            inference_time = time.perf_counter() - t
            t = time.perf_counter()
            result.save(out_path, "PNG")
            save_time = time.perf_counter() - t
        print(f"[CUTOUT] {f.name} -> {out_path.name}: resize={resize_time:.2f}s, inference={inference_time:.2f}s, save={save_time:.2f}s", flush=True)
    print("\nУспешно! Все прозрачные PNG сохранены в assets/items/фламберг/")


