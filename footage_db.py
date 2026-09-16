"""
FOOTAGE DATABASE — локальная база UFC-футажа для генерации видео.

Скачивает все Shorts с YouTube-канала (по умолчанию @UFCEurasia), тегирует
каждое видео через Gemini Vision (бойцы / экшен / ивент), считает текстовый
embedding (gemini-embedding-001) и раскладывает файлы по категориям:

    footage_db/
        fighters/islam_makhachev/…
        events/ufc_320/…
        actions/knockout/…
        misc/…
        _raw/          ← скачанные, но ещё не проиндексированные файлы
        index.json     ← полный индекс (метаданные + embeddings)

Каждый клип в индексе:
    {
      "id": "dQw4w9WgXcQ",
      "file": "fighters/alex_pereira/alex_pereira_knockout_dQw4w9WgXcQ.mp4",
      "fighter": ["Alex Pereira"],
      "action": ["knockout", "left hook"],
      "event": "UFC 295",
      "timestamp": 0.0,
      "duration": 34.2,
      "title": "…",
      "description": "…",
      "upload_date": "20250101",
      "embedding": [...]
    }

CLI:
    python footage_db.py download [--limit N] [--channel URL]   # выкачать Shorts
    python footage_db.py index    [--limit N]                   # тегировать + embeddings
    python footage_db.py search "Islam Makhachev walkout" [-k 5]
    python footage_db.py stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time

import aiohttp

import config
import gemini_keys

logger = logging.getLogger(__name__)

FOOTAGE_DIR = os.path.join(config.BASE_DIR, "footage_db")
RAW_DIR = os.path.join(FOOTAGE_DIR, "_raw")
INDEX_PATH = os.path.join(FOOTAGE_DIR, "index.json")
ARCHIVE_PATH = os.path.join(FOOTAGE_DIR, "_archive.txt")

DEFAULT_CHANNEL = "https://www.youtube.com/@UFCEurasia/shorts"

# Список каналов, по которым идёт скачивание/автоподкачка по умолчанию.
# Архив (_archive.txt) общий для всех каналов — одинаковый video id, встреченный на
# двух каналах (репост), скачивается только один раз. Переопределяется через
# FOOTAGE_CHANNELS (через запятую) или --channel в CLI.
CHANNELS: list[str] = [
    c.strip() for c in os.getenv("FOOTAGE_CHANNELS", "").split(",") if c.strip()
] or [
    DEFAULT_CHANNEL,
    "https://www.youtube.com/@ufc/shorts",
]

CATEGORY_DIRS = ("fighters", "events", "actions", "misc")

# Контролируемый словарь экшенов (Vision может добавлять свободные теги сверх него)
STANDARD_ACTIONS = [
    "knockout", "punch", "kick", "takedown", "submission", "grappling",
    "faceoff", "staredown", "walkout", "press_conference", "interview",
    "weigh_in", "training", "celebration", "crowd", "highlight",
]

EMBED_MODEL = os.getenv("FOOTAGE_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("FOOTAGE_EMBED_DIM", "768"))
# Минимальный итоговый score, при котором клип считается подходящим сцене
MIN_MATCH_SCORE = float(os.getenv("FOOTAGE_MIN_SCORE", "0.45"))

# Пейсинг Gemini-вызовов при индексации: free tier = 15 RPM на ключ.
# Держим 12 вызовов/мин (запас на ретраи), чтобы не ловить 429 и не
# крутить ротацию ключей вхолостую. Настраивается через FOOTAGE_INDEX_RPM.
INDEX_RPM = float(os.getenv("FOOTAGE_INDEX_RPM", "12"))
_MIN_CALL_INTERVAL = 60.0 / max(INDEX_RPM, 0.1)
_last_api_call = 0.0


async def _pace() -> None:
    """Выдержать паузу между Gemini-вызовами (лимит RPM бесплатного тира)."""
    global _last_api_call
    wait = _last_api_call + _MIN_CALL_INTERVAL - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_api_call = time.monotonic()

_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


def setup_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for d in CATEGORY_DIRS:
        os.makedirs(os.path.join(FOOTAGE_DIR, d), exist_ok=True)
    # Стандартные подпапки actions/ — по спецификации базы
    for a in ("knockout", "takedown", "submission", "faceoff", "walkout",
              "press_conference", "crowd", "celebration"):
        os.makedirs(os.path.join(FOOTAGE_DIR, "actions", a), exist_ok=True)


def slugify(text: str) -> str:
    """'Alex Pereira' → 'alex_pereira'; 'UFC 295' → 'ufc_295'."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s\-]+", "_", text).strip("_")
    return text[:60] or "unknown"


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index() -> list[dict]:
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_index(index: list[dict]) -> None:
    setup_dirs()
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    # Windows: os.replace падает с WinError 5, если index.json в этот момент
    # открыт другим процессом (проверка прогресса, антивирус, превью) —
    # ретраим с паузой вместо потери всего прогона индексации.
    for attempt in range(6):
        try:
            os.replace(tmp, INDEX_PATH)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.3 * (attempt + 1))


def index_size() -> int:
    return len(load_index())


def clip_abspath(record: dict) -> str:
    return os.path.join(FOOTAGE_DIR, record.get("file", ""))


# ---------------------------------------------------------------------------
# Gemini embeddings (REST, с ротацией ключей как в ai_adapter._gemini_chat)
# ---------------------------------------------------------------------------

class _QuotaExceeded(Exception):
    pass


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _embed_batch_once(texts: list[str], task_type: str, api_key: str) -> list[list[float]]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:batchEmbedContents"
    )
    body = {
        "requests": [
            {
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": t[:6000]}]},
                "taskType": task_type,
                "outputDimensionality": EMBED_DIM,
            }
            for t in texts
        ]
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 429:
                raise _QuotaExceeded((await resp.text())[:300])
            resp.raise_for_status()
            data = await resp.json()
    return [_normalize(e.get("values", [])) for e in data.get("embeddings", [])]


async def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embeddings для списка текстов (батчами по 32, с ротацией API-ключей)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        chunk = texts[i:i + 32]
        max_attempts = max(gemini_keys.key_count(), 1)
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            api_key = gemini_keys.get_current_key() or config.GEMINI_API_KEY
            try:
                out.extend(await _embed_batch_once(chunk, task_type, api_key))
                last_exc = None
                break
            except _QuotaExceeded as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    logger.warning("Embedding quota exceeded — rotating Gemini key")
                    gemini_keys.rotate_key(reason="embed 429")
        if last_exc is not None:
            raise RuntimeError(f"Embedding failed on all API keys: {last_exc}")
    return out


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


# ---------------------------------------------------------------------------
# Download: все Shorts канала одним вызовом yt-dlp (resumable через archive)
# ---------------------------------------------------------------------------

def download_shorts(
    channel_url: str = DEFAULT_CHANNEL,
    limit: int | None = None,
    new_only: bool = False,
    quiet: bool = False,
) -> int:
    """Выкачать Shorts канала в footage_db/_raw/ (+ .info.json на каждый).

    Повторный запуск докачивает только новые видео (--download-archive).

    ``new_only``: shorts-таб отдаёт видео от новых к старым, поэтому
    --lazy-playlist + --break-on-existing останавливает обход на ПЕРВОМ уже
    скачанном ролике — ежедневная подкачка занимает секунды вместо
    перечисления всего канала (28+ страниц плейлиста).

    Возвращает число новых файлов, появившихся в _raw/.
    """
    setup_dirs()
    import video_generator as vg  # cookie / extractor args — единая настройка с пайплайном

    before = {f for f in os.listdir(RAW_DIR) if f.endswith(_VIDEO_EXTS)}
    args = [
        "yt-dlp", channel_url,
        "--output", os.path.join(RAW_DIR, "%(id)s.%(ext)s"),
        "--write-info-json",
        "--download-archive", ARCHIVE_PATH,
        "--format", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--match-filters", "live_status=not_live",
        "--ignore-errors", "--no-warnings",
        "--sleep-requests", "1",
        *vg._YT_COOKIE_ARGS,
        *vg._YT_EXTRACTOR_ARGS,
    ]
    if new_only:
        args += ["--lazy-playlist", "--break-on-existing"]
    if limit:
        args += ["--max-downloads", str(limit)]
    logger.info(
        "Downloading shorts from %s → %s%s", channel_url, RAW_DIR,
        " (new only)" if new_only else "",
    )
    if quiet:
        # Фоновый режим (периодическая подкачка из бота): вывод в лог,
        # таймаут как страховка от зависшего yt-dlp.
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
            if r.returncode not in (0, 101):
                logger.warning("yt-dlp update rc=%d: %s", r.returncode, (r.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            logger.error("yt-dlp update timed out after 1h")
    else:
        # CLI-режим: без capture/timeout — прогресс виден в консоли,
        # полное скачивание канала может идти часами.
        subprocess.run(args)
    after = {f for f in os.listdir(RAW_DIR) if f.endswith(_VIDEO_EXTS)}
    new_files = len(after - before)
    logger.info("Shorts download done: +%d new, %d awaiting indexing", new_files, len(after))
    return new_files


_UPDATE_LOCK = asyncio.Lock()


async def update(
    channels: str | list[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Инкрементальное обновление базы по ВСЕМ каналам (по умолчанию — CHANNELS):
    докачать новые Shorts каждого канала + проиндексировать все новые файлы одним проходом.

    Безопасно вызывать из фоновой задачи бота (не блокирует event loop,
    повторный вызов при ещё идущем обновлении просто ждёт очереди).
    Архив общий для всех каналов, поэтому одно и то же видео, встреченное на двух
    каналах, скачивается и тегируется только один раз.
    Возвращает (скачано, проиндексировано).
    """
    if channels is None:
        chans = CHANNELS
    elif isinstance(channels, str):
        chans = [channels]
    else:
        chans = list(channels)

    async with _UPDATE_LOCK:
        downloaded = 0
        for ch in chans:
            downloaded += await asyncio.to_thread(
                download_shorts, ch, limit, True, True,
            )
        indexed = await index_videos()
        return downloaded, indexed


# ---------------------------------------------------------------------------
# Indexing: Vision-тегирование + embedding + раскладка по категориям
# ---------------------------------------------------------------------------

def _probe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def _load_info_json(video_path: str) -> dict:
    base = os.path.splitext(video_path)[0]
    for suffix in (".info.json",):
        p = base + suffix
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
    return {}


_TAG_PROMPT = """This frame is from a UFC/MMA YouTube Short.
Video title: "{title}"
Video description: "{description}"

Using BOTH the frame and the title/description, return STRICT JSON only (no markdown, no extra text):
{{
  "fighter": ["Full Name"],
  "action": ["knockout"],
  "event": "UFC 295",
  "description": "one short English sentence describing what is visually happening"
}}

Rules:
- "fighter": UFC fighters / persons shown or clearly referenced. Use canonical ENGLISH names in Latin script (e.g. "Islam Makhachev", not "Ислам Махачев"). Empty list if none identifiable.
- "action": 1-3 keywords, prefer these: {actions}. You may add one specific free-form tag (e.g. "left hook").
- "event": the UFC event (e.g. "UFC 320") if identifiable from title/description, else "".
"""


async def _tag_video(video_path: str, title: str, description: str) -> dict:
    """Gemini Vision по среднему кадру + заголовку/описанию → dict тегов."""
    import video_generator as vg  # lazy: тяжёлый импорт нужен только при индексации

    frame_path = video_path + ".frame.jpg"
    tags: dict = {}
    try:
        ok = await asyncio.to_thread(vg._extract_mid_frame, video_path, frame_path)
        if not ok:
            return {}
        with open(frame_path, "rb") as fh:
            image_bytes = fh.read()
        prompt = _TAG_PROMPT.format(
            title=(title or "")[:300],
            description=(description or "")[:500],
            actions=", ".join(STANDARD_ACTIONS),
        )
        await _pace()   # держимся под лимитом 15 RPM бесплатного тира
        text = await vg._gemini_vision_generate(prompt, image_bytes)
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            tags = json.loads(m.group(0))
    except Exception as exc:
        logger.warning("Vision tagging failed for %s: %s", os.path.basename(video_path), exc)
    finally:
        try:
            os.remove(frame_path)
        except OSError:
            pass
    if not isinstance(tags, dict):
        return {}
    return tags


def _target_relpath(tags: dict, vid: str) -> str:
    """Куда положить файл: fighters/<slug>/ → events/<slug>/ → actions/<a>/ → misc/."""
    fighters = [f for f in (tags.get("fighter") or []) if isinstance(f, str) and f.strip()]
    actions = [a for a in (tags.get("action") or []) if isinstance(a, str) and a.strip()]
    event = (tags.get("event") or "").strip()

    parts = []
    if fighters:
        parts.append(slugify(fighters[0]))
    if actions:
        parts.append(slugify(actions[0]))
    fname = "_".join(parts + [vid]) + ".mp4" if parts else f"{vid}.mp4"

    if fighters:
        return os.path.join("fighters", slugify(fighters[0]), fname)
    if event:
        return os.path.join("events", slugify(event), fname)
    if actions:
        return os.path.join("actions", slugify(actions[0]), fname)
    return os.path.join("misc", fname)


def _embed_text_for(record: dict) -> str:
    bits = (
        record.get("fighter", [])
        + record.get("action", [])
        + [record.get("event", ""), record.get("description", ""), record.get("title", "")]
    )
    return " | ".join(b for b in bits if b)


async def index_videos(limit: int | None = None) -> int:
    """Проиндексировать все файлы из _raw/: теги → embedding → категория → index.json.

    Инкрементально: индекс сохраняется после каждого видео (прерывание безопасно).
    Возвращает число добавленных записей.
    """
    setup_dirs()
    index = load_index()
    known_ids = {r.get("id") for r in index}

    # Дозаполнить embeddings у уже проиндексированных записей, где они
    # не посчитались из-за 429 (без embedding запись невидима для поиска).
    missing = [r for r in index if not r.get("embedding")]
    if missing:
        logger.info("Re-embedding %d records that lost embeddings to quota errors…", len(missing))
        for r in missing:
            try:
                embs = await embed_texts([_embed_text_for(r)], task_type="RETRIEVAL_DOCUMENT")
                r["embedding"] = embs[0] if embs else []
                save_index(index)
            except Exception as exc:
                logger.warning("Re-embedding failed for %s: %s", r.get("id"), exc)

    raw_files = sorted(
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(_VIDEO_EXTS) and not f.endswith(".part")
    )
    added = 0
    for path in raw_files:
        if limit and added >= limit:
            break
        # Файл мог быть обработан параллельным запуском (CLI vs фоновая подкачка)
        if not os.path.exists(path):
            continue
        vid = os.path.splitext(os.path.basename(path))[0]
        if vid in known_ids:
            continue
        info = _load_info_json(path)
        title = info.get("title", "")
        description = info.get("description", "")
        duration = _probe_duration(path)
        if duration < 1.0:
            logger.warning("Skipping broken video %s (duration %.1fs)", vid, duration)
            continue

        logger.info("Indexing %s — '%s' (%.0fs)", vid, title[:60], duration)
        tags = await _tag_video(path, title, description)
        fighters = [f.strip() for f in (tags.get("fighter") or []) if isinstance(f, str) and f.strip()]
        actions = [a.strip().lower() for a in (tags.get("action") or []) if isinstance(a, str) and a.strip()]
        event = (tags.get("event") or "").strip() if isinstance(tags.get("event"), str) else ""
        ai_desc = (tags.get("description") or "").strip() if isinstance(tags.get("description"), str) else ""

        record = {
            "id": vid,
            "file": "",
            "fighter": fighters,
            "action": actions,
            "event": event,
            "timestamp": 0.0,
            "duration": round(duration, 1),
            "title": title,
            "description": ai_desc,
            "upload_date": info.get("upload_date", ""),
            "embedding": [],
        }
        try:
            embs = await embed_texts([_embed_text_for(record)], task_type="RETRIEVAL_DOCUMENT")
            record["embedding"] = embs[0] if embs else []
        except Exception as exc:
            logger.warning("Embedding failed for %s (индексация продолжится без него): %s", vid, exc)

        # Переместить видео в категорию + сайдкар-метаданные рядом с файлом
        rel = _target_relpath(tags, vid)
        dest = os.path.join(FOOTAGE_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
        record["file"] = rel
        info_json = os.path.splitext(path)[0] + ".info.json"
        if os.path.exists(info_json):
            try:
                os.remove(info_json)
            except OSError:
                pass
        with open(os.path.splitext(dest)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)

        index.append(record)
        known_ids.add(vid)
        save_index(index)
        added += 1
        logger.info(
            "Indexed %s → %s | fighters=%s actions=%s event='%s'",
            vid, rel, fighters, actions, event,
        )
    logger.info("Indexing done: +%d records (total %d)", added, len(index))
    return added


# ---------------------------------------------------------------------------
# Search / scene matching
# ---------------------------------------------------------------------------

def _keyword_bonus(record: dict, subjects: list[str], actions: list[str]) -> float:
    bonus = 0.0
    rec_fighters = " ".join(record.get("fighter", [])).lower()
    for s in subjects:
        s = (s or "").strip().lower()
        if s and s in rec_fighters:
            bonus += 0.25
            break
    rec_actions = {a.lower() for a in record.get("action", [])}
    overlap = rec_actions & {a.strip().lower() for a in actions if a}
    bonus += min(0.16, 0.08 * len(overlap))
    return bonus


def _score_record(
    record: dict,
    q_emb: list[float],
    subjects: list[str],
    actions: list[str],
) -> float:
    emb = record.get("embedding") or []
    sim = cosine(q_emb, emb) if emb else 0.0
    return sim + _keyword_bonus(record, subjects, actions)


async def search_clips(query: str, top_k: int = 5) -> list[dict]:
    """Семантический поиск по базе. Возвращает [{score, path, record}, …]."""
    index = [r for r in load_index() if r.get("embedding")]
    if not index:
        return []
    q_emb = (await embed_texts([query], task_type="RETRIEVAL_QUERY"))[0]
    scored = [
        {"score": _score_record(r, q_emb, [], []), "path": clip_abspath(r), "record": r}
        for r in index
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


async def match_scenes(scenes: list[dict]) -> list[dict | None]:
    """Подбор клипа под каждую сцену planner'а.

    Каждая сцена: {"text", "subject": [...], "action": [...], "queries": [...]}.
    Возвращает список той же длины: {"path", "score", "record"} | None (если
    ничего не набрало MIN_MATCH_SCORE). Один клип не используется дважды,
    пока есть неиспользованные кандидаты.
    """
    index = [r for r in load_index() if r.get("embedding")]
    if not index or not scenes:
        return [None] * len(scenes)

    query_texts = []
    for sc in scenes:
        queries = [q for q in (sc.get("queries") or []) if q]
        base = "; ".join(queries) or sc.get("text", "")
        extras = " ".join((sc.get("subject") or []) + (sc.get("action") or []))
        query_texts.append(f"{base} {extras}".strip() or "UFC highlight")

    q_embs = await embed_texts(query_texts, task_type="RETRIEVAL_QUERY")

    used_files: set[str] = set()
    picks: list[dict | None] = []
    for sc, q_emb in zip(scenes, q_embs):
        subjects = sc.get("subject") or []
        actions = sc.get("action") or []
        best: dict | None = None
        best_score = -1.0
        for r in index:
            path = clip_abspath(r)
            if not os.path.exists(path):
                continue
            score = _score_record(r, q_emb, subjects, actions)
            if path in used_files:
                score -= 0.15  # мягкий штраф за повтор — не жёсткий запрет
            if score > best_score:
                best_score = score
                best = {"path": path, "score": round(score, 3), "record": r}
        if best is not None and best_score >= MIN_MATCH_SCORE:
            used_files.add(best["path"])
            picks.append(best)
        else:
            picks.append(None)
    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_stats() -> None:
    index = load_index()
    print(f"Indexed clips: {len(index)}")
    raw = [f for f in os.listdir(RAW_DIR) if f.endswith(_VIDEO_EXTS)] if os.path.isdir(RAW_DIR) else []
    print(f"Raw (not indexed): {len(raw)}")
    by_fighter: dict[str, int] = {}
    by_action: dict[str, int] = {}
    with_emb = 0
    for r in index:
        if r.get("embedding"):
            with_emb += 1
        for f in r.get("fighter", []):
            by_fighter[f] = by_fighter.get(f, 0) + 1
        for a in r.get("action", []):
            by_action[a] = by_action.get(a, 0) + 1
    print(f"With embeddings: {with_emb}")
    print("\nTop fighters:")
    for name, n in sorted(by_fighter.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:4d}  {name}")
    print("\nActions:")
    for name, n in sorted(by_action.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {name}")


def main() -> None:
    # Windows cp1251-консоль падает на эмодзи в названиях видео — заменяем безопасно
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="UFC footage database")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="download channel shorts into _raw/")
    p_dl.add_argument("--channel", action="append", default=None,
                       help="repeatable; defaults to all configured CHANNELS")
    p_dl.add_argument("--limit", type=int, default=None)

    p_ix = sub.add_parser("index", help="tag + embed raw videos, move into categories")
    p_ix.add_argument("--limit", type=int, default=None)

    p_up = sub.add_parser("update", help="fetch NEW shorts only (stops at first known) + index them")
    p_up.add_argument("--channel", action="append", default=None,
                       help="repeatable; defaults to all configured CHANNELS")
    p_up.add_argument("--limit", type=int, default=None)

    p_se = sub.add_parser("search", help="semantic search over the index")
    p_se.add_argument("query")
    p_se.add_argument("-k", type=int, default=5)

    sub.add_parser("stats", help="index statistics")

    args = ap.parse_args()
    if args.cmd == "download":
        for ch in (args.channel or CHANNELS):
            download_shorts(ch, args.limit)
    elif args.cmd == "index":
        asyncio.run(index_videos(args.limit))
    elif args.cmd == "update":
        downloaded, indexed = asyncio.run(update(args.channel, args.limit))
        print(f"Downloaded: {downloaded}, indexed: {indexed}, total: {index_size()}")
    elif args.cmd == "search":
        results = asyncio.run(search_clips(args.query, args.k))
        for res in results:
            r = res["record"]
            print(f"{res['score']:.3f}  {r['file']}")
            print(f"       fighters={r.get('fighter')} actions={r.get('action')} event='{r.get('event')}'")
            print(f"       {r.get('title', '')[:100]}")
    elif args.cmd == "stats":
        _cmd_stats()


if __name__ == "__main__":
    main()
