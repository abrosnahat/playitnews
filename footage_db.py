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
import difflib
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
    "https://www.youtube.com/@PFLMMA/shorts",
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
    # Уникальное имя tmp-файла НА ПРОЦЕСС — без этого параллельный CLI-запуск
    # и фоновая автоподкачка бота (main.py, отдельный процесс) писали в один
    # и тот же index.json.tmp и могли удалить/перезаписать файл друг друга
    # между open() и os.replace() (гонка → FileNotFoundError/WinError 2).
    tmp = f"{INDEX_PATH}.{os.getpid()}.tmp"
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
# Межпроцессная блокировка index.json
#
# CLI-запуск (`footage_db.py index/update`) и фоновая автоподкачка бота
# (main.py, ОТДЕЛЬНЫЙ процесс со своим event loop) могут оба вызвать
# index_videos() одновременно. Внутрипроцессный asyncio.Lock тут НЕ помогает —
# это две РАЗНЫХ программы, каждая со своим снимком `index` в памяти —
# последний записавший os.replace() тихо затирает записи, добавленные
# другим процессом в тот же промежуток. Файловый мьютекс (эксклюзивное
# создание файла) сериализует доступ к индексации между ЛЮБЫМи процессами.
# ---------------------------------------------------------------------------

_LOCK_PATH = os.path.join(FOOTAGE_DIR, "_index.lock")
# Если лок держится дольше этого времени — считаем владельца умершим
# (crash без уборки за собой) и снимаем лок принудительно.
_LOCK_STALE_SECONDS = 6 * 3600


class _CrossProcessLock:
    """Файловый мьютекс на index.json (между ПРОЦЕССАМи, не потоками)."""

    def __init__(self, path: str):
        self.path = path
        self._held = False

    async def acquire(self) -> None:
        setup_dirs()
        warned = False
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._held = True
                return
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.path)
                except OSError:
                    age = 0.0
                if age > _LOCK_STALE_SECONDS:
                    logger.warning("Footage DB lock is stale (%.0fs old) — removing", age)
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                if not warned:
                    logger.info("Footage DB index is busy (another process is indexing) — waiting…")
                    warned = True
                await asyncio.sleep(2.0)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            os.remove(self.path)
        except OSError:
            pass

    async def __aenter__(self) -> "_CrossProcessLock":
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.release()


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
        # Cookies и player_client (tv_downgraded) НЕ передаём: публичные Shorts в
        # них не нуждаются, а они дают "The page needs to be reloaded" и молча
        # блокируют скачивание новых роликов (--ignore-errors). Дефолтный клиент
        # yt-dlp (android vr) работает стабильно.
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
        # таймаут как страховка от зависшего yt-dlp. Явная utf-8 кодировка —
        # yt-dlp пишет эмодзи/кириллицу, Windows иначе декодирует как cp1251
        # и падает с UnicodeDecodeError.
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=3600,
                encoding="utf-8", errors="replace",
            )
            if r.returncode not in (0, 101):
                logger.warning("yt-dlp update rc=%d: %s", r.returncode, (r.stderr or "")[-400:])
            elif "ERROR:" in (r.stderr or ""):
                logger.warning("yt-dlp update errors: %s", (r.stderr or "")[-400:])
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


# ---------------------------------------------------------------------------
# Защита от ошибочной индексации: текстовые факты (событие, имена из названия)
# важнее догадок Vision-модели; имена приводятся к каноническим; подозрительные
# записи помечаются needs_review (см. audit).
# ---------------------------------------------------------------------------

_UFC_EVENT_RE = re.compile(r"\bUFC\s*[-#:]?\s*(\d{2,3})\b", re.IGNORECASE)
_MONTAGE_FRACTIONS = (0.2, 0.5, 0.8)
_FUZZY_NAME_CUTOFF = 0.88


def _event_from_text(*texts: str) -> str:
    """'UFC 333' из названия/описания (в т.ч. хэштег #ufc333). Берёт первый текст,
    где назван РОВНО один номер; если номеров несколько (анонс нескольких турниров) —
    текст пропускается как неоднозначный."""
    for t in texts:
        nums: list[str] = []
        for m in _UFC_EVENT_RE.finditer(t or ""):
            if m.group(1) not in nums:
                nums.append(m.group(1))
        if len(nums) == 1:
            return f"UFC {nums[0]}"
    return ""


def _event_in_text(event: str, text: str) -> bool:
    """Подтверждено ли событие текстом (номер турнира или название целиком)."""
    event = (event or "").strip()
    if not event:
        return False
    nums = re.findall(r"\d{2,3}", event)
    if nums:
        return all(re.search(rf"(?<!\d){n}(?!\d)", text or "") for n in nums)
    # Без пробелов: "UFC London" должно находиться в хэштеге #UFCLondon
    compact = lambda s: re.sub(r"[\s#_-]+", "", s.lower())  # noqa: E731
    return compact(event) in compact(text or "")


class _Roster:
    """Известные имена бойцов из индекса: канонизация написаний и поиск имён в тексте."""

    def __init__(self, index: list[dict] | None = None) -> None:
        self.counts: dict[str, int] = {}
        self._by_lower: dict[str, str] = {}
        self._last: dict[str, set[str]] = {}
        for r in index or []:
            self.add(r.get("fighter", []))

    def add(self, names: list[str]) -> None:
        for n in names or []:
            n = (n or "").strip()
            if not n:
                continue
            self.counts[n] = self.counts.get(n, 0) + 1
            self._by_lower.setdefault(n.lower(), n)
            parts = n.split()
            if len(parts) >= 2 and len(parts[-1]) >= 4:
                self._last.setdefault(parts[-1].lower(), set()).add(n)

    def canonicalize(self, name: str) -> str:
        """Привести имя к уже известному написанию (точное совпадение без регистра,
        иначе близкое по написанию к достаточно частому имени)."""
        name = (name or "").strip()
        hit = self._by_lower.get(name.lower())
        if hit:
            return hit
        pool = [n for n, c in self.counts.items() if c >= 2]
        m = difflib.get_close_matches(name, pool, n=1, cutoff=_FUZZY_NAME_CUTOFF)
        return m[0] if m else name

    def _resolve_last(self, last: str) -> str | None:
        names = self._last.get(last)
        if not names:
            return None
        if len(names) == 1:
            return next(iter(names))
        # Несколько написаний одной фамилии: берём доминирующее (≥2× частоты второго)
        ranked = sorted(names, key=lambda n: -self.counts.get(n, 0))
        if self.counts[ranked[0]] >= 2 * self.counts[ranked[1]]:
            return ranked[0]
        return None

    def find_in_text(self, text: str) -> list[str]:
        """Известные бойцы, названные в тексте (полное имя или однозначная фамилия)."""
        t = (text or "").lower()
        if not t.strip():
            return []
        found: list[str] = []
        for lower, canon in self._by_lower.items():
            if " " in lower and len(lower) >= 6 and lower in t and canon not in found:
                found.append(canon)
        words = set(re.findall(r"[\w'’-]+", t))
        for last in self._last:
            if last in words:
                canon = self._resolve_last(last)
                if canon and canon not in found:
                    found.append(canon)
        return found


def _postprocess_tags(tags: dict, title: str, description: str, roster: _Roster) -> dict:
    """Привести ответ Vision к проверенному виду: каноничные имена бойцов и
    событие ТОЛЬКО из текста (модель выдумывает его по лицам/контексту)."""
    out = dict(tags)
    fighters: list[str] = []
    for f in tags.get("fighter") or []:
        if isinstance(f, str) and f.strip():
            c = roster.canonicalize(f)
            if c not in fighters:
                fighters.append(c)
    out["fighter"] = fighters

    model_event = tags.get("event") if isinstance(tags.get("event"), str) else ""
    model_event = model_event.strip()
    text = f"{title} {description}"
    text_event = _event_from_text(title, description)
    if text_event:
        event = text_event
    elif model_event and _event_in_text(model_event, text):
        event = model_event
    else:
        event = ""
    if model_event and model_event != event:
        logger.info("Event from Vision %r not backed by text → using %r", model_event, event)
    out["event"] = event
    return out


def _review_reasons(record: dict, roster: _Roster, text: str | None = None) -> list[str]:
    """Почему запись выглядит подозрительной (пусто = всё сходится).
    ``text`` — название (+описание YouTube, если доступно при индексации)."""
    title = record.get("title", "") or ""
    text = text if text is not None else title
    reasons: list[str] = []

    ev = (record.get("event") or "").strip()
    text_event = _event_from_text(title) or _event_from_text(text)
    if text_event and ev != text_event:
        reasons.append(f"event {ev or '-'} != text {text_event}")
    elif ev and not _event_in_text(ev, text):
        reasons.append(f"event {ev!r} not stated in text")

    named = roster.find_in_text(title)
    have = {f.lower() for f in record.get("fighter", [])}
    if named and not any(n.lower() in have for n in named):
        reasons.append(f"title names {named} but fighters={record.get('fighter', [])}")
    return reasons


def _apply_review(record: dict, roster: _Roster, text: str | None = None) -> list[str]:
    reasons = _review_reasons(record, roster, text)
    if reasons:
        record["needs_review"] = reasons
        logger.warning("Needs review %s: %s", record.get("id"), "; ".join(reasons))
    else:
        record.pop("needs_review", None)
    return reasons


def _write_sidecar(record: dict) -> None:
    dest = clip_abspath(record)
    if os.path.exists(dest):
        with open(os.path.splitext(dest)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)


def _extract_montage(video_path: str, out_path: str) -> int:
    """Кадры на 20/50/80% длительности → один JPEG (горизонтальная склейка).
    Один вызов Vision вместо трёх. Возвращает число панелей (0 = неудача);
    для очень коротких/проблемных клипов откатывается на один средний кадр."""
    import video_generator as vg

    duration = vg._get_audio_duration(video_path)
    if duration >= 1.5:
        frames: list[str] = []
        try:
            for i, frac in enumerate(_MONTAGE_FRACTIONS):
                fp = f"{out_path}.{i}.jpg"
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{duration * frac:.2f}", "-i", video_path,
                     "-frames:v", "1", "-vf", "scale=360:-2", fp],
                    capture_output=True, text=True, timeout=20,
                )
                if proc.returncode == 0 and os.path.exists(fp) and os.path.getsize(fp) > 0:
                    frames.append(fp)
            if len(frames) >= 2:
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
                for fp in frames:
                    cmd += ["-i", fp]
                cmd += ["-filter_complex", f"hstack=inputs={len(frames)}", out_path]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return len(frames)
        except Exception as exc:
            logger.debug("Montage extraction failed for %s: %s", video_path, exc)
        finally:
            for fp in frames:
                try:
                    os.remove(fp)
                except OSError:
                    pass
    return 1 if vg._extract_mid_frame(video_path, out_path) else 0


_TAG_PROMPT = """This image is {panels} from one UFC/MMA YouTube Short{order}.
Video title: "{title}"
Video description: "{description}"
Upload date: {upload_date}
Candidate fighters named in the text: {candidates}
Event stated in the text: {event_hint}

Using the image AND the title/description, return STRICT JSON only (no markdown, no extra text):
{{
  "fighter": ["Full Name"],
  "action": ["knockout"],
  "event": "UFC 295",
  "description": "one short English sentence describing what is visually happening"
}}

Rules:
- "fighter": ONLY people you can actually SEE and visually recognize. Prefer names from the candidate list when they match what is shown; add any other name only if you are confident about it. The title/description often names fighters from a different fight, a reaction video, an interview about someone else, or a compilation — do NOT list a fighter just because a name appears in the text if that person is not the one shown. Use canonical ENGLISH names in Latin script (e.g. "Islam Makhachev", not "Ислам Махачев"). Empty list if no identifiable fighter is visible.
- "action": 1-3 keywords, prefer these: {actions}. You may add one specific free-form tag (e.g. "left hook").
- "event": the UFC event ONLY if the title/description explicitly state it (e.g. "UFC 320" or a hashtag like #ufc333). NEVER infer it from the faces or from who is fighting; if the text does not state it, return "". Clips are usually uploaded within days of their event, so use the upload date only to sanity-check, not to guess a number.
"""


async def _tag_video(
    video_path: str,
    title: str,
    description: str,
    upload_date: str = "",
    roster: _Roster | None = None,
) -> dict:
    """Gemini Vision по кадрам (монтаж 20/50/80%) + заголовку/описанию → dict тегов.
    Событие берётся из текста, имена канонизируются по ростеру индекса."""
    import video_generator as vg  # lazy: тяжёлый импорт нужен только при индексации

    roster = roster if roster is not None else _Roster()
    frame_path = video_path + ".frame.jpg"
    tags: dict = {}
    try:
        panels = await asyncio.to_thread(_extract_montage, video_path, frame_path)
        if not panels:
            return {}
        with open(frame_path, "rb") as fh:
            image_bytes = fh.read()

        title_names = roster.find_in_text(title)
        desc_names = [n for n in roster.find_in_text(description) if n not in title_names]
        candidates = (title_names + desc_names)[:6]
        prompt = _TAG_PROMPT.format(
            panels=f"a montage of {panels} frames" if panels > 1 else "a single frame",
            order=" (left to right = time order)" if panels > 1 else "",
            title=(title or "")[:300],
            description=(description or "")[:500],
            upload_date=upload_date or "unknown",
            candidates=", ".join(candidates) or "none",
            event_hint=_event_from_text(title, description) or "none",
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
    if not isinstance(tags, dict) or not tags:
        return {}
    return _postprocess_tags(tags, title, description, roster)


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
    Защищён межпроцессным локом (_CrossProcessLock) — когда бот в фоне делает
    свою автоподкачку, ручной CLI-запуск просто ждёт очереди вместо того
    чтобы оба одновременно перезаписывали index.json и теряли записи друг друга.
    Возвращает число добавленных записей.
    """
    setup_dirs()
    async with _CrossProcessLock(_LOCK_PATH):
        return await _index_videos_locked(limit)


async def _index_videos_locked(limit: int | None = None) -> int:
    index = load_index()
    known_ids = {r.get("id") for r in index}
    roster = _Roster(index)

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
        tags = await _tag_video(
            path, title, description,
            upload_date=info.get("upload_date", ""), roster=roster,
        )
        fighters = [f.strip() for f in (tags.get("fighter") or []) if isinstance(f, str) and f.strip()]
        actions = [a.strip().lower() for a in (tags.get("action") or []) if isinstance(a, str) and a.strip()]
        event = (tags.get("event") or "").strip() if isinstance(tags.get("event"), str) else ""
        if not tags:
            # Vision не ответил — событие всё равно можно взять из текста
            event = _event_from_text(title, description)
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
        _apply_review(record, roster, text=f"{title} {description}")
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
        roster.add(fighters)
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

# Penalty for a clip whose tagged fighter(s) are all UNRELATED to the current
# story — without this, a scene with no per-scene subject (LLM missed it, or
# a generic hype line like "this is where champions are made") is matched by
# raw embedding similarity alone, which can hand-pick a real, correctly-tagged
# clip of a totally different, semantically-similar-sounding fighter (e.g. a
# Gaethje title celebration for an Adesanya/Tsarukyan/Ruffy story). Only
# applies when we actually know who the story is about (``story_subjects``
# non-empty) and the candidate has an explicit named fighter tag that matches
# none of them — generic/untagged action footage is never penalized.
_OFF_TOPIC_PENALTY = float(os.getenv("FOOTAGE_OFF_TOPIC_PENALTY", "0.35"))
# Смягчённый штраф: клип с нужным бойцом, но ещё и с ПОСТОРОННИМ именным бойцом из другой
# истории (напр. Ян vs Фейбер 2018 в сюжете про Ян — Двалишвили). Применяется только
# когда в сюжете названо ≥2 бойцов (матчап), и не трогает клипы, где все теги — из сюжета.
_FOREIGN_COFIGHTER_PENALTY = float(os.getenv("FOOTAGE_FOREIGN_COFIGHTER_PENALTY", "0.3"))
# Бонус клипу, где есть ≥2 бойцов из сцены вместе (сам матчап: Ян + Двалишвили), а не
# один боец из старого боя с кем-то другим.
_MATCHUP_BONUS = float(os.getenv("FOOTAGE_MATCHUP_BONUS", "0.2"))


def _keyword_bonus(record: dict, subjects: list[str], actions: list[str]) -> float:
    bonus = 0.0
    rec_fighters = " ".join(record.get("fighter", [])).lower()
    subj = [(s or "").strip().lower() for s in subjects]
    matched = [s for s in subj if s and s in rec_fighters]
    if matched:
        bonus += 0.25
    if len(matched) >= 2:
        bonus += _MATCHUP_BONUS
    rec_actions = {a.lower() for a in record.get("action", [])}
    overlap = rec_actions & {a.strip().lower() for a in actions if a}
    bonus += min(0.16, 0.08 * len(overlap))
    return bonus


def _off_topic_penalty(
    record: dict,
    story_subjects: frozenset[str],
    scene_subjects: frozenset[str] = frozenset(),
) -> float:
    if not story_subjects:
        return 0.0
    rec_fighters = [f.strip().lower() for f in record.get("fighter", []) if f and f.strip()]
    if not rec_fighters:
        return 0.0

    def _in(rf: str, subs: frozenset[str]) -> bool:
        return any(s and (s in rf or rf in s) for s in subs)

    if not any(_in(rf, story_subjects) for rf in rec_fighters):
        return _OFF_TOPIC_PENALTY
    # Для «постороннего» со-бойца смотрим на бойцов СЦЕНЫ (если их ≥2), иначе — всего сюжета:
    # иначе боец из другой сцены того же сюжета (О'Мэлли в финале) проходит в сцену про бой Ян — Мераб.
    ref = scene_subjects if len(scene_subjects) >= 2 else story_subjects
    if len(ref) >= 2 and not all(_in(rf, ref) for rf in rec_fighters):
        return _FOREIGN_COFIGHTER_PENALTY
    return 0.0


def _score_record(
    record: dict,
    q_emb: list[float],
    subjects: list[str],
    actions: list[str],
    story_subjects: frozenset[str] = frozenset(),
) -> float:
    emb = record.get("embedding") or []
    sim = cosine(q_emb, emb) if emb else 0.0
    scene_subjects = frozenset(
        s.strip().lower() for s in subjects if isinstance(s, str) and s.strip()
    )
    return (
        sim
        + _keyword_bonus(record, subjects, actions)
        - _off_topic_penalty(record, story_subjects, scene_subjects)
    )


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

    # Every fighter named ANYWHERE in the scene plan — used to penalize clips
    # tagged with a named fighter who isn't part of this story at all (see
    # _off_topic_penalty). Empty when the plan never named anyone (heuristic
    # fallback) — penalty is a no-op in that case, same as before this change.
    story_subjects = frozenset(
        s.strip().lower()
        for sc in scenes
        for s in (sc.get("subject") or [])
        if isinstance(s, str) and s.strip()
    )

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
            score = _score_record(r, q_emb, subjects, actions, story_subjects)
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
# Maintenance — fix a single mis-indexed clip (wrong fighter/action/event tag).
#
# Root cause of most mis-tags: indexing tags off ONE mid-frame + the video's
# title/description, and title text can bias Gemini into naming a fighter who
# isn't actually shown (reaction/compilation/interview videos, multi-fighter
# previews, etc.). These helpers let a bad entry be corrected without having
# to re-run the whole indexing pass.
# ---------------------------------------------------------------------------

def _locate(index: list[dict], video_id: str) -> dict | None:
    """Find a record WITHIN an already-loaded index list, by its YouTube id
    (exact) or by a filename fragment (e.g. pasted straight from a
    "[FOOTAGE] scene N ... -> <file>" log line, with or without .mp4)."""
    video_id = (video_id or "").strip()
    stem = os.path.splitext(os.path.basename(video_id))[0]
    for r in index:
        if r.get("id") == stem or r.get("id") == video_id:
            return r
    for r in index:
        fname = os.path.splitext(os.path.basename(r.get("file", "")))[0]
        if stem and (fname == stem or stem in fname):
            return r
    return None


def find_record(video_id: str) -> dict | None:
    """Locate a record by its YouTube id or a filename fragment (loads a
    fresh copy of the index — use this for a one-off lookup/CLI ``info``)."""
    return _locate(load_index(), video_id)


def remove_video(video_id: str) -> bool:
    """Delete a mis-indexed clip entirely: removes the video file, its sidecar
    .json, and its entry from index.json. Use for clips that are simply wrong
    and not worth keeping."""
    index = load_index()
    match = _locate(index, video_id)
    if match is None:
        logger.warning("remove_video: no record found for %r", video_id)
        return False
    path = clip_abspath(match)
    for p in (path, os.path.splitext(path)[0] + ".json"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as exc:
            logger.warning("remove_video: could not delete %s: %s", p, exc)
    index = [r for r in index if r is not match]
    save_index(index)
    logger.info("Removed clip %s (%s) from footage DB", match.get("id"), match.get("file"))
    return True


def _move_record_file(record: dict) -> None:
    """Recompute the category path for *record* (after its tags changed) and
    move the underlying file + sidecar .json there if it moved categories."""
    old_path = clip_abspath(record)
    new_rel = _target_relpath(record, record["id"])
    new_path = os.path.join(FOOTAGE_DIR, new_rel)
    if os.path.abspath(new_path) == os.path.abspath(old_path):
        return
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    old_sidecar = os.path.splitext(old_path)[0] + ".json"
    new_sidecar = os.path.splitext(new_path)[0] + ".json"
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
    if os.path.exists(old_sidecar):
        shutil.move(old_sidecar, new_sidecar)
    record["file"] = new_rel


async def relabel_video(
    video_id: str,
    fighter: list[str] | None = None,
    action: list[str] | None = None,
    event: str | None = None,
    description: str | None = None,
    clear_fighter: bool = False,
    clear_action: bool = False,
    clear_description: bool = False,
) -> bool:
    """Manually overwrite a clip's tags (when a human already knows the
    correct fighter/action/event) — moves the file to the right category
    folder and re-computes its embedding so search/scene-matching picks it up
    correctly going forward."""
    index = load_index()
    record = _locate(index, video_id)
    if record is None:
        logger.warning("relabel_video: no record found for %r", video_id)
        return False
    if clear_fighter:
        record["fighter"] = []
    elif fighter is not None:
        record["fighter"] = [f.strip() for f in fighter if f.strip()]
    if clear_action:
        record["action"] = []
    elif action is not None:
        record["action"] = [a.strip().lower() for a in action if a.strip()]
    if event is not None:
        record["event"] = event.strip()
    # The (often AI-hallucinated) free-text description feeds the semantic
    # embedding too — leaving a wrong name in there (e.g. "Arman Tsarukyan is
    # lying on the canvas...") would keep biasing cosine-similarity search
    # towards that fighter even after the structured fighter/action tags are
    # fixed, so it must be clearable/settable just like the other fields.
    if clear_description:
        record["description"] = ""
    elif description is not None:
        record["description"] = description.strip()
    # Ручная правка = запись проверена человеком
    record.pop("needs_review", None)

    embs = await embed_texts([_embed_text_for(record)], task_type="RETRIEVAL_DOCUMENT")
    record["embedding"] = embs[0] if embs else record.get("embedding", [])
    _move_record_file(record)
    dest = clip_abspath(record)
    if os.path.exists(dest):
        with open(os.path.splitext(dest)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
    save_index(index)
    logger.info(
        "Relabeled %s → fighters=%s actions=%s event=%r (%s)",
        record["id"], record.get("fighter"), record.get("action"), record.get("event"), record["file"],
    )
    return True


async def _retag_record(record: dict, roster: _Roster | None = None) -> bool:
    """Re-run Gemini Vision tagging on *record*'s CURRENT file and overwrite
    its fighter/action/event/description/embedding in place, then move the
    file to the (possibly new) category folder. Does NOT touch index.json —
    callers own the load/save so this can be used both for a single clip
    (``retag_video``) and a bulk pass (``reindex_all``) with one save per
    clip or one save at the end, as appropriate."""
    path = clip_abspath(record)
    if not os.path.exists(path):
        logger.warning("_retag_record: file missing on disk: %s", path)
        return False

    # Описание записи — это уже вывод модели (может содержать галлюцинации), поэтому как
    # текстовый источник используется только название ролика.
    roster = roster if roster is not None else _Roster(load_index())
    tags = await _tag_video(
        path, record.get("title", ""), "",
        upload_date=record.get("upload_date", ""), roster=roster,
    )
    if not tags:
        logger.warning("_retag_record: Vision tagging returned nothing for %s", record["id"])
        return False
    record["fighter"] = [f.strip() for f in (tags.get("fighter") or []) if isinstance(f, str) and f.strip()]
    record["action"] = [a.strip().lower() for a in (tags.get("action") or []) if isinstance(a, str) and a.strip()]
    record["event"] = (tags.get("event") or "").strip() if isinstance(tags.get("event"), str) else record.get("event", "")
    ai_desc = (tags.get("description") or "").strip() if isinstance(tags.get("description"), str) else ""
    if ai_desc:
        record["description"] = ai_desc
    _apply_review(record, roster)

    # A transient embedding-API error (e.g. 503) must NOT lose the freshly
    # corrected tags above, and (critically for reindex_all's multi-hour
    # unattended run) must not raise — same fallback-and-keep-going pattern
    # already used by _index_videos_locked's own re-embedding pass.
    try:
        embs = await embed_texts([_embed_text_for(record)], task_type="RETRIEVAL_DOCUMENT")
        record["embedding"] = embs[0] if embs else record.get("embedding", [])
    except Exception as exc:
        logger.warning("_retag_record: re-embedding failed for %s (tags still updated): %s", record["id"], exc)
    _move_record_file(record)
    dest = clip_abspath(record)
    if os.path.exists(dest):
        with open(os.path.splitext(dest)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
    return True


async def retag_video(video_id: str) -> bool:
    """Re-run Gemini Vision tagging on an ALREADY-indexed clip (fresh mid-frame
    + the stricter anti-title-bias prompt) and overwrite its tags/embedding.
    Useful when a clip was mis-tagged by an earlier indexing pass — a retry
    with the current prompt often fixes it since Vision calls aren't fully
    deterministic. For a clip you can already identify by eye, ``relabel_video``
    (no extra API call, no guessing) is more reliable."""
    index = load_index()
    record = _locate(index, video_id)
    if record is None:
        logger.warning("retag_video: no record found for %r", video_id)
        return False
    ok = await _retag_record(record)
    if not ok:
        return False
    save_index(index)
    logger.info(
        "Retagged %s → fighters=%s actions=%s event=%r (%s)",
        record["id"], record.get("fighter"), record.get("action"), record.get("event"), record["file"],
    )
    return True


async def reindex_all(
    limit: int | None = None,
    only_fighter: str | None = None,
    offset: int = 0,
) -> tuple[int, int]:
    """Re-run Gemini Vision tagging on EVERY already-indexed clip (not just new
    arrivals in _raw/) — use after tightening the tagging prompt, or when you
    suspect widespread mis-tags and want a bulk pass instead of fixing clips
    one at a time. Rate-limited via the same ``_pace()`` as normal indexing
    (~1 clip every few seconds — a full pass over ~1000+ clips can take HOURS,
    run it in the background). Safe to interrupt: index.json is saved after
    EVERY clip, and re-running just continues (nothing is skipped based on
    prior state, so re-running twice just re-tags everything again).

    ``only_fighter``: optional case-insensitive substring filter — only
    re-tag clips whose CURRENT fighter list contains a match (handy for a
    narrower, cheaper pass instead of the whole DB).

    ``offset``: skip this many candidates before starting — lets a huge index
    be worked through in batches (e.g. ``offset=1000, limit=1000`` to reindex
    the NEXT 1000 clips after an earlier ``limit=1000`` pass) instead of
    always re-doing the same clips from the start.

    Returns (attempted, changed) — ``changed`` counts clips whose fighter
    list actually differs after the retag (worth eyeballing in the log).
    """
    async with _CrossProcessLock(_LOCK_PATH):
        index = load_index()
        roster = _Roster(index)
        candidates = index
        if only_fighter:
            needle = only_fighter.strip().lower()
            candidates = [r for r in index if any(needle in f.lower() for f in r.get("fighter", []))]
        if offset:
            candidates = candidates[offset:]
        if limit:
            candidates = candidates[:limit]
        total = len(candidates)
        attempted = 0
        changed = 0
        for i, record in enumerate(candidates, 1):
            old_fighters = list(record.get("fighter", []))
            vid = record.get("id")
            logger.info("Reindexing %d/%d (offset %d): %s (was %s)", i, total, offset, vid, old_fighters)
            try:
                ok = await _retag_record(record, roster)
            except Exception as exc:
                # Must never abort the whole (potentially hours-long,
                # unattended) bulk pass over one bad clip/transient API error
                # — log it and move on, index already has the old tags intact.
                logger.warning("Reindexing %s failed, keeping old tags: %s", vid, exc)
                ok = False
            attempted += 1
            if not ok:
                save_index(index)
                continue
            if record.get("fighter", []) != old_fighters:
                changed += 1
                logger.info("  → changed: %s → %s", old_fighters, record.get("fighter"))
            save_index(index)
        logger.info("Reindex-all done: %d attempted, %d changed fighter tags", attempted, changed)
        return attempted, changed


async def audit_index(
    fix_events: bool = False,
    mark: bool = False,
) -> tuple[list[tuple[dict, list[str]]], int]:
    """Найти подозрительно размеченные клипы без обращения к Vision (только текст
    названия против тегов): событие не совпадает с названием / не подтверждено
    текстом, боец из названия отсутствует в тегах.

    ``fix_events``: исправить событие там, где название однозначно называет
    турнир (#ufc333 → "UFC 333"); пересчитывает embedding этих записей.
    ``mark``: записать причины в ``needs_review`` каждой записи (и убрать флаг
    у тех, что стали корректными).
    Возвращает ([(запись, причины)], число_исправленных_событий).
    """
    async with _CrossProcessLock(_LOCK_PATH):
        index = load_index()
        roster = _Roster(index)
        fixed = 0
        if fix_events:
            for r in index:
                text_event = _event_from_text(r.get("title", ""))
                if not text_event or (r.get("event") or "").strip() == text_event:
                    continue
                logger.info("Fix event %s: %r → %r", r.get("id"), r.get("event"), text_event)
                r["event"] = text_event
                try:
                    embs = await embed_texts([_embed_text_for(r)], task_type="RETRIEVAL_DOCUMENT")
                    r["embedding"] = embs[0] if embs else r.get("embedding", [])
                except Exception as exc:
                    logger.warning("Re-embedding failed for %s (event still fixed): %s", r.get("id"), exc)
                _move_record_file(r)
                _write_sidecar(r)
                fixed += 1
                if fixed % 25 == 0:
                    save_index(index)
        flagged: list[tuple[dict, list[str]]] = []
        for r in index:
            reasons = _review_reasons(r, roster)
            if reasons:
                flagged.append((r, reasons))
                if mark:
                    r["needs_review"] = reasons
            elif mark:
                r.pop("needs_review", None)
        if mark or fixed:
            save_index(index)
            if mark:
                for r, _ in flagged:
                    _write_sidecar(r)
        return flagged, fixed


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


def _cmd_info(video_id: str) -> None:
    r = find_record(video_id)
    if r is None:
        print(f"No record found for {video_id!r}")
        return
    print(f"id:       {r.get('id')}")
    print(f"file:     {r.get('file')}")
    print(f"fighters: {r.get('fighter')}")
    print(f"actions:  {r.get('action')}")
    print(f"event:    {r.get('event')!r}")
    print(f"title:    {r.get('title', '')[:120]}")
    print(f"desc:     {r.get('description', '')[:200]}")
    if r.get("needs_review"):
        print(f"REVIEW:   {'; '.join(r['needs_review'])}")


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

    p_info = sub.add_parser("info", help="show a clip's current tags (id or filename fragment)")
    p_info.add_argument("video_id")

    p_relabel = sub.add_parser("relabel", help="manually overwrite a clip's tags (fixes a mis-indexed clip)")
    p_relabel.add_argument("video_id")
    p_relabel.add_argument("--fighter", help="comma-separated correct fighter name(s)")
    p_relabel.add_argument("--action", help="comma-separated correct action tag(s)")
    p_relabel.add_argument("--event", help="correct event name, e.g. 'UFC 320'")
    p_relabel.add_argument("--description", help="correct free-text description (also feeds the embedding)")
    p_relabel.add_argument("--clear-fighter", action="store_true", help="wipe the fighter list (no fighter shown)")
    p_relabel.add_argument("--clear-action", action="store_true", help="wipe the action list")
    p_relabel.add_argument("--clear-description", action="store_true", help="wipe the free-text description")

    p_retag = sub.add_parser("retag", help="re-run Gemini Vision tagging on one already-indexed clip")
    p_retag.add_argument("video_id")

    p_remove = sub.add_parser("remove", help="delete a mis-indexed clip entirely (file + index entry)")
    p_remove.add_argument("video_id")

    p_reall = sub.add_parser(
        "reindex-all",
        help="re-run Vision tagging on EVERY already-indexed clip (slow, rate-limited — use --limit to test first)",
    )
    p_reall.add_argument("--limit", type=int, default=None, help="only re-tag N clips")
    p_reall.add_argument("--offset", type=int, default=0, help="skip this many clips before starting (batch through a huge index, e.g. --offset 1000 --limit 1000 for the next 1000)")
    p_reall.add_argument("--fighter", default=None, help="only re-tag clips currently tagged with this fighter (substring match)")

    p_audit = sub.add_parser(
        "audit",
        help="find suspicious tags by comparing titles with tags (no API calls unless --fix-events)",
    )
    p_audit.add_argument("--fix-events", action="store_true",
                         help="set event from the title where it names exactly one UFC number (re-embeds those clips)")
    p_audit.add_argument("--mark", action="store_true",
                         help="write reasons to needs_review in index.json / sidecars")
    p_audit.add_argument("--show", type=int, default=100, help="max flagged clips to print (default 100)")

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
    elif args.cmd == "info":
        _cmd_info(args.video_id)
    elif args.cmd == "relabel":
        ok = asyncio.run(relabel_video(
            args.video_id,
            fighter=args.fighter.split(",") if args.fighter else None,
            action=args.action.split(",") if args.action else None,
            event=args.event,
            description=args.description,
            clear_fighter=args.clear_fighter,
            clear_action=args.clear_action,
            clear_description=args.clear_description,
        ))
        print("OK" if ok else "FAILED — see log above")
    elif args.cmd == "retag":
        ok = asyncio.run(retag_video(args.video_id))
        print("OK" if ok else "FAILED — see log above")
    elif args.cmd == "remove":
        ok = remove_video(args.video_id)
        print("OK" if ok else "FAILED — see log above")
    elif args.cmd == "reindex-all":
        attempted, changed = asyncio.run(reindex_all(args.limit, args.fighter, args.offset))
        print(f"Attempted: {attempted}, changed fighter tags: {changed}")
    elif args.cmd == "audit":
        flagged, fixed = asyncio.run(audit_index(args.fix_events, args.mark))
        print(f"Events fixed: {fixed}")
        print(f"Flagged: {len(flagged)} of {index_size()}")
        for r, reasons in flagged[: args.show]:
            print(f"  {r.get('id')}  {r.get('title', '')[:60]!r}")
            for reason in reasons:
                print(f"      - {reason}")


if __name__ == "__main__":
    main()
