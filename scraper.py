import asyncio
import logging
import os
import re
import hashlib
import ssl
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import certifi
from bs4 import BeautifulSoup

from config import IMAGES_DIR, VIDEOS_DIR, NEWS_URL, BLOCKED_URL_CATEGORIES

# Use certifi CA bundle; falls back to no-verify if still failing on macOS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
}


@dataclass
class Article:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    local_images: list[str] = field(default_factory=list)
    pg_embeds: list[dict] = field(default_factory=list)  # {"type": "youtube"|"playground", "id": str}


async def fetch(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30), ssl=SSL_CONTEXT) as resp:
            if resp.status == 200:
                return await resp.text()
            logger.warning("HTTP %s — %s", resp.status, url)
    except Exception as exc:
        logger.error("Ошибка загрузки страницы [%s]: %s", url, exc)
    return None


async def get_latest_article_links(session: aiohttp.ClientSession) -> list[dict]:
    """Scrape the news listing page and return list of {url, title}."""
    # Cookie pg_post_sorting=creation_date forces "newest first" sort order
    try:
        async with session.get(
            NEWS_URL,
            headers=HEADERS,
            cookies={"pg_post_sorting": "%7B%22news%22%3A%22creation_date%22%7D"},
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=SSL_CONTEXT,
        ) as resp:
            html = await resp.text() if resp.status == 200 else None
    except Exception as exc:
        logger.error("Ошибка загрузки страницы [%s]: %s", NEWS_URL, exc)
        html = None
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles = []
    seen_urls: set[str] = set()

    # playground.ru wraps each article in div.post; title is in div.post-title > a
    for post in soup.select("div.post"):
        title_tag = post.select_one("div.post-title a") or post.select_one("a[href*='/news/']")
        if not title_tag:
            continue
        href = title_tag.get("href", "")
        if not href:
            continue
        full_url = urljoin("https://www.playground.ru", href)
        # Accept URLs with /news/ anywhere in path, skip category-only pages
        if "/news/" not in full_url or full_url.rstrip("/") in (
            "https://www.playground.ru/news",
            "https://www.playground.ru/news/industry",
            "https://www.playground.ru/news/movies",
            "https://www.playground.ru/news/trailers",
            "https://www.playground.ru/news/pc",
            "https://www.playground.ru/news/updates",
            "https://www.playground.ru/news/hardware",
            "https://www.playground.ru/news/rumors",
            "https://www.playground.ru/news/consoles",
        ):
            continue
        if full_url in seen_urls:
            continue
        title = title_tag.get_text(strip=True)

        # --- Filter by URL category ---
        if any(cat in full_url for cat in BLOCKED_URL_CATEGORIES):
            logger.debug("Пропускаем (категория): %s", full_url)
            continue

        seen_urls.add(full_url)
        articles.append({"url": full_url, "title": title})

    logger.info("Найдено статей на странице: %d", len(articles))
    return articles


async def scrape_article(session: aiohttp.ClientSession, url: str) -> Optional[Article]:
    """Scrape full article content and image URLs."""
    html = await fetch(session, url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Body — use playground.ru specific classes first, then fallbacks
    body = (
        soup.select_one("div.post-content")
        or soup.select_one("div.article-content")
        or soup.select_one("article .content")
        or soup.select_one("article")
        or soup.select_one("main")
    )

    if body:
        # Remove scripts, styles, nav, ads
        for tag in body.select("script, style, nav, aside, .ad, .advertisement, .share, .comments"):
            tag.decompose()
        paragraphs = [p.get_text(separator=" ", strip=True) for p in body.find_all(["p", "li", "blockquote"])]
        text = "\n".join(p for p in paragraphs if len(p) > 20)
    else:
        text = soup.get_text(separator="\n", strip=True)[:3000]

    # Remove comments sections from the entire page before extracting images
    for tag in soup.select(
        ".comments, #comments, .comment, .comment-list, "
        ".comment-block, .comments-section, .disqus_thread, "
        "[id*='comment'], [class*='comment']"
    ):
        tag.decompose()

    # og:image — most reliable source for the main article image
    og_image = soup.find("meta", property="og:image")
    og_image_url: str | None = og_image["content"].strip() if og_image and og_image.get("content") else None

    # Images — only from article body, not the whole page
    article_scope = (
        soup.select_one("div.post-content")
        or soup.select_one("div.article-content")
        or soup.select_one("article")
        or soup
    )
    img_tags = article_scope.select("img")
    image_urls: list[str] = []
    seen_imgs: set[str] = set()

    def _img_dedup_key(img_url: str) -> str:
        """Return a normalised key for deduplication.
        For i.playground.ru the same image may appear under /e/ and /p/ paths
        but shares the same filename hash, so we key on filename only."""
        base = img_url.split("?")[0]
        filename = base.rsplit("/", 1)[-1]
        if "i.playground.ru" in base:
            return filename
        return base

    # Seed with og:image first so it appears at position 0
    if og_image_url and _is_valid_image_url(og_image_url):
        norm = _img_dedup_key(og_image_url)
        seen_imgs.add(norm)
        image_urls.append(og_image_url)

    _dim_re = re.compile(r'^\d+x\d+$')

    for img in img_tags:
        # Always prefer img src/data-src over parent <a> href
        # (href usually points to a gallery page, not a direct image)
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src", "")
        if src:
            full = urljoin(url, src)
            # Strip ?NxM resize query params to get full-resolution image
            # e.g. https://i.playground.ru/p/xyz.png?255x255 → .../xyz.png
            if "?" in full:
                base, qs = full.split("?", 1)
                if _dim_re.match(qs):
                    full = base
        else:
            # Fall back to parent <a> href only if it looks like an image
            parent_a = img.find_parent("a")
            if parent_a and parent_a.get("href"):
                full = urljoin(url, parent_a.get("href", ""))
            else:
                continue
        norm = _img_dedup_key(full)  # dedup by filename for i.playground.ru
        if norm not in seen_imgs and _is_valid_image_url(full):
            seen_imgs.add(norm)
            image_urls.append(full)

    # Also collect standalone <a href="...jpg/png/webp"> with no <img> inside
    for a in article_scope.select("a[href]"):
        if a.find("img"):
            continue
        href = a.get("href", "")
        full = urljoin(url, href)
        norm = _img_dedup_key(full)
        if norm not in seen_imgs and _is_valid_image_url(full):
            seen_imgs.add(norm)
            image_urls.append(full)

    # Extract pg-embed video tags (YouTube and playground internal videos)
    pg_embeds = _extract_pg_embeds(article_scope)

    return Article(url=url, title=title, text=text, image_urls=image_urls, pg_embeds=pg_embeds)


# ---------------------------------------------------------------------------
# Video extraction and download
# ---------------------------------------------------------------------------

def _extract_pg_embeds(scope) -> list[dict]:
    """Find all <pg-embed> tags and return list of {type, id}."""
    embeds = []
    for tag in scope.find_all("pg-embed"):
        embed_type = tag.get("type", "")
        embed_id = tag.get("src", "").strip()
        if embed_type in ("youtube", "playground", "vk") and embed_id:
            embeds.append({"type": embed_type, "id": embed_id})
    return embeds


async def _fetch_playground_m3u8(session: aiohttp.ClientSession, video_id: str) -> Optional[str]:
    """Fetch the iframe page for a playground video and return the best m3u8 URL ≤1080p."""
    iframe_url = f"https://www.playground.ru/video/iframe/{video_id}/"
    html = await fetch(session, iframe_url)
    if not html:
        return None
    matches = re.findall(r'file:\s*"(https://video\.playground\.ru/[^"]+\.m3u8)"', html)
    if not matches:
        return None

    # Each master playlist is ~200 bytes and contains a RESOLUTION tag — fetch all in parallel
    async def _get_height(url: str) -> tuple[int, str]:
        content = await fetch(session, url)
        if content:
            m = re.search(r'RESOLUTION=\d+x(\d+)', content)
            if m:
                return int(m.group(1)), url
        return 0, url

    results = await asyncio.gather(*(_get_height(u) for u in matches))

    # Best quality that fits within 1080p
    candidates = [(h, u) for h, u in results if 0 < h <= 1080]
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    # All streams exceed 1080p — fall back to lowest bitrate number
    def _bitrate_key(url: str) -> int:
        m = re.search(r'm-(\d+)', url)
        return int(m.group(1)) if m else 0
    return min(matches, key=_bitrate_key)


async def download_videos(
    session: aiohttp.ClientSession,
    pg_embeds: list[dict],
) -> tuple[list[str], list[str]]:
    """Download videos from pg-embed tags.

    Returns:
        youtube_urls  — list of non-YouTube external URLs (e.g. VK) to append to post text
        video_paths   — list of local mp4 file paths (playground + YouTube videos)
    """
    youtube_urls: list[str] = []
    video_paths: list[str] = []

    for embed in pg_embeds:
        if embed["type"] == "youtube":
            path = await _download_youtube_video(embed["id"])
            if path:
                video_paths.append(path)
            else:
                logger.warning("Не удалось скачать YouTube видео: %s", embed["id"])

        elif embed["type"] == "vk":
            # src is a query string like "oid=-231353027&id=456239049"
            params = dict(p.split("=") for p in embed["id"].split("&") if "=" in p)
            oid = params.get("oid", "")
            vid = params.get("id", "")
            if oid and vid:
                vk_url = f"https://vk.com/video{oid}_{vid}"
                path = await _download_vk_video(oid, vid)
                if path:
                    video_paths.append(path)
                else:
                    logger.warning("Не удалось скачать VK видео: %s", vk_url)
                    youtube_urls.append(vk_url)

        elif embed["type"] == "playground":
            for attempt in range(1, 4):
                m3u8_url = await _fetch_playground_m3u8(session, embed["id"])
                if not m3u8_url:
                    logger.warning("Не удалось получить m3u8 для видео %s (попытка %d/3)", embed["id"], attempt)
                    if attempt < 3:
                        await asyncio.sleep(2.0 * attempt)
                    continue
                path = await _download_hls_video(embed["id"], m3u8_url)
                if path:
                    video_paths.append(path)
                    logger.info("Видео скачано: %s", path)
                    break
                logger.warning("Не удалось скачать видео %s (попытка %d/3)", embed["id"], attempt)
                if attempt < 3:
                    await asyncio.sleep(2.0 * attempt)

    return youtube_urls, video_paths


async def _download_youtube_video(video_id: str) -> Optional[str]:
    """Download a YouTube video by ID using yt-dlp. Returns local .mp4 path or None."""
    output_path = os.path.join(VIDEOS_DIR, f"yt_{video_id}.mp4")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
        return output_path  # already downloaded
    url = f"https://www.youtube.com/watch?v={video_id}"
    env = os.environ.copy()
    extra_paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        os.path.expanduser("~/.nvm/versions/node/v20.19.5/bin"),
    ]
    env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--cookies-from-browser", "chrome",
            "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]/best",
            "--max-filesize", "1500M",
            "--merge-output-format", "mp4",
            "--output", output_path,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode not in (0, 101):
            logger.warning(
                "yt-dlp error downloading %s (rc=%d): %s",
                video_id, proc.returncode, stderr.decode()[-400:],
            )
            return None
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            logger.info("YouTube видео скачано: %s", output_path)
            return output_path
        # yt-dlp may have chosen a different extension
        for ext in (".webm", ".mkv"):
            alt = os.path.join(VIDEOS_DIR, f"yt_{video_id}{ext}")
            if os.path.exists(alt) and os.path.getsize(alt) > 10240:
                return alt
        logger.warning("YouTube видео не найдено после скачивания: %s", video_id)
        return None
    except asyncio.TimeoutError:
        logger.error("yt-dlp timeout при скачивании YouTube видео: %s", video_id)
        return None
    except Exception as exc:
        logger.error("Ошибка скачивания YouTube видео %s: %s", video_id, exc)
        return None


async def _download_vk_video(oid: str, vid: str) -> Optional[str]:
    """Download a VK video using yt-dlp. Returns local .mp4 path or None."""
    file_id = f"vk_{oid}_{vid}".replace("-", "m")
    output_path = os.path.join(VIDEOS_DIR, f"{file_id}.mp4")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
        return output_path  # already downloaded
    url = f"https://vk.com/video{oid}_{vid}"
    env = os.environ.copy()
    extra_paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        os.path.expanduser("~/.nvm/versions/node/v20.19.5/bin"),
    ]
    env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]/best",
            "--max-filesize", "1500M",
            "--merge-output-format", "mp4",
            "--output", output_path,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode not in (0, 101):
            logger.warning(
                "yt-dlp error downloading VK %s (rc=%d): %s",
                url, proc.returncode, stderr.decode()[-400:],
            )
            return None
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            logger.info("VK видео скачано: %s", output_path)
            return output_path
        for ext in (".webm", ".mkv"):
            alt = os.path.join(VIDEOS_DIR, f"{file_id}{ext}")
            if os.path.exists(alt) and os.path.getsize(alt) > 10240:
                return alt
        logger.warning("VK видео не найдено после скачивания: %s", url)
        return None
    except asyncio.TimeoutError:
        logger.error("yt-dlp timeout при скачивании VK видео: %s", url)
        return None
    except Exception as exc:
        logger.error("Ошибка скачивания VK видео %s: %s", url, exc)
        return None


async def _download_hls_video(video_id: str, m3u8_url: str) -> Optional[str]:
    """Download HLS stream to mp4 using ffmpeg. Returns local path or None."""
    output_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    # Skip cache if file is corrupt (moov atom missing) — probe it first
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
        probe = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", output_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await probe.communicate()
        # Valid only if duration returned AND no decode errors in stderr
        if stdout.strip() and not stderr.strip():
            return output_path  # valid cached file
        logger.warning("Кэшированный файл %s повреждён, перескачиваем", output_path)
        os.remove(output_path)
    async def _run_ffmpeg(*args: str, timeout: int) -> bool:
        nonlocal proc
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", *args,
            "-movflags", "faststart",
            "-y", output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode == 0 and os.path.exists(output_path)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return False

    proc = None
    try:
        # Try stream copy first (fast, works when codec is already H264/AAC)
        ok = await _run_ffmpeg(
            "-i", m3u8_url,
            "-c", "copy",
            timeout=300,
        )
        if not ok:
            # Corrupt or incompatible codec — re-encode
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.info("Копирование не удалось, перекодируем %s", video_id)
            ok = await _run_ffmpeg(
                "-i", m3u8_url,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-vf", "scale=iw*sar:ih,setsar=1",
                "-c:a", "aac",
                timeout=600,
            )
        if ok:
            return output_path
        logger.warning("Таймаут/ошибка при скачивании видео %s", video_id)
    except Exception as exc:
        logger.warning("Ошибка скачивания видео %s: %s", video_id, exc)
    if os.path.exists(output_path):
        os.remove(output_path)
    return None


def _is_valid_image_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    # Skip tiny icons, logos, avatars
    skip_patterns = ["logo", "icon", "avatar", "banner", "sprite", "pixel"]
    if any(p in path for p in skip_patterns):
        return False
    return bool(re.search(r"\.(jpg|jpeg|png|webp|gif)(\?|$)", path))


async def download_images(session: aiohttp.ClientSession, image_urls: list[str]) -> list[str]:
    """Download images and return local file paths (max 10)."""
    paths: list[str] = []
    # Limit to 2 concurrent downloads to avoid rate-limiting from i.playground.ru
    sem = asyncio.Semaphore(2)

    async def _guarded(url):
        async with sem:
            return await _download_image(session, url)

    tasks = [_guarded(url) for url in image_urls[:10]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, str):
            paths.append(r)
    return paths


async def _download_image(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    name = hashlib.md5(url.encode()).hexdigest()
    for attempt in range(1, 4):
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=160), ssl=SSL_CONTEXT) as resp:
                if resp.status != 200:
                    logger.warning("Картинка HTTP %s [%s] (попытка %d/3)", resp.status, url, attempt)
                    break  # non-retriable
                content_type = resp.headers.get("content-type", "")
                ext = _ext_from_content_type(content_type) or _ext_from_url(url) or "jpg"
                data = await resp.read()
                if len(data) < 2048:  # skip tiny images
                    return None
                path = os.path.join(IMAGES_DIR, f"{name}.{ext}")
                with open(path, "wb") as f:
                    f.write(data)
                return path
        except Exception as exc:
            logger.warning("Не удалось скачать картинку [%s] (попытка %d/3): %s", url, attempt, exc)
            if attempt < 3:
                await asyncio.sleep(3.0 * attempt)
    return None


def _ext_from_content_type(ct: str) -> Optional[str]:
    mapping = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}
    for k, v in mapping.items():
        if k in ct:
            return v
    return None


def _ext_from_url(url: str) -> Optional[str]:
    m = re.search(r"\.(jpg|jpeg|png|webp|gif)", url.lower())
    return m.group(1).replace("jpeg", "jpg") if m else None
