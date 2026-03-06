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

from config import IMAGES_DIR, VIDEOS_DIR, NEWS_URL

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
        seen_urls.add(full_url)
        title = title_tag.get_text(strip=True)
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
    for img in img_tags:
        # If this img is inside an <a href>, prefer the href (full-size) over src (thumbnail)
        parent_a = img.find_parent("a")
        if parent_a and parent_a.get("href"):
            href = parent_a.get("href", "")
            full = urljoin(url, href)
        else:
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src", "")
            if not src:
                continue
            full = urljoin(url, src)
        norm = full.split("?")[0]  # strip query params for dedup
        if norm not in seen_imgs and _is_valid_image_url(full):
            seen_imgs.add(norm)
            image_urls.append(full)

    # Also collect standalone <a href="...jpg/png/webp"> with no <img> inside
    for a in article_scope.select("a[href]"):
        if a.find("img"):
            continue
        href = a.get("href", "")
        full = urljoin(url, href)
        norm = full.split("?")[0]
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
        if embed_type in ("youtube", "playground") and embed_id:
            embeds.append({"type": embed_type, "id": embed_id})
    return embeds


async def _fetch_playground_m3u8(session: aiohttp.ClientSession, video_id: str) -> Optional[str]:
    """Fetch the iframe page for a playground video and return the lowest-quality m3u8 URL."""
    iframe_url = f"https://www.playground.ru/video/iframe/{video_id}/"
    html = await fetch(session, iframe_url)
    if not html:
        return None
    # Extract all m3u8 URLs — last one tends to be the lowest quality (360p)
    matches = re.findall(r'file:\s*"(https://video\.playground\.ru/[^"]+\.m3u8)"', html)
    if not matches:
        return None
    # Prefer 360p label if available, otherwise take the last (lowest bitrate)
    for m in matches:
        if "m-100" in m:  # 360p
            return m
    return matches[-1]


async def download_videos(
    session: aiohttp.ClientSession,
    pg_embeds: list[dict],
) -> tuple[list[str], list[str]]:
    """Download videos from pg-embed tags.

    Returns:
        youtube_urls  — list of YouTube watch URLs to append to post text
        video_paths   — list of local mp4 file paths (playground internal videos)
    """
    youtube_urls: list[str] = []
    video_paths: list[str] = []

    for embed in pg_embeds:
        if embed["type"] == "youtube":
            youtube_urls.append(f"https://www.youtube.com/watch?v={embed['id']}")

        elif embed["type"] == "playground":
            m3u8_url = await _fetch_playground_m3u8(session, embed["id"])
            if not m3u8_url:
                logger.warning("Не удалось получить m3u8 для видео %s", embed["id"])
                continue
            path = await _download_hls_video(embed["id"], m3u8_url)
            if path:
                video_paths.append(path)
                logger.info("Видео скачано: %s", path)
            else:
                logger.warning("Не удалось скачать видео %s", embed["id"])

    return youtube_urls, video_paths


async def _download_hls_video(video_id: str, m3u8_url: str) -> Optional[str]:
    """Download HLS stream to mp4 using ffmpeg. Returns local path or None."""
    output_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
        return output_path  # already downloaded
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", m3u8_url,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-vf", "scale=iw*sar:ih,setsar=1",
            "-c:a", "aac",
            "-movflags", "faststart",
            "-y", output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode == 0 and os.path.exists(output_path):
            return output_path
    except asyncio.TimeoutError:
        logger.warning("Таймаут при скачивании видео %s", video_id)
    except Exception as exc:
        logger.warning("Ошибка скачивания видео %s: %s", video_id, exc)
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
    tasks = [_download_image(session, url) for url in image_urls[:10]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, str):
            paths.append(r)
    return paths


async def _download_image(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30), ssl=SSL_CONTEXT) as resp:
            if resp.status != 200:
                return None
            content_type = resp.headers.get("content-type", "")
            ext = _ext_from_content_type(content_type) or _ext_from_url(url) or "jpg"
            data = await resp.read()
            if len(data) < 2048:  # skip tiny images
                return None
            name = hashlib.md5(url.encode()).hexdigest() + f".{ext}"
            path = os.path.join(IMAGES_DIR, name)
            with open(path, "wb") as f:
                f.write(data)
            return path
    except Exception as exc:
        logger.warning("Не удалось скачать картинку [%s]: %s", url, exc)
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
