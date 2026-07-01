"""
Main entry point.
Starts the Telegram bot and schedules the periodic news check.
"""
import asyncio
import logging
import sys
import warnings

# Silence harmless shutdown warning from python-telegram-bot on Ctrl+C:
# "coroutine 'Updater.stop' was never awaited" — internal PTB shutdown
# path occasionally drops the coroutine on macOS; bot still stops cleanly.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'Updater\.stop' was never awaited",
    category=RuntimeWarning,
)
from datetime import datetime, timezone

import aiohttp
import certifi
from telegram.ext import Application
from telegram.request import HTTPXRequest

import database as db
from ai_adapter import adapt_article, adapt_article_ru, is_gaming_related
from bot import build_handlers, error_handler, send_admin_notification
from config import (
    CHECK_INTERVAL_MINUTES,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_ADMIN_CHAT_ID,
    TELEGRAM_LOCAL_API_URL,
    TELEGRAM_LOCAL_API_FILE_URL,
    TELEGRAM_LOCAL_MODE,
    get_project,
    project_ai,
    project_names,
    setup_dirs,
)
from scraper import download_images, download_videos
from sources import get_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)-12s  %(message)s",
    datefmt="%d.%m %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("playitnews.log", encoding="utf-8"),
    ],
)
# Убираем шумные логгеры сторонних библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("telegram.ext._updater").setLevel(logging.CRITICAL)  # suppress disconnect noise
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline: check news → scrape → adapt → schedule
# ---------------------------------------------------------------------------

async def process_article(app: Application, article_url: str, article_title: str,
                          project_name: str = "gaming") -> None:
    """Full pipeline for a single new article."""
    logger.info("Обрабатываем статью [%s]: %s", project_name, article_url)

    source = get_source(get_project(project_name).get("source"))
    async with aiohttp.ClientSession() as session:
        article = await source.scrape_article(session, article_url)
        if not article:
            logger.warning("Не удалось спарсить статью: %s", article_url)
            return

        # Download images
        image_paths = await download_images(session, article.image_urls)
        logger.info("Скачано картинок: %d  [%s]", len(image_paths), article_url)

        # Download videos (playground HLS) and collect YouTube links
        youtube_urls, video_paths = await download_videos(session, article.pg_embeds)
        if video_paths:
            logger.info("Скачано видео: %d  [%s]", len(video_paths), article_url)
        if youtube_urls:
            logger.info("YouTube ссылок: %d  [%s]", len(youtube_urls), article_url)

    # Adapt via AI (English for main channel, Russian for second channel)
    post_text, ru_post_text = await asyncio.gather(
        adapt_article(article.title, article.text, prompt=project_ai(project_name, "post_text_en")),
        adapt_article_ru(article.title, article.text, prompt=project_ai(project_name, "post_text_ru")),
    )

    # Schedule the post (no delay — admin approves to publish immediately)
    scheduled_at = datetime.now(timezone.utc)
    post_id = db.create_scheduled_post(
        article_url=article_url,
        article_title=article.title,
        post_text=post_text,
        image_paths=image_paths,
        scheduled_at=scheduled_at,
        video_paths=video_paths,
        ru_post_text=ru_post_text,
        project=project_name,
    )
    db.mark_article_seen(article_url, article.title, project_name)

    logger.info("Пост #%d создан, ожидает проверки", post_id)

    # Notify admin
    await send_admin_notification(
        bot=app.bot,
        post_id=post_id,
        article_title=article.title,
        article_url=article_url,
        post_text=post_text,
        image_paths=image_paths,
        video_paths=video_paths,
        scheduled_at=scheduled_at,
    )


async def check_news(app: Application, project_name: str = "gaming") -> None:
    """Check a single project's news source for new articles."""
    proj = get_project(project_name)
    source = get_source(proj.get("source"))
    logger.info("Проверка новых статей проекта '%s'...", project_name)
    try:
        async with aiohttp.ClientSession() as session:
            articles = await source.get_latest_links(session)

        new_count = 0
        # Relevance filter is driven by projects.json → ai.relevance. When no
        # config is present, only the default gaming project filters (legacy
        # behaviour); other topic-specific sources are already on-topic.
        relevance = project_ai(project_name, "relevance")
        if relevance is None:
            do_relevance = (project_name == "gaming")
        elif isinstance(relevance, dict):
            do_relevance = bool(relevance.get("enabled", True))
        else:
            do_relevance = bool(relevance)
        for art in articles:
            if db.is_article_seen(art["url"]):
                continue
            # AI relevance check (if enabled for this project).
            if do_relevance and not await is_gaming_related(art["title"], prompt=relevance):
                logger.info("AI: не по теме проекта, пропускаем: %s", art["title"])
                db.mark_article_seen(art["url"], art["title"], project_name)
                continue
            new_count += 1
            await process_article(app, art["url"], art["title"], project_name)
            # Small delay between articles to avoid hammering the site or Claude
            await asyncio.sleep(3)

        logger.info("Проверка '%s' завершена. Новых статей: %d", project_name, new_count)
    except Exception as exc:
        logger.exception("Ошибка при проверке новостей проекта '%s': %s", project_name, exc)


# ---------------------------------------------------------------------------
# Periodic job wrappers
# ---------------------------------------------------------------------------

async def job_check_news(context) -> None:
    project_name = context.job.data if context.job and context.job.data else "gaming"
    await check_news(context.application, project_name)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def validate_config() -> None:
    errors = []
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN is not set")
    if not TELEGRAM_ADMIN_CHAT_ID:
        errors.append("TELEGRAM_ADMIN_CHAT_ID is not set")
    if errors:
        for e in errors:
            logger.error("Ошибка конфигурации: %s", e)
        sys.exit(1)


def main() -> None:
    validate_config()
    setup_dirs()
    db.init_db()
    logger.info("База данных инициализирована")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(HTTPXRequest(
            httpx_kwargs={"verify": certifi.where()},
            connect_timeout=15,
            read_timeout=60,
            write_timeout=300,   # large file uploads (video) can take time
            pool_timeout=30,
        ))
    )
    if TELEGRAM_LOCAL_MODE:
        # Use a self-hosted telegram-bot-api server (raises the 50 MB upload cap to 2 GB).
        app = (
            app
            .base_url(f"{TELEGRAM_LOCAL_API_URL}/bot")
            .base_file_url(f"{TELEGRAM_LOCAL_API_FILE_URL}/bot")
            .local_mode(True)
        )
        logger.info("Telegram: local Bot API server at %s", TELEGRAM_LOCAL_API_URL)
    app = app.build()

    # Register Telegram bot handlers
    for handler in build_handlers():
        app.add_handler(handler)
    app.add_error_handler(error_handler)

    job_queue = app.job_queue

    # On startup: schedule one repeating news-check per project, each with its
    # own interval (falls back to the global CHECK_INTERVAL_MINUTES).
    names = project_names() or ["gaming"]
    for offset, name in enumerate(names):
        proj = get_project(name)
        try:
            interval_min = int(proj.get("check_interval_minutes", CHECK_INTERVAL_MINUTES))
        except (TypeError, ValueError):
            interval_min = CHECK_INTERVAL_MINUTES
        job_queue.run_repeating(
            job_check_news,
            interval=interval_min * 60,
            first=15 + offset * 5,  # stagger project checks a few seconds apart
            name=f"check-{name}",
            data=name,
        )
        logger.info("Проект '%s': проверка каждые %d мин", name, interval_min)

    logger.info(
        "Бот запущен. Проектов: %d. Публикация по Approve.",
        len(names),
    )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
