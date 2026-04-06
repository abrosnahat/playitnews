"""
Main entry point.
Starts the Telegram bot and schedules the periodic news check.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

import aiohttp
import certifi
from telegram.ext import Application
from telegram.request import HTTPXRequest

import database as db
from ai_adapter import adapt_article, adapt_article_ru, is_gaming_related
from bot import build_handlers, publish_post, send_admin_notification
from config import (
    CHECK_INTERVAL_MINUTES,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_ADMIN_CHAT_ID,
)
from scraper import download_images, download_videos, get_latest_article_links, scrape_article

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

async def process_article(app: Application, article_url: str, article_title: str) -> None:
    """Full pipeline for a single new article."""
    logger.info("Обрабатываем статью: %s", article_url)

    async with aiohttp.ClientSession() as session:
        article = await scrape_article(session, article_url)
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
        adapt_article(article.title, article.text),
        adapt_article_ru(article.title, article.text),
    )

    # Append YouTube links to post text (after AI adaptation)
    if youtube_urls:
        post_text = post_text + "\n\n" + "\n".join(youtube_urls)
        ru_post_text = ru_post_text + "\n\n" + "\n".join(youtube_urls)

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
    )
    db.mark_article_seen(article_url, article.title)

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


async def check_news(app: Application) -> None:
    """Check playground.ru/news for new articles."""
    logger.info("Проверка новых статей на playground.ru...")
    try:
        async with aiohttp.ClientSession() as session:
            articles = await get_latest_article_links(session)

        new_count = 0
        for art in articles:
            if not db.is_article_seen(art["url"]):
                # AI relevance check — skip non-gaming articles
                if not await is_gaming_related(art["title"]):
                    logger.info("AI: не игровая тематика, пропускаем: %s", art["title"])
                    db.mark_article_seen(art["url"], art["title"])
                    continue
                new_count += 1
                await process_article(app, art["url"], art["title"])
                # Small delay between articles to avoid hammering the site or Claude
                await asyncio.sleep(3)

        logger.info("Проверка завершена. Новых статей: %d", new_count)
    except Exception as exc:
        logger.exception("Ошибка при проверке новостей: %s", exc)


async def dispatch_due_posts(app: Application) -> None:
    """Publish any posts that are past their scheduled time."""
    now = datetime.now(timezone.utc)
    due_posts = db.get_pending_posts_due(now)
    for post in due_posts:
        # Auto-publish if admin hasn't explicitly approved or cancelled
        # (both 'pending' and 'approved' are published; only 'cancelled' is skipped)
        logger.info("Publishing scheduled post #%d", post["id"])
        await publish_post(bot=app.bot, post_id=post["id"])


# ---------------------------------------------------------------------------
# Periodic job wrappers for python-telegram-bot job queue
# ---------------------------------------------------------------------------

async def job_check_news(context) -> None:
    await check_news(context.application)


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
        .build()
    )

    # Register Telegram bot handlers
    for handler in build_handlers():
        app.add_handler(handler)

    job_queue = app.job_queue

    # Check news every CHECK_INTERVAL_MINUTES
    job_queue.run_repeating(
        job_check_news,
        interval=CHECK_INTERVAL_MINUTES * 60,
        first=15,
    )

    logger.info(
        "Бот запущен. Проверка каждые %d мин. Публикация по Approve.",
        CHECK_INTERVAL_MINUTES,
    )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
