"""
Telegram bot handlers and post-sending logic.
"""
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from telegram import (
    Bot,
    InputMediaPhoto,
    InputMediaVideo,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.error import TelegramError, TimedOut

import database as db
from config import TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_CHANNEL_ID
from ai_adapter import shorten_post

logger = logging.getLogger(__name__)


def _probe_video_dims(path: str) -> tuple[int, int]:
    """Return (width, height) of a video file using ffprobe, or (0, 0) on failure."""
    try:
        import subprocess, json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        stream = (data.get("streams") or data.get("programs", [{}])[0].get("streams", []))
        if stream:
            return stream[0].get("width", 0), stream[0].get("height", 0)
    except Exception as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
    return 0, 0

# ---------------------------------------------------------------------------
# Notification to admin
# ---------------------------------------------------------------------------

async def send_admin_notification(
    bot: Bot,
    post_id: int,
    article_title: str,
    article_url: str,
    post_text: str,
    image_paths: list[str],
    scheduled_at: datetime,
    video_paths: list[str] | None = None,
) -> Optional[int]:
    """Send post preview to admin exactly as it will appear in the channel,
    then a control message with Approve / Edit / Cancel buttons."""
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Approve", callback_data=f"approve:{post_id}"),
        InlineKeyboardButton("Cancel", callback_data=f"cancel:{post_id}"),
    ]])

    valid_images = [p for p in image_paths if os.path.exists(p)]
    valid_videos = [p for p in (video_paths or []) if os.path.exists(p)]

    try:
        # 1. Send exact post preview (photos + videos in one post with caption)
        await _send_media_post(bot, TELEGRAM_ADMIN_CHAT_ID, post_text, valid_images, valid_videos)

        # 2. Control message with buttons
        msg = await bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Post #{post_id} — {article_title}\n{article_url}",
            reply_markup=keyboard,
            parse_mode=None,
            disable_web_page_preview=True,
        )
        db.set_notification_message_id(post_id, msg.message_id)
        return msg.message_id
    except TelegramError as exc:
        logger.error("Не удалось отправить уведомление: %s", exc)
        return None


async def _send_media_post(
    bot: Bot,
    chat_id,
    text: str,
    image_paths: list[str],
    video_paths: list[str] | None = None,
) -> None:
    """Send photos and/or videos with text as caption in a single post."""
    LIMIT = 1024
    FOOTER = "\n\n@playitnews"

    # Extract YouTube URLs from the text so they survive AI shortening
    yt_pattern = re.compile(r'\nhttps?://(?:www\.)?youtube\.com/watch\S*', re.MULTILINE)
    yt_links = "".join(yt_pattern.findall(text.rstrip()))
    body = yt_pattern.sub("", text.rstrip()).rstrip()

    suffix = yt_links + FOOTER  # e.g. \nhttps://youtu...\nhttps://youtu...\n\n@playitnews

    # If body + suffix exceeds the caption limit, ask AI to rewrite shorter
    if len(body) + len(suffix) > LIMIT:
        body = await shorten_post(body, target_chars=LIMIT - len(suffix))

    text = body.rstrip() + suffix
    caption = text[:LIMIT]
    all_media_paths = list(image_paths[:10]) + list((video_paths or [])[:max(0, 10 - len(image_paths))])

    if not all_media_paths:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=False)
        return

    if len(all_media_paths) == 1:
        path = all_media_paths[0]
        with open(path, "rb") as f:
            data = f.read()
        if path in image_paths:
            await bot.send_photo(chat_id=chat_id, photo=data, caption=caption, parse_mode=ParseMode.HTML)
        else:
            w, h = _probe_video_dims(path)
            await bot.send_video(
                chat_id=chat_id, video=data, caption=caption, parse_mode=ParseMode.HTML,
                width=w or None, height=h or None, supports_streaming=True,
            )
        return

    media = []
    for i, path in enumerate(all_media_paths):
        with open(path, "rb") as f:
            data = f.read()
        cap = caption if i == 0 else None
        if path in image_paths:
            media.append(InputMediaPhoto(media=data, caption=cap, parse_mode=ParseMode.HTML if cap else None))
        else:
            w, h = _probe_video_dims(path)
            media.append(InputMediaVideo(
                media=data, caption=cap, parse_mode=ParseMode.HTML if cap else None,
                width=w or None, height=h or None, supports_streaming=True,
            ))
    await bot.send_media_group(chat_id=chat_id, media=media)


async def _send_preview_with_images(bot: Bot, chat_id, text: str, image_paths: list[str]) -> None:
    """Backwards-compat wrapper."""
    await _send_media_post(bot, chat_id, text, image_paths)


# ---------------------------------------------------------------------------
# Media cleanup
# ---------------------------------------------------------------------------

def _cleanup_media_files(image_paths: list[str], video_paths: list[str]) -> None:
    """Delete downloaded image and video files after publish or cancel."""
    for path in list(image_paths) + list(video_paths):
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.debug("Удалён медиафайл: %s", path)
        except OSError as exc:
            logger.warning("Не удалось удалить файл %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Publish post to channel
# ---------------------------------------------------------------------------

async def publish_post(bot: Bot, post_id: int) -> bool:
    """Send the post (text + images) to the Telegram channel."""
    post = db.get_scheduled_post(post_id)
    if not post:
        logger.error("Пост #%d не найден в БД", post_id)
        return False

    if post["status"] in ("cancelled", "sent"):
        logger.info("Пост #%d уже имеет статус '%s', пропускаем", post_id, post["status"])
        return False

    text = post["post_text"]
    image_paths: list[str] = [p for p in post["image_paths"] if os.path.exists(p)]
    video_paths: list[str] = [p for p in post.get("video_paths", []) if os.path.exists(p)]

    try:
        await _send_media_post(bot, TELEGRAM_CHANNEL_ID, text, image_paths, video_paths)

        db.update_post_status(post_id, "sent")
        _cleanup_media_files(image_paths, video_paths)
        logger.info("Пост #%d опубликован в %s", post_id, TELEGRAM_CHANNEL_ID)
        return True
    except TimedOut:
        # Telegram принял запрос, но ответ не успел прийти — пост доставлен
        db.update_post_status(post_id, "sent")
        _cleanup_media_files(image_paths, video_paths)
        logger.warning("Пост #%d опубликован (ответ Telegram задержался, но пост в канале)", post_id)
        return True
    except TelegramError as exc:
        logger.error("Ошибка публикации поста #%d: %s", post_id, exc)
        return False


async def _send_with_images(bot: Bot, text: str, image_paths: list[str]) -> None:
    """Backwards-compat wrapper."""
    await _send_media_post(bot, TELEGRAM_CHANNEL_ID, text, image_paths)


# ---------------------------------------------------------------------------
# Callback query handlers
# ---------------------------------------------------------------------------

async def handle_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    post_id = int(query.data.split(":")[1])

    post = db.get_scheduled_post(post_id)
    if not post or post["status"] != "pending":
        await query.edit_message_text("This post is no longer pending.")
        return

    await query.edit_message_text(f"Post #{post_id} approved. Publishing now...")
    await publish_post(bot=context.bot, post_id=post_id)


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    post_id = int(query.data.split(":")[1])

    post = db.get_scheduled_post(post_id)
    db.update_post_status(post_id, "cancelled")
    if post:
        _cleanup_media_files(post.get("image_paths", []), post.get("video_paths", []))
    await query.edit_message_text(f"Post #{post_id} has been cancelled.")


# ---------------------------------------------------------------------------
# Build handlers list for Application
# ---------------------------------------------------------------------------

def build_handlers():
    return [
        CallbackQueryHandler(handle_approve, pattern=r"^approve:\d+$"),
        CallbackQueryHandler(handle_cancel, pattern=r"^cancel:\d+$"),
    ]
