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
from ai_adapter import shorten_post, generate_video_script
import video_generator
import instagram_publisher

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
    """Delete downloaded image and video files after cancel."""
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
        logger.info("Пост #%d опубликован в %s", post_id, TELEGRAM_CHANNEL_ID)
        return True
    except TimedOut:
        # Telegram принял запрос, но ответ не успел прийти — пост доставлен
        db.update_post_status(post_id, "sent")
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

    create_video_keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Create video", callback_data=f"create_video:{post_id}"),
    ]])
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Post #{post_id} published. Generate a TikTok/Shorts video?",
        reply_markup=create_video_keyboard,
    )


def _build_video_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Create video", callback_data=f"create_video:{post_id}"),
    ]])


def _build_video_done_keyboard(post_id: int) -> InlineKeyboardMarkup:
    """Keyboard shown after a video has been successfully generated."""
    row1 = [InlineKeyboardButton("🔄 Regenerate", callback_data=f"create_video:{post_id}")]
    row2 = []
    if instagram_publisher.is_configured():
        row2.append(InlineKeyboardButton("📲 Post to Instagram", callback_data=f"post_instagram:{post_id}"))
    buttons = [row1, row2] if row2 else [row1]
    return InlineKeyboardMarkup(buttons)


async def handle_create_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a TikTok/Reels/Shorts video for the approved article."""
    query = update.callback_query
    # Acknowledge the tap without altering the button message — keeps it reusable
    await query.answer("Starting video generation...")
    post_id = int(query.data.split(":")[1])

    post = db.get_scheduled_post(post_id)
    if not post:
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Post #{post_id} not found.",
        )
        return

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Generating video for post #{post_id}...\nStep 1/4: Writing TikTok script",
    )

    # 1. Generate narration script via AI
    script = await generate_video_script(
        post_text=post["post_text"],
        article_title=post["article_title"],
    )

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Script for post #{post_id}:\n\n{script}",
    )

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text="Step 2/4: Synthesizing voice (edge-tts)...",
    )

    # Derive search keywords from the narration script — more thematic than 
    # just the headline, gives Pixabay better context.
    # Extract nouns/game-specific words: capitalised words + words >4 chars,
    # skip stop-words, take up to 3 for a focused query.
    stop = {"a","an","the","and","or","of","in","is","to","for","was",
            "are","on","at","by","it","with","this","that","its","has",
            "but","not","yet","still","very","just","even","been","will",
            "from","than","what","how","long","time","could","would","they"}
    script_words = re.findall(r"[A-Za-z][a-z]{3,}|[A-Z][A-Za-z]{2,}", script)
    keywords = [w for w in script_words if w.lower() not in stop]
    # Prefer capitalised (proper noun) words first, then any long word
    proper   = [w for w in keywords if w[0].isupper()][:3]
    common   = [w for w in keywords if not w[0].isupper()][:2]
    chosen   = (proper + common)[:3]
    search_query = " ".join(chosen) if chosen else "gaming video game"

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Step 3/4: Collecting media (Pixabay: '{search_query}')...",
    )

    # 2. Generate video
    video_path = await video_generator.create_short_video(
        post=post,
        script=script,
        search_query=search_query,
    )

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=(
                f"Video generation failed for post #{post_id}.\n"
                "Check logs for details. Tap the button below to retry."
            ),
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    # 3. Send the finished video to admin
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text="Step 4/4: Uploading video to Telegram...",
    )
    try:
        file_size = os.path.getsize(video_path)
        if file_size > 50 * 1024 * 1024:        # Telegram bot limit: 50 MB
            await context.bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=(
                    f"Video file is too large ({file_size // (1024*1024)} MB > 50 MB).\n"
                    f"Saved locally at:\n{video_path}"
                ),
                reply_markup=_build_video_keyboard(post_id),
            )
            return
        with open(video_path, "rb") as vf:
            await context.bot.send_document(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                document=vf,
                filename=os.path.basename(video_path),
                caption=f"TikTok/Shorts video for post #{post_id} (no compression)",
                write_timeout=300,
                read_timeout=120,
            )
        # Keep video on disk and persist path to DB so it survives bot restarts
        db.set_generated_video_path(post_id, video_path)
        context.bot_data[f"video_caption:{post_id}"] = script
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Video for post #{post_id} ready.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
    except TelegramError as exc:
        logger.error("Failed to send video for post #%d: %s", post_id, exc)
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Could not send video: {exc}\nSaved at: {video_path}",
            reply_markup=_build_video_keyboard(post_id),
        )


async def handle_post_instagram(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated video to Instagram as a Reel."""
    query = update.callback_query
    await query.answer("Uploading to Instagram…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path(post_id)
    post       = db.get_scheduled_post(post_id)
    caption    = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading Reel for post #{post_id} to Instagram…\n(1) Uploading to temporary storage\u2026",
    )

    try:
        # instagram_publisher handles: R2 upload → container → poll → publish → R2 delete
        media_id = await instagram_publisher.publish_reel(
            video_path=video_path,
            caption=caption,
        )

        # Clean up local file and clear DB path
        try:
            os.remove(video_path)
        except OSError:
            pass
        db.set_generated_video_path(post_id, None)
        context.bot_data.pop(f"video_caption:{post_id}", None)

        await status_msg.edit_text(
            f"\u2705 Reel published to Instagram for post #{post_id}\n"
            f"Media ID: {media_id}\n\n"
            f"Generate another video?",
            reply_markup=_build_video_keyboard(post_id),
        )
    except Exception as exc:
        logger.error("Instagram publish failed for post #%d: %s", post_id, exc)
        await status_msg.edit_text(
            f"\u274c Instagram publish failed for post #{post_id}:\n{exc}\n\n"
            f"Video file is still available locally.",
            reply_markup=_build_video_done_keyboard(post_id),
        )


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
        CallbackQueryHandler(handle_approve,        pattern=r"^approve:\d+$"),
        CallbackQueryHandler(handle_cancel,         pattern=r"^cancel:\d+$"),
        CallbackQueryHandler(handle_create_video,   pattern=r"^create_video:\d+$"),
        CallbackQueryHandler(handle_post_instagram, pattern=r"^post_instagram:\d+$"),
    ]
