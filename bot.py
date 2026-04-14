"""
Telegram bot handlers and post-sending logic.
"""
import asyncio
import html
import logging
import os
import re
import shutil
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
from telegram.error import TelegramError, TimedOut, BadRequest

import database as db
from config import TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_CHANNEL_ID, TELEGRAM_SECOND_CHANNEL_ID
from ai_adapter import shorten_post, generate_video_script, generate_video_script_ru
import ai_adapter
import video_generator
import instagram_publisher
import youtube_publisher

# Telegram HTML subset: only these tags are allowed
_TG_ALLOWED = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
               "code", "pre", "a", "tg-spoiler"}
# Only match tags where the name is immediately followed by >, />, or whitespace
_TAG_RE = re.compile(r'<(/?)([a-zA-Z][a-zA-Z0-9-]*)(\s[^>]*)?>',  re.DOTALL)


def _sanitize_telegram_html(text: str) -> str:
    """Remove HTML tags not in Telegram's allowed set; escape bare < and > in plain text."""
    result = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        start, end = m.start(), m.end()
        # Escape only < and > in literal text before this tag (don't touch &entities;)
        result.append(text[pos:start].replace("<", "&lt;").replace(">", "&gt;"))
        tag_name = m.group(2).lower()
        if tag_name in _TG_ALLOWED:
            result.append(m.group(0))
        pos = end
    result.append(text[pos:].replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(result)

# How many YouTube results to skip forward on each regenerate
YT_SKIP_STEP = 3

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
    footer: str = "@playitgamesnews",
) -> None:
    """Send photos and/or videos with text as caption in a single post."""
    LIMIT = 1024
    FOOTER = f"\n\n{footer}"

    # Extract YouTube/VK URLs from the text so they survive truncation
    yt_pattern = re.compile(r'\nhttps?://(?:www\.)?(?:youtube\.com/watch|vk\.com/video)\S*', re.MULTILINE)
    yt_links = "".join(yt_pattern.findall(text.rstrip()))
    body = yt_pattern.sub("", text.rstrip()).rstrip()

    suffix = yt_links + FOOTER  # e.g. \nhttps://youtu...\n\n@playitnews

    # Hard-truncate body to fit the caption limit (Telegram = 1024 chars for media captions)
    max_body = LIMIT - len(suffix)
    if len(body) > max_body:
        body = body[:max_body].rstrip()
        # Don't cut mid-tag — strip any partial opening tag at the end
        last_open = body.rfind("<")
        if last_open != -1 and ">" not in body[last_open:]:
            body = body[:last_open].rstrip()
        # Close any open <b>/<i> tags
        for tag in ("i", "b"):
            opens  = body.count(f"<{tag}>")
            closes = body.count(f"</{tag}>")
            body += f"</{tag}>" * max(0, opens - closes)

    text = body + suffix
    text = _sanitize_telegram_html(text)
    caption = text[:LIMIT]
    # Fallback plain text (all HTML tags stripped) used if Telegram rejects the HTML
    plain_text = re.sub(r'<[^>]+>', '', text)
    plain_caption = plain_text[:LIMIT]
    all_media_paths = list(image_paths[:10]) + list((video_paths or [])[:max(0, 10 - len(image_paths))])

    async def _send_with_fallback(make_html, make_plain):
        try:
            await make_html()
        except TelegramError as exc:
            if "parse entities" in str(exc).lower() or "can't parse" in str(exc).lower():
                logger.warning("HTML parse error, retrying as plain text: %s", exc)
                await make_plain()
            else:
                raise

    if not all_media_paths:
        await _send_with_fallback(
            lambda: bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=False),
            lambda: bot.send_message(chat_id=chat_id, text=plain_text, parse_mode=None, disable_web_page_preview=False),
        )
        return

    if len(all_media_paths) == 1:
        path = all_media_paths[0]
        with open(path, "rb") as f:
            data = f.read()
        if path in image_paths:
            await _send_with_fallback(
                lambda: bot.send_photo(chat_id=chat_id, photo=data, caption=caption, parse_mode=ParseMode.HTML),
                lambda: bot.send_photo(chat_id=chat_id, photo=data, caption=plain_caption, parse_mode=None),
            )
        else:
            w, h = _probe_video_dims(path)
            await _send_with_fallback(
                lambda: bot.send_video(chat_id=chat_id, video=data, caption=caption, parse_mode=ParseMode.HTML,
                               width=w or None, height=h or None, supports_streaming=True),
                lambda: bot.send_video(chat_id=chat_id, video=data, caption=plain_caption, parse_mode=None,
                               width=w or None, height=h or None, supports_streaming=True),
            )
        return

    def _build_media(cap):
        items = []
        for i, path in enumerate(all_media_paths):
            with open(path, "rb") as f:
                data = f.read()
            c = cap if i == 0 else None
            pm = ParseMode.HTML if c else None
            if path in image_paths:
                items.append(InputMediaPhoto(media=data, caption=c, parse_mode=pm))
            else:
                w, h = _probe_video_dims(path)
                items.append(InputMediaVideo(
                    media=data, caption=c, parse_mode=pm,
                    width=w or None, height=h or None, supports_streaming=True,
                ))
        return items

    await _send_with_fallback(
        lambda: bot.send_media_group(chat_id=chat_id, media=_build_media(caption)),
        lambda: bot.send_media_group(chat_id=chat_id, media=_build_media(plain_caption)),
    )


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
    ru_text = post.get("ru_post_text")
    image_paths: list[str] = [p for p in post["image_paths"] if os.path.exists(p)]
    video_paths: list[str] = [p for p in post.get("video_paths", []) if os.path.exists(p)]

    try:
        await _send_media_post(bot, TELEGRAM_CHANNEL_ID, text, image_paths, video_paths)
        logger.info("Пост #%d опубликован в %s", post_id, TELEGRAM_CHANNEL_ID)

        if TELEGRAM_SECOND_CHANNEL_ID:
            if not ru_text:
                logger.warning("Пост #%d: нет русского текста, пропускаем %s", post_id, TELEGRAM_SECOND_CHANNEL_ID)
            else:
                try:
                    await _send_media_post(bot, TELEGRAM_SECOND_CHANNEL_ID, ru_text, image_paths, video_paths, footer="@readitgames")
                    logger.info("Пост #%d опубликован в %s", post_id, TELEGRAM_SECOND_CHANNEL_ID)
                except TelegramError as exc2:
                    logger.error("Ошибка публикации в второй канал #%d: %s", post_id, exc2)

        db.update_post_status(post_id, "sent")
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
        InlineKeyboardButton("🎬 Create video (EN+RU)", callback_data=f"create_video:{post_id}"),
    ]])
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Post #{post_id} published. Generate a Reels/Shorts video?",
        reply_markup=create_video_keyboard,
    )


def _build_video_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🎬 Create video (EN+RU)", callback_data=f"create_video:{post_id}"),
    ]])


def _build_video_done_keyboard(post_id: int) -> InlineKeyboardMarkup:
    """Keyboard shown after a video has been successfully generated."""
    row_regen = [
        InlineKeyboardButton("🔄 EN+RU",   callback_data=f"create_video:{post_id}"),
        InlineKeyboardButton("🇬🇧 EN",      callback_data=f"create_video_en:{post_id}"),
        InlineKeyboardButton("🇷🇺 RU",      callback_data=f"create_video_ru:{post_id}"),
    ]
    ig_ok    = instagram_publisher.is_configured()
    ig_ru_ok = instagram_publisher.is_configured_ru()
    yt_ok    = youtube_publisher.is_configured()
    yt_ru_ok = youtube_publisher.is_configured_ru()
    row2 = []
    if ig_ok:
        row2.append(InlineKeyboardButton("📲 Instagram",    callback_data=f"post_instagram:{post_id}"))
    if ig_ru_ok:
        row2.append(InlineKeyboardButton("📲 Instagram RU", callback_data=f"post_instagram_ru:{post_id}"))
    if yt_ok:
        row2.append(InlineKeyboardButton("▶️ YouTube",      callback_data=f"post_youtube:{post_id}"))
    if yt_ru_ok:
        row2.append(InlineKeyboardButton("▶️ YouTube RU",   callback_data=f"post_youtube_ru:{post_id}"))
    row3 = []
    if sum([ig_ok, yt_ok]) >= 2:
        row3.append(InlineKeyboardButton("🌐 Post to All",       callback_data=f"post_all:{post_id}"))
    if sum([ig_ru_ok, yt_ru_ok]) >= 1:
        row3.append(InlineKeyboardButton("🌐 Post to All RU",    callback_data=f"post_all_ru:{post_id}"))
    row4 = []
    if sum([ig_ok, yt_ok]) >= 1 and sum([ig_ru_ok, yt_ru_ok]) >= 1:
        row4.append(InlineKeyboardButton("🌍 Post to All EN+RU", callback_data=f"post_all_combined:{post_id}"))
    buttons = [row_regen]
    if row2:
        buttons.append(row2)
    if row3:
        buttons.append(row3)
    if row4:
        buttons.append(row4)
    return InlineKeyboardMarkup(buttons)


async def handle_create_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate EN and RU Reels/Shorts videos simultaneously."""
    query = update.callback_query
    await query.answer("Starting video generation...")
    post_id = int(query.data.split(":")[1])

    post = db.get_scheduled_post(post_id)
    if not post:
        await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text=f"Post #{post_id} not found.")
        return

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Generating EN + RU videos for post #{post_id}...\nStep 1/4: Writing scripts",
    )

    # 1. Generate both scripts in parallel (translate title to EN first for the EN script)
    en_article_title = await ai_adapter.translate_title_to_english(post["article_title"])
    en_script, ru_script = await asyncio.gather(
        generate_video_script(post_text=post["post_text"], article_title=en_article_title),
        generate_video_script_ru(post_text=post.get("ru_post_text") or post["post_text"], article_title=post["article_title"]),
    )

    # Fallback if AI returned empty or too-short scripts (< 20 words = model failure)
    en_clean = post["post_text"][:350].strip()
    ru_clean = (post.get("ru_post_text") or post["post_text"])[:350].strip()

    if not en_script or len(re.sub(r'<[^>]+>', '', en_script).split()) < 20:
        logger.warning("EN video script too short (%s), using post_text fallback",
                       repr(en_script[:60]) if en_script else "empty")
        en_script = re.sub(r'<[^>]+>', '', en_clean).strip()
    else:
        en_script = re.sub(r'<[^>]+>', '', en_script).strip()

    if not ru_script or len(re.sub(r'<[^>]+>', '', ru_script).split()) < 20:
        logger.warning("RU video script too short (%s), using post_text fallback",
                       repr(ru_script[:60]) if ru_script else "empty")
        ru_script = re.sub(r'<[^>]+>', '', ru_clean).strip()
    else:
        ru_script = re.sub(r'<[^>]+>', '', ru_script).strip()

    # Send scripts as separate messages to avoid 4096-char Telegram limit
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"🇬🇧 EN script:\n{en_script}",
    )
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"🇷🇺 RU script:\n{ru_script}",
    )

    await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text="Step 2/4: Synthesizing voices...")

    title = post.get("article_title", "")
    search_query = await ai_adapter.extract_game_name(title)
    if not search_query:
        first_chunk = re.split(r"[:–—|]", title)[0].strip()
        latin_tokens = [t for t in first_chunk.split() if re.match(r"^[A-Za-z0-9'&.]+$", t)]
        search_query = " ".join(latin_tokens).strip() or title[:50]

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Step 3/4: Searching YouTube for '{search_query}'",
    )

    yt_skip = db.increment_yt_skip(post_id, YT_SKIP_STEP) - YT_SKIP_STEP

    # Download clips once — shared between EN and RU renders
    article_videos, yt_clips, clips_workdir = await video_generator.fetch_gameplay_clips(
        post=post, search_query=search_query, yt_skip=yt_skip,
    )
    shared_clips = article_videos + yt_clips

    # 2. Generate EN and RU videos sequentially to avoid OOM from parallel ffmpeg/Pillow loads
    try:
        en_path = await video_generator.create_short_video(post=post, script=en_script, search_query=search_query, yt_skip=yt_skip, lang="en", prefetched_clips=shared_clips, n_article_clips=len(article_videos))
        ru_path = await video_generator.create_short_video(post=post, script=ru_script, search_query=search_query, yt_skip=yt_skip, lang="ru", prefetched_clips=shared_clips, n_article_clips=len(article_videos))
    finally:
        shutil.rmtree(clips_workdir, ignore_errors=True)

    if not en_path and not ru_path:
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Video generation failed for post #{post_id}.\nCheck logs. Tap below to retry.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text="Step 4/4: Uploading videos to Telegram...")

    async def _send_video(path: str | None, label: str, script: str) -> None:
        if not path or not os.path.exists(path):
            await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text=f"{label} video failed.")
            return
        file_size = os.path.getsize(path)
        if file_size > 50 * 1024 * 1024:
            await context.bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=f"{label} video too large ({file_size // (1024*1024)} MB > 50 MB).\nSaved at: {path}",
            )
            return
        try:
            w, h = _probe_video_dims(path)
            with open(path, "rb") as vf:
                await context.bot.send_video(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID,
                    video=vf,
                    caption=f"{label} video for post #{post_id}",
                    width=w or None, height=h or None,
                    supports_streaming=True,
                    write_timeout=300,
                    read_timeout=120,
                )
        except TelegramError as exc:
            logger.warning("send_video failed (%s), falling back to send_document", exc)
            try:
                with open(path, "rb") as vf:
                    await context.bot.send_document(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        document=vf,
                        filename=os.path.basename(path),
                        caption=f"{label} video for post #{post_id}",
                        write_timeout=300,
                        read_timeout=120,
                    )
            except TelegramError as exc2:
                logger.error("Failed to send %s video for post #%d: %s", label, post_id, exc2)
                await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text=f"Could not send {label} video: {exc2}")

    await _send_video(en_path, "🇬🇧 EN", en_script)
    await _send_video(ru_path, "🇷🇺 RU", ru_script)

    # Persist EN video path for publish buttons
    final_path = en_path or ru_path
    db.set_generated_video_path(post_id, final_path)
    context.bot_data[f"video_caption:{post_id}"] = en_script
    if ru_path:
        db.set_generated_video_path_ru(post_id, ru_path)
        context.bot_data[f"ru_video_caption:{post_id}"] = ru_script

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Videos for post #{post_id} ready.",
        reply_markup=_build_video_done_keyboard(post_id),
    )


async def handle_create_video_en(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Regenerate EN video only."""
    query = update.callback_query
    await query.answer("Generating EN video...")
    post_id = int(query.data.split(":")[1])
    await _create_single_video(context, post_id, lang="en")


async def handle_create_video_ru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Regenerate RU video only."""
    query = update.callback_query
    await query.answer("Generating RU video...")
    post_id = int(query.data.split(":")[1])
    await _create_single_video(context, post_id, lang="ru")


async def _create_single_video(
    context: ContextTypes.DEFAULT_TYPE,
    post_id: int,
    lang: str,  # "en" or "ru"
) -> None:
    """Generate a single-language video and send it to the admin chat."""
    post = db.get_scheduled_post(post_id)
    if not post:
        await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text=f"Post #{post_id} not found.")
        return

    label = "\U0001f1ec\U0001f1e7 EN" if lang == "en" else "\U0001f1f7\U0001f1fa RU"
    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Generating {label} video for post #{post_id}...\nStep 1/4: Writing script",
    )

    if lang == "en":
        en_article_title = await ai_adapter.translate_title_to_english(post["article_title"])
        script = await generate_video_script(
            post_text=post["post_text"], article_title=en_article_title
        )
        fallback = re.sub(r'<[^>]+>', '', post["post_text"][:350]).strip()
    else:
        script = await generate_video_script_ru(
            post_text=post.get("ru_post_text") or post["post_text"],
            article_title=post["article_title"],
        )
        fallback = re.sub(r'<[^>]+>', '', (post.get("ru_post_text") or post["post_text"])[:350]).strip()

    script = re.sub(r'<[^>]+>', '', script or '').strip()
    if len(script.split()) < 20:
        logger.warning("%s video script too short (%s), using post_text fallback",
                       lang.upper(), repr(script[:60]))
        script = fallback

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"{label} script:\n{script}",
    )
    await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text="Step 2/4: Synthesizing voice...")

    title = post.get("article_title", "")
    search_query = await ai_adapter.extract_game_name(title)
    if not search_query:
        first_chunk = re.split(r"[:–—|]", title)[0].strip()
        latin_tokens = [t for t in first_chunk.split() if re.match(r"^[A-Za-z0-9'&.]+$", t)]
        search_query = " ".join(latin_tokens).strip() or title[:50]

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Step 3/4: Searching YouTube for '{search_query}'",
    )

    yt_skip = db.increment_yt_skip(post_id, YT_SKIP_STEP) - YT_SKIP_STEP
    article_videos, yt_clips, clips_workdir = await video_generator.fetch_gameplay_clips(
        post=post, search_query=search_query, yt_skip=yt_skip,
    )
    shared_clips = article_videos + yt_clips

    try:
        path = await video_generator.create_short_video(
            post=post, script=script, search_query=search_query,
            yt_skip=yt_skip, lang=lang,
            prefetched_clips=shared_clips, n_article_clips=len(article_videos),
        )
    finally:
        shutil.rmtree(clips_workdir, ignore_errors=True)

    if not path:
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"{label} video generation failed for post #{post_id}.\nCheck logs. Tap below to retry.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
        return

    await context.bot.send_message(chat_id=TELEGRAM_ADMIN_CHAT_ID, text="Step 4/4: Uploading video to Telegram...")

    file_size = os.path.getsize(path)
    if file_size > 50 * 1024 * 1024:
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"{label} video too large ({file_size // (1024*1024)} MB > 50 MB).\nSaved at: {path}",
        )
    else:
        try:
            w, h = _probe_video_dims(path)
            with open(path, "rb") as vf:
                await context.bot.send_video(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID, video=vf,
                    caption=f"{label} video for post #{post_id}",
                    width=w or None, height=h or None,
                    supports_streaming=True, write_timeout=300, read_timeout=120,
                )
        except TelegramError as exc:
            logger.warning("send_video failed (%s), falling back to send_document", exc)
            try:
                with open(path, "rb") as vf:
                    await context.bot.send_document(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID, document=vf,
                        filename=os.path.basename(path),
                        caption=f"{label} video for post #{post_id}",
                        write_timeout=300, read_timeout=120,
                    )
            except TelegramError as exc2:
                logger.error("Failed to send %s video for post #%d: %s", label, post_id, exc2)
                await context.bot.send_message(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID,
                    text=f"Could not send {label} video: {exc2}",
                )

    # Persist path and caption
    if lang == "en":
        db.set_generated_video_path(post_id, path)
        context.bot_data[f"video_caption:{post_id}"] = script
    else:
        db.set_generated_video_path_ru(post_id, path)
        context.bot_data[f"ru_video_caption:{post_id}"] = script

    await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"{label} video for post #{post_id} ready.",
        reply_markup=_build_video_done_keyboard(post_id),
    )


async def handle_post_instagram_ru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated RU video to the Russian Instagram account as a Reel."""
    from config import INSTAGRAM_USER_ID_RU, INSTAGRAM_ACCESS_TOKEN_RU
    query = update.callback_query
    await query.answer("Uploading to Instagram RU…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path_ru(post_id)
    post = db.get_scheduled_post(post_id)
    raw_caption = context.bot_data.get(f"ru_video_caption:{post_id}") or (post.get("ru_post_text", "") if post else "")
    caption = re.sub(r'[*_`~]', '', raw_caption)
    caption = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption)
    caption = re.sub(r'<[^>]+>', '', caption)
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("ru_post_text") or post.get("post_text", ""))
        hashtags = " ".join(re.findall(r'#\w+', post_text_clean))
        if hashtags and hashtags not in caption:
            caption = caption.rstrip() + "\n\n" + hashtags
    caption = "\u0411\u043e\u043b\u044c\u0448\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439 \u0432 telegram-\u043a\u0430\u043d\u0430\u043b\u0435, \u0441\u0441\u044b\u043b\u043a\u0430 \u0432 \u0431\u0438\u043e\n\n" + caption

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"RU video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading RU Reel for post #{post_id} to Instagram RU\u2026",
    )
    try:
        media_id = await instagram_publisher.publish_reel(
            video_path=video_path,
            caption=caption,
            user_id=INSTAGRAM_USER_ID_RU,
            access_token=INSTAGRAM_ACCESS_TOKEN_RU,
        )
        await status_msg.edit_text(
            f"\u2705 Reel published to Instagram RU for post #{post_id}\n"
            f"Media ID: {media_id}\n\n"
            f"Video file kept for further publishing.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
    except Exception as exc:
        logger.error("Instagram RU publish failed for post #%d: %s", post_id, exc)
        await status_msg.edit_text(
            f"\u274c Instagram RU publish failed for post #{post_id}:\n{exc}\n\n"
            f"Video file is still available locally.",
            reply_markup=_build_video_done_keyboard(post_id),
        )


async def handle_post_instagram(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated video to Instagram as a Reel."""
    query = update.callback_query
    await query.answer("Uploading to Instagram…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path(post_id)
    post       = db.get_scheduled_post(post_id)
    raw_caption = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")
    # Strip Telegram/Markdown formatting tags for Instagram plain-text caption
    caption = re.sub(r'[*_`~]', '', raw_caption)
    caption = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption)  # [text](url) → text
    caption = re.sub(r'<[^>]+>', '', caption)              # HTML tags
    # Append hashtags from post_text (they survive stripping as-is)
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("post_text", ""))
        hashtags = " ".join(re.findall(r'#\w+', post_text_clean))
        if hashtags and hashtags not in caption:
            caption = caption.rstrip() + "\n\n" + hashtags
    caption = "More news in the telegram channel, link in the bio\n\n" + caption

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

        await status_msg.edit_text(
            f"\u2705 Reel published to Instagram for post #{post_id}\n"
            f"Media ID: {media_id}\n\n"
            f"Video file kept for further publishing.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
    except Exception as exc:
        logger.error("Instagram publish failed for post #%d: %s", post_id, exc)
        await status_msg.edit_text(
            f"\u274c Instagram publish failed for post #{post_id}:\n{exc}\n\n"
            f"Video file is still available locally.",
            reply_markup=_build_video_done_keyboard(post_id),
        )


async def handle_post_youtube(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated video to YouTube Shorts."""
    query = update.callback_query
    await query.answer("Uploading to YouTube…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path(post_id)
    post = db.get_scheduled_post(post_id)

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    # Build title in English (translate from Russian via AI, fallback to original)
    raw_title = (post.get("article_title", "") if post else "") or f"Gaming news #{post_id}"
    title = await ai_adapter.translate_title_to_english(raw_title)
    title = title[:100]

    # Build description: clean script + hashtags + CTA
    raw_desc = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")
    desc = re.sub(r'[*_`~]', '', raw_desc)
    desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc)  # [text](url) → text
    desc = re.sub(r'<[^>]+>', '', desc)
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("post_text", ""))
        hashtags = " ".join(re.findall(r'#\w+', post_text_clean))
        if hashtags and hashtags not in desc:
            desc = desc.rstrip() + "\n\n" + hashtags
    desc = "More news in the telegram channel, link in the bio\n\n" + desc

    # Extract tags from hashtags in post_text
    tags: list[str] = []
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("post_text", ""))
        tags = [h.lstrip("#") for h in re.findall(r'#\w+', post_text_clean)]

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading Short for post #{post_id} to YouTube…",
    )

    try:
        video_id = await youtube_publisher.upload_short(
            video_path=video_path,
            title=title,
            description=desc,
            tags=tags,
        )
        await status_msg.edit_text(
            f"\u2705 YouTube Short published for post #{post_id}\n"
            f"https://youtu.be/{video_id}\n\n"
            f"Video file kept for further publishing.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
    except Exception as exc:
        logger.error("YouTube publish failed for post #%d: %s", post_id, exc)
        await status_msg.edit_text(
            f"\u274c YouTube publish failed for post #{post_id}:\n{exc}\n\n"
            f"Video file is still available locally.",
            reply_markup=_build_video_done_keyboard(post_id),
        )


async def handle_post_youtube_ru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated RU video to the Russian YouTube channel."""
    query = update.callback_query
    await query.answer("Uploading to YouTube RU…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path_ru(post_id)
    post = db.get_scheduled_post(post_id)

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"RU video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    raw_title = (post.get("article_title", "") if post else "") or f"Gaming news #{post_id}"
    title = raw_title[:100]  # Keep original Russian title for RU channel

    raw_desc = context.bot_data.get(f"ru_video_caption:{post_id}") or (post.get("ru_post_text", "") if post else "")
    desc = re.sub(r'[*_`~]', '', raw_desc)
    desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc)
    desc = re.sub(r'<[^>]+>', '', desc)
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("ru_post_text") or post.get("post_text", ""))
        hashtags = " ".join(re.findall(r'#\w+', post_text_clean))
        if hashtags and hashtags not in desc:
            desc = desc.rstrip() + "\n\n" + hashtags
    desc = "\u0411\u043e\u043b\u044c\u0448\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439 \u0432 telegram-\u043a\u0430\u043d\u0430\u043b\u0435, \u0441\u0441\u044b\u043b\u043a\u0430 \u0432 \u0431\u0438\u043e\u0432\u0435\n\n" + desc

    tags: list[str] = []
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("ru_post_text") or post.get("post_text", ""))
        tags = [h.lstrip("#") for h in re.findall(r'#\w+', post_text_clean)]

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading RU Short for post #{post_id} to YouTube RU…",
    )
    try:
        video_id = await youtube_publisher.upload_short_ru(
            video_path=video_path,
            title=title,
            description=desc,
            tags=tags,
        )
        await status_msg.edit_text(
            f"\u2705 YouTube RU Short published for post #{post_id}\n"
            f"https://youtu.be/{video_id}\n\n"
            f"Video file kept for further publishing.",
            reply_markup=_build_video_done_keyboard(post_id),
        )
    except Exception as exc:
        logger.error("YouTube RU publish failed for post #%d: %s", post_id, exc)
        await status_msg.edit_text(
            f"\u274c YouTube RU publish failed for post #{post_id}:\n{exc}\n\n"
            f"Video file is still available locally.",
            reply_markup=_build_video_done_keyboard(post_id),
        )




async def handle_post_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated EN video to all configured EN platforms simultaneously."""
    query = update.callback_query
    await query.answer("Uploading to all platforms…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path(post_id)
    post = db.get_scheduled_post(post_id)

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    # --- Shared caption (Instagram) ---
    raw_caption = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")
    caption = re.sub(r'[*_`~]', '', raw_caption)
    caption = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption)
    caption = re.sub(r'<[^>]+>', '', caption)
    tags: list[str] = []
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("post_text", ""))
        hashtags_list = re.findall(r'#\w+', post_text_clean)
        tags = [h.lstrip("#") for h in hashtags_list]
        hashtags_str = " ".join(hashtags_list)
        if hashtags_str and hashtags_str not in caption:
            caption = caption.rstrip() + "\n\n" + hashtags_str
    caption = "More news in the telegram channel, link in the bio\n\n" + caption

    # --- YouTube title / description ---
    raw_title = (post.get("article_title", "") if post else "") or f"Gaming news #{post_id}"
    yt_title = (await ai_adapter.translate_title_to_english(raw_title))[:100]
    raw_desc = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")
    yt_desc = re.sub(r'[*_`~]', '', raw_desc)
    yt_desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', yt_desc)
    yt_desc = re.sub(r'<[^>]+>', '', yt_desc)
    if tags:
        hashtags_str = " ".join(f"#{t}" for t in tags)
        if hashtags_str not in yt_desc:
            yt_desc = yt_desc.rstrip() + "\n\n" + hashtags_str
    yt_desc = "More news in the telegram channel, link in the bio\n\n" + yt_desc

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading to all platforms for post #{post_id}…",
    )

    # Launch all configured platforms concurrently
    tasks: dict[str, asyncio.Task] = {}
    if instagram_publisher.is_configured():
        tasks["Instagram"] = asyncio.create_task(
            instagram_publisher.publish_reel(video_path=video_path, caption=caption)
        )
    if youtube_publisher.is_configured():
        tasks["YouTube"] = asyncio.create_task(
            youtube_publisher.upload_short(video_path=video_path, title=yt_title, description=yt_desc, tags=tags)
        )
    results: list[str] = []
    any_err = False
    for platform, task in tasks.items():
        try:
            result = await task
            if platform == "YouTube" and isinstance(result, str) and result:
                results.append(f"\u2705 YouTube: https://youtu.be/{result}")
            elif platform == "Instagram" and isinstance(result, str) and result:
                results.append(f"\u2705 Instagram: Media ID {result}")
            else:
                results.append(f"\u2705 {platform}: published")
        except Exception as exc:
            any_err = True
            results.append(f"\u274c {platform}: {exc}")
            logger.error("%s publish failed for post #%d: %s", platform, post_id, exc)

    await status_msg.edit_text(
        "\n".join(results) + "\n\n"
        + ("Video file kept for further publishing." if any_err else "Done!"),
        reply_markup=_build_video_done_keyboard(post_id) if any_err else _build_video_keyboard(post_id),
    )


async def handle_post_all_ru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload the generated RU video to all configured RU platforms simultaneously."""
    query = update.callback_query
    await query.answer("Uploading RU to all platforms…")
    post_id = int(query.data.split(":")[1])

    video_path = db.get_generated_video_path_ru(post_id)
    post = db.get_scheduled_post(post_id)

    if not video_path or not os.path.exists(video_path):
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"RU video file for post #{post_id} not found. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    # --- Shared caption (Instagram RU) ---
    raw_caption = context.bot_data.get(f"ru_video_caption:{post_id}") or (post.get("ru_post_text", "") if post else "")
    caption = re.sub(r'[*_`~]', '', raw_caption)
    caption = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption)
    caption = re.sub(r'<[^>]+>', '', caption)
    tags: list[str] = []
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("ru_post_text") or post.get("post_text", ""))
        hashtags_list = re.findall(r'#\w+', post_text_clean)
        tags = [h.lstrip("#") for h in hashtags_list]
        hashtags_str = " ".join(hashtags_list)
        if hashtags_str and hashtags_str not in caption:
            caption = caption.rstrip() + "\n\n" + hashtags_str
    caption = "Больше новостей в telegram-канале, ссылка в био\n\n" + caption

    # --- YouTube RU title / description ---
    raw_title = (post.get("article_title", "") if post else "") or f"Gaming news #{post_id}"
    yt_title = raw_title[:100]  # Keep Russian title for RU channel
    yt_desc = caption  # reuse caption built above

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading RU to all platforms for post #{post_id}…",
    )

    from config import INSTAGRAM_USER_ID_RU, INSTAGRAM_ACCESS_TOKEN_RU
    tasks: dict[str, asyncio.Task] = {}
    if instagram_publisher.is_configured_ru():
        tasks["Instagram RU"] = asyncio.create_task(
            instagram_publisher.publish_reel(
                video_path=video_path, caption=caption,
                user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
            )
        )
    if youtube_publisher.is_configured_ru():
        tasks["YouTube RU"] = asyncio.create_task(
            youtube_publisher.upload_short_ru(
                video_path=video_path, title=yt_title, description=yt_desc, tags=tags,
            )
        )

    results: list[str] = []
    any_err = False
    for platform, task in tasks.items():
        try:
            result = await task
            if "YouTube" in platform and isinstance(result, str) and result:
                results.append(f"\u2705 {platform}: https://youtu.be/{result}")
            elif "Instagram" in platform and isinstance(result, str) and result:
                results.append(f"\u2705 {platform}: Media ID {result}")
            else:
                results.append(f"\u2705 {platform}: published")
        except Exception as exc:
            any_err = True
            results.append(f"\u274c {platform}: {exc}")
            logger.error("%s publish failed for post #%d: %s", platform, post_id, exc)

    await status_msg.edit_text(
        "\n".join(results) + "\n\n"
        + ("Video file kept for further publishing." if any_err else "Done!"),
        reply_markup=_build_video_done_keyboard(post_id) if any_err else _build_video_keyboard(post_id),
    )


async def handle_post_all_combined(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload EN video to EN platforms and RU video to RU platforms simultaneously."""
    query = update.callback_query
    await query.answer("Uploading to all EN+RU platforms…")
    post_id = int(query.data.split(":")[1])

    video_path_en = db.get_generated_video_path(post_id)
    video_path_ru = db.get_generated_video_path_ru(post_id)
    post = db.get_scheduled_post(post_id)

    missing: list[str] = []
    if not video_path_en or not os.path.exists(video_path_en):
        missing.append("EN")
    if not video_path_ru or not os.path.exists(video_path_ru):
        missing.append("RU")
    if missing:
        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f"Missing video files for post #{post_id}: {', '.join(missing)}. Please regenerate first.",
            reply_markup=_build_video_keyboard(post_id),
        )
        return

    # --- EN caption ---
    raw_en = context.bot_data.get(f"video_caption:{post_id}") or (post.get("post_text", "") if post else "")
    caption_en = re.sub(r'[*_`~]', '', raw_en)
    caption_en = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption_en)
    caption_en = re.sub(r'<[^>]+>', '', caption_en)
    tags_en: list[str] = []
    if post:
        post_text_clean = re.sub(r'<[^>]+>', '', post.get("post_text", ""))
        hashtags_list = re.findall(r'#\w+', post_text_clean)
        tags_en = [h.lstrip("#") for h in hashtags_list]
        hashtags_str = " ".join(hashtags_list)
        if hashtags_str and hashtags_str not in caption_en:
            caption_en = caption_en.rstrip() + "\n\n" + hashtags_str
    caption_en = "More news in the telegram channel, link in the bio\n\n" + caption_en

    # --- EN YouTube title/desc ---
    raw_title = (post.get("article_title", "") if post else "") or f"Gaming news #{post_id}"
    yt_title_en = (await ai_adapter.translate_title_to_english(raw_title))[:100]
    yt_desc_en = re.sub(r'[*_`~]', '', raw_en)
    yt_desc_en = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', yt_desc_en)
    yt_desc_en = re.sub(r'<[^>]+>', '', yt_desc_en)
    if tags_en:
        hashtags_str = " ".join(f"#{t}" for t in tags_en)
        if hashtags_str not in yt_desc_en:
            yt_desc_en = yt_desc_en.rstrip() + "\n\n" + hashtags_str
    yt_desc_en = "More news in the telegram channel, link in the bio\n\n" + yt_desc_en

    # --- RU caption ---
    raw_ru = context.bot_data.get(f"ru_video_caption:{post_id}") or (post.get("ru_post_text", "") if post else "")
    caption_ru = re.sub(r'[*_`~]', '', raw_ru)
    caption_ru = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption_ru)
    caption_ru = re.sub(r'<[^>]+>', '', caption_ru)
    tags_ru: list[str] = []
    if post:
        post_text_clean_ru = re.sub(r'<[^>]+>', '', post.get("ru_post_text") or post.get("post_text", ""))
        hashtags_list_ru = re.findall(r'#\w+', post_text_clean_ru)
        tags_ru = [h.lstrip("#") for h in hashtags_list_ru]
        hashtags_str_ru = " ".join(hashtags_list_ru)
        if hashtags_str_ru and hashtags_str_ru not in caption_ru:
            caption_ru = caption_ru.rstrip() + "\n\n" + hashtags_str_ru
    caption_ru = "Больше новостей в telegram-канале, ссылка в био\n\n" + caption_ru

    # --- RU YouTube title ---
    yt_title_ru = raw_title[:100]
    yt_desc_ru = caption_ru

    status_msg = await context.bot.send_message(
        chat_id=TELEGRAM_ADMIN_CHAT_ID,
        text=f"Uploading EN+RU to all platforms for post #{post_id}…",
    )

    from config import INSTAGRAM_USER_ID_RU, INSTAGRAM_ACCESS_TOKEN_RU
    tasks: dict[str, asyncio.Task] = {}
    if instagram_publisher.is_configured():
        tasks["Instagram"] = asyncio.create_task(
            instagram_publisher.publish_reel(video_path=video_path_en, caption=caption_en)
        )
    if youtube_publisher.is_configured():
        tasks["YouTube"] = asyncio.create_task(
            youtube_publisher.upload_short(video_path=video_path_en, title=yt_title_en, description=yt_desc_en, tags=tags_en)
        )
    if instagram_publisher.is_configured_ru():
        tasks["Instagram RU"] = asyncio.create_task(
            instagram_publisher.publish_reel(
                video_path=video_path_ru, caption=caption_ru,
                user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
            )
        )
    if youtube_publisher.is_configured_ru():
        tasks["YouTube RU"] = asyncio.create_task(
            youtube_publisher.upload_short_ru(
                video_path=video_path_ru, title=yt_title_ru, description=yt_desc_ru, tags=tags_ru,
            )
        )

    results: list[str] = []
    any_err = False
    for platform, task in tasks.items():
        try:
            result = await task
            if "YouTube" in platform and isinstance(result, str) and result:
                results.append(f"\u2705 {platform}: https://youtu.be/{result}")
            elif "Instagram" in platform and isinstance(result, str) and result:
                results.append(f"\u2705 {platform}: Media ID {result}")
            else:
                results.append(f"\u2705 {platform}: published")
        except Exception as exc:
            any_err = True
            results.append(f"\u274c {platform}: {exc}")
            logger.error("%s publish failed for post #%d: %s", platform, post_id, exc)

    await status_msg.edit_text(
        "\n".join(results) + "\n\n"
        + ("Video files kept for further publishing." if any_err else "Done!"),
        reply_markup=_build_video_done_keyboard(post_id) if any_err else _build_video_keyboard(post_id),
    )


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest:
        pass
    post_id = int(query.data.split(":")[1])

    post = db.get_scheduled_post(post_id)
    db.update_post_status(post_id, "cancelled")
    if post:
        _cleanup_media_files(post.get("image_paths", []), post.get("video_paths", []))
    await query.edit_message_text(f"Post #{post_id} has been cancelled.")


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors; ignore expired callback queries silently."""
    exc = context.error
    if isinstance(exc, BadRequest) and "query is too old" in str(exc).lower():
        logger.debug("Ignoring expired callback query: %s", exc)
        return
    logger.error("Unhandled exception: %s", exc, exc_info=exc)


# ---------------------------------------------------------------------------
# Build handlers list for Application
# ---------------------------------------------------------------------------

def build_handlers():
    return [
        CallbackQueryHandler(handle_approve,            pattern=r"^approve:\d+$"),
        CallbackQueryHandler(handle_cancel,             pattern=r"^cancel:\d+$"),
        CallbackQueryHandler(handle_create_video,        pattern=r"^create_video:\d+$"),
        CallbackQueryHandler(handle_create_video_en,     pattern=r"^create_video_en:\d+$"),
        CallbackQueryHandler(handle_create_video_ru,     pattern=r"^create_video_ru:\d+$"),
        CallbackQueryHandler(handle_post_instagram,      pattern=r"^post_instagram:\d+$"),
        CallbackQueryHandler(handle_post_instagram_ru,   pattern=r"^post_instagram_ru:\d+$"),
        CallbackQueryHandler(handle_post_youtube,        pattern=r"^post_youtube:\d+$"),
        CallbackQueryHandler(handle_post_youtube_ru,     pattern=r"^post_youtube_ru:\d+$"),
        CallbackQueryHandler(handle_post_all,            pattern=r"^post_all:\d+$"),
        CallbackQueryHandler(handle_post_all_ru,         pattern=r"^post_all_ru:\d+$"),
        CallbackQueryHandler(handle_post_all_combined,   pattern=r"^post_all_combined:\d+$"),
        # legacy alias kept for in-flight messages
        CallbackQueryHandler(handle_post_all,            pattern=r"^post_both:\d+$"),
    ]
