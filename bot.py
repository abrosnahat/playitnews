"""
Telegram bot — notification-only mode.
Sends a brief admin notification when a new post is created.
All management is done via the web dashboard.
"""
import logging
import subprocess
from typing import Optional

from telegram import Bot
from telegram.error import TelegramError
from telegram.ext import ContextTypes

import database as db
from config import TELEGRAM_ADMIN_CHAT_ID

logger = logging.getLogger(__name__)

DASHBOARD_LOCAL = "http://localhost:5001"


def _get_cloudflare_url() -> Optional[str]:
    """Read the current cloudflared tunnel URL from the log file."""
    try:
        result = subprocess.run(
            ["grep", "-o", "https://[a-z0-9-]*\\.trycloudflare\\.com",
             "/tmp/cloudflared_playitnews.log"],
            capture_output=True, text=True, timeout=3,
        )
        url = result.stdout.strip().split("\n")[0]
        return url or None
    except Exception:
        return None


async def send_admin_notification(
    bot: Bot,
    post_id: int,
    article_title: str,
    article_url: str,
    **kwargs,
) -> Optional[int]:
    """Send a short notification to the admin about a new post."""
    tunnel_url = _get_cloudflare_url()

    lines = [f"📰 Post #{post_id} — {article_title}"]
    if article_url:
        lines.append(f"🔗 {article_url}")
    lines.append("")
    if tunnel_url:
        lines.append(f"🌐 Dashboard: {tunnel_url}")
    else:
        lines.append(f"💻 Dashboard (local): {DASHBOARD_LOCAL}")

    text = "\n".join(lines)
    try:
        msg = await bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=text,
            disable_web_page_preview=True,
        )
        db.set_notification_message_id(post_id, msg.message_id)
        return msg.message_id
    except TelegramError as exc:
        logger.error("Не удалось отправить уведомление: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stubs used by main.py / webapp.py
# ---------------------------------------------------------------------------

def build_handlers():
    return []


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


async def publish_post(bot: Bot, post_id: int) -> bool:
    """Stub — publishing is handled via the web dashboard."""
    logger.warning("publish_post called for #%d — use dashboard instead", post_id)
    return False
