import re
import logging
import aiohttp
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def _md_to_html(text: str) -> str:
    """Convert Markdown bold/italic to Telegram HTML tags."""
    # **bold** / __bold__  →  <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__',     r'<b>\1</b>', text, flags=re.DOTALL)
    # *italic* / _italic_  →  <i>italic</i>  (single star/underscore)
    text = re.sub(r'\*(?!\*)(.+?)(?<!\*)\*', r'<i>\1</i>', text, flags=re.DOTALL)
    text = re.sub(r'_(?!_)(.+?)(?<!_)_',    r'<i>\1</i>', text, flags=re.DOTALL)
    return text

SYSTEM_PROMPT = (
    "You are an expert gaming news editor and social media content creator. "
    "Your task is to transform Russian gaming news articles into engaging English Telegram posts.\n\n"
    "Rules:\n"
    "- Translate from Russian to English accurately.\n"
    "- Rephrase in a dynamic, engaging tone suited for a gaming audience.\n"
    "- Use Telegram HTML formatting: <b>bold</b> for the headline, <i>italic</i> for key details.\n"
    "- Keep it concise: 2-3 short paragraphs.\n"
    "- Add 5-8 relevant English hashtags at the end.\n"
    "- Do NOT include any emojis.\n"
    "- Do NOT mention playground.ru or the source.\n"
    "- Start directly with the news hook.\n"
    "- End with the hashtag block on a new line."
)


async def adapt_article(title: str, body: str) -> str:
    """
    Translate, rephrase, and adapt the Russian article into an English Telegram post.
    Uses local Ollama (llama3.2) — no API key required.
    """
    user_message = (
        "Transform the following Russian gaming news article into a Telegram post.\n"
        "IMPORTANT: Your entire response must be in English only.\n\n"
        "Format rules (use Telegram HTML):\n"
        "- First line: <b>catchy headline in bold</b>\n"
        "- Then 2-3 short paragraphs of plain text (no tags)\n"
        "- If there is a notable quote or key stat, wrap it in <i>italic</i>\n"
        "- Last line: 5-8 hashtags separated by spaces, NO tags around them\n"
        "- One blank line between the text and the hashtag line\n"
        "- Maximum 900 characters total including HTML tags\n"
        "- No emojis, do not mention the source website\n\n"
        "Example structure:\n"
        "<b>Headline Here</b>\n\n"
        "Paragraph one with the main news.\n\n"
        "Paragraph two with details. <i>Key quote or stat if any.</i>\n\n"
        "#tag1 #tag2 #tag3\n\n"
        f"Title: {title}\n\nBody:\n{body[:4000]}\n\n"
        "Write the Telegram post now:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"num_predict": 400},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                post_text = _md_to_html(data["message"]["content"].strip())
                logger.info("Gemma адаптировал статью: '%s'  (%d симв.)", title[:60], len(post_text))
                return post_text
    except Exception as exc:
        logger.error("Ошибка Ollama API: %s", exc)
        return f"{title}\n\n[AI processing failed. Please edit before publishing.]\n\n#gaming #news"


async def shorten_post(text: str, target_chars: int = 900) -> str:
    """Rewrite the post using fewer words to fit within target_chars."""
    user_message = (
        f"The following Telegram gaming news post is too long. "
        f"Rewrite it so the total length is under {target_chars} characters, "
        f"using fewer words in the paragraphs. "
        f"Preserve the exact structure: <b>headline</b>, short paragraphs, optional <i>italic</i>, "
        f"and the hashtag line at the end unchanged. Keep the same HTML tags. Output only the post.\n\n"
        f"{text}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"num_predict": 350},
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                shortened = _md_to_html(data["message"]["content"].strip())
                logger.info("Gemma сократил пост: %d → %d симв.", len(text), len(shortened))
                return shortened
    except Exception as exc:
        logger.error("Ошибка Ollama shorten API: %s", exc)
        return text  # fallback: return original
