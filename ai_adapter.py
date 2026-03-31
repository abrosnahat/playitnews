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


async def is_gaming_related(title: str) -> bool:
    """
    Ask Ollama whether a news article title is about video games.
    Returns True  → process the article.
    Returns False → skip it.
    Falls back to True (fail-open) if Ollama is unavailable.
    """
    user_message = (
        "You are a strict gaming news filter. Answer with a single word only: YES or NO.\n\n"
        "Is the following Russian news article title about VIDEO GAMES?"
        " This includes: games, gaming industry, game consoles, esports, game engines, "
        "GPUs/CPUs/hardware specifically for gaming.\n"
        "Answer NO for: movies, TV shows, series, anime (unless a game adaptation), "
        "smartphones, tablets, TVs, or any non-gaming consumer electronics.\n\n"
        f"Title: {title}\n\n"
        "Answer (YES or NO):"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
        "options": {"num_predict": 5},
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                answer = data["message"]["content"].strip().upper()
                result = "YES" in answer
                logger.info("AI фильтр [%s]: '%s'", "+" if result else "-", title[:70])
                return result
    except Exception as exc:
        logger.warning("AI фильтр недоступен (%s), разрешаем статью: %s", exc, title[:70])
        return True  # fail-open: не блокируем если Ollama лежит


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


async def generate_video_script(post_text: str, article_title: str) -> str:
    """
    Generate a spoken narration script for TikTok/Reels/Shorts.
    Formula: [What happened] → [Short fact] → [Detail] → [Why it matters] → [Question/conclusion]
    Target: 65-90 words (~35-45 seconds spoken at natural pace).
    """
    # Strip HTML tags for clean input
    clean_text = re.sub(r"<[^>]+>", "", post_text)

    user_message = (
        "Write a short spoken video narration script for TikTok/Reels/Shorts based on the gaming news below.\n\n"
        "STRICT RULES:\n"
        "- Language: English ONLY\n"
        "- Length: 65–90 words MAXIMUM (critical — must fit in 35–45 seconds of speech)\n"
        "- Plain spoken text only — NO hashtags, NO HTML, NO emojis, NO markdown\n"
        "- The text will be read aloud by a voice AI, so write naturally spoken sentences\n"
        "- Follow this exact narrative formula with 1-2 sentences per step:\n"
        "  1. [What happened] — hook sentence that grabs attention immediately\n"
        "  2. [Short fact] — one key number, stat, or specific detail\n"
        "  3. [Detail] — one interesting or surprising detail from the news\n"
        "  4. [Why it matters] — why gamers should care about this\n"
        "  5. [Question or conclusion] — end with a question or strong closing line\n\n"
        "- Do NOT start with 'In today's news', 'Hey guys', or any generic opener\n"
        "- Start directly with the most exciting or surprising fact\n\n"
        f"Article title: {article_title}\n\n"
        f"Post content:\n{clean_text[:1800]}\n\n"
        "Write the script now (plain text, 65–90 words):"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a viral TikTok script writer specializing in gaming news. "
                    "You write punchy, engaging narration scripts that hook viewers in the first 3 seconds. "
                    "You always write in plain English, no formatting, no symbols — just natural spoken words."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"num_predict": 180},
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
                script = data["message"]["content"].strip()
                # Strip any residual markdown or HTML
                script = re.sub(r"<[^>]+>", "", script)
                script = re.sub(r"[*_`#]", "", script)
                script = script.strip()
                logger.info("Video script generated: %d words", len(script.split()))
                return script
    except Exception as exc:
        logger.error("Error generating video script: %s", exc)
        # Fallback: use first 350 chars of clean post text
        return clean_text[:350].strip()
