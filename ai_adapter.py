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


async def adapt_article_ru(title: str, body: str) -> str:
    """
    Создаёт русскоязычный Telegram-пост из русской статьи.
    """
    user_message = (
        "Перепиши следующую игровую новость как пост для русскоязычного Telegram-канала.\n"
        "ВАЖНО: Весь ответ должен быть исключительно на русском языке.\n\n"
        "Правила форматирования (Telegram HTML):\n"
        "- Первая строка: <b>цепляющий заголовок жирным</b>\n"
        "- Затем 2-3 коротких абзаца обычного текста\n"
        "- Яркую цитату или ключевой факт оберни в <i>курсив</i>\n"
        "- Последняя строка: 5-8 русских и английских хэштегов через пробел\n"
        "- Пустая строка между текстом и хэштегами\n"
        "- Максимум 900 символов включая HTML-теги\n"
        "- Без эмодзи, не упоминай источник\n\n"
        "Пример структуры:\n"
        "<b>Заголовок</b>\n\n"
        "Первый абзац с главной новостью.\n\n"
        "Второй абзац с деталями. <i>Ключевая цитата если есть.</i>\n\n"
        "#тег1 #тег2 #gaming\n\n"
        f"Заголовок: {title}\n\nТекст:\n{body[:4000]}\n\n"
        "Напиши Telegram-пост на русском:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": (
                "Ты опытный редактор игровых новостей. Пишешь живые, увлекательные посты "
                "для русскоязычной аудитории. Только русский язык в ответе."
            )},
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
                logger.info("Gemma (RU) адаптировал статью: '%s' (%d симв.)", title[:60], len(post_text))
                return post_text
    except Exception as exc:
        logger.error("Ошибка Ollama API (RU): %s", exc)
        return f"{title}\n\n[Ошибка AI обработки]\n\n#игры #новости"


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


# Keywords that guarantee a title is gaming-related — bypass LLM entirely
_GAMING_KEYWORDS = [
    # Игровые термины (RU)
    "игр", "геймпл", "патч", "обновлени", "длс", "dlc", "моддинг", "мод ",
    "датамайн", "катсцен", "геймер", "игрок", "игровой", "игровая",
    "консол", "приставк", "релиз", "анонс", "трейлер", "геймплей",
    "разработчик", "издател", "студи", "esports", "киберспорт",
    "стример", "стрим",
    # Game franchise / engine names (EN, recognizable in RU titles)
    "fallout", "elder scrolls", "elden ring", "stalker", "s.t.a.l.k.e.r",
    "cyberpunk", "witcher", "assassin", "call of duty", "battlefield",
    "resident evil", "silent hill", "final fantasy", "grand theft",
    "gta", "red dead", "halo", "forza", "minecraft", "roblox",
    "baldur", "diablo", "starcraft", "warcraft", "world of warcraft",
    "overwatch", "counter-strike", "half-life", "portal", "dota",
    "league of legends", "valorant", "fortnite", "apex legends",
    "death stranding", "god of war", "horizon", "spider-man", "marvel",
    "souls", "bloodborne", "sekiro", "unreal engine", "re engine",
    "nintendo", "playstation", "xbox", "steam ", "epic games",
    "zompiercer", "remake", "remaster", "expansion",
]


async def is_gaming_related(title: str) -> bool:
    """
    Ask Ollama whether a news article title is about video games.
    Returns True  → process the article.
    Returns False → skip it.
    Falls back to True (fail-open) if Ollama is unavailable.
    """
    title_lower = title.lower()
    for kw in _GAMING_KEYWORDS:
        if kw in title_lower:
            logger.info("AI фильтр [+] (keyword '%s'): '%s'", kw.strip(), title[:70])
            return True

    user_message = (
        "You are a gaming news filter for a Russian gaming news site. Answer with YES or NO only.\n\n"
        "Answer YES if the title is about:\n"
        "- Any video game (known or unknown), game update, patch, DLC, expansion\n"
        "- Game development news, game engines, studios, publishers\n"
        "- Game consoles, gaming hardware, peripherals\n"
        "- Esports, streamers, game mods\n"
        "- ANY title that could plausibly be a video game name\n\n"
        "When in doubt — answer YES. Only answer NO if the title is clearly and obviously "
        "about something unrelated to gaming: a non-gaming movie, TV series, smartphone, "
        "tablet, or non-gaming consumer product.\n\n"
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


async def extract_game_name(article_title: str) -> str:
    """
    Ask Ollama to extract just the game title (including part/edition number)
    from a Russian or English article headline, for use in YouTube search.
    Returns an English game name string, or empty string on failure.

    Examples:
      "GTA 6 вступила в решающий этап" → "GTA 6"
      "Forza Horizon 6: легендарная Acura" → "Forza Horizon 6"
      "Elden Ring 2 анонсирован" → "Elden Ring 2"
    """
    user_message = (
        "Extract only the video game title (including its part/sequel number or subtitle "
        "if present) from the following news headline. "
        "Return ONLY the game name in English, nothing else — no punctuation, no explanation.\n\n"
        f"Headline: {article_title}\n\n"
        "Game title:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
        "options": {"num_predict": 20},
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
                name = data["message"]["content"].strip().strip('"\'')
                # Sanity check: must be non-empty and reasonably short
                if name and len(name) <= 60:
                    logger.info("AI game name: '%s' ← '%s'", name, article_title[:60])
                    return name
    except Exception as exc:
        logger.warning("extract_game_name failed: %s", exc)
    return ""


async def translate_title_to_english(article_title: str) -> str:
    """
    Translate a Russian (or mixed) article headline to a concise English YouTube title.
    Returns the English title, or the original string on failure.
    """
    user_message = (
        "Translate the following gaming news headline into English. "
        "Keep it concise (max 80 characters), catchy, and suitable as a YouTube video title. "
        "Return ONLY the translated title, nothing else.\n\n"
        f"Headline: {article_title}\n\n"
        "English title:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
        "options": {"num_predict": 40},
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
                translated = data["message"]["content"].strip().strip('"\'')
                if translated and len(translated) <= 100:
                    logger.info("Translated title: '%s' ← '%s'", translated, article_title[:60])
                    return translated
    except Exception as exc:
        logger.warning("translate_title_to_english failed: %s", exc)
    return article_title


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
