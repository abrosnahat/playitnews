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


def _sanitize_telegram_html(text: str) -> str:
    """Close any unclosed Telegram HTML tags (<b>, <i>) to prevent parse errors."""
    allowed = ("b", "i")
    stack: list[str] = []
    for m in re.finditer(r'<(/?)(\w+)[^>]*>', text):
        closing, tag = m.group(1), m.group(2).lower()
        if tag not in allowed:
            continue
        if closing:
            if stack and stack[-1] == tag:
                stack.pop()
        else:
            stack.append(tag)
    # Close remaining open tags in reverse order
    for tag in reversed(stack):
        text += f"</{tag}>"
    return text


# Telegram caption limit minus buffer for footer (\n\n@readitgames ~15 chars) and HTML tag overhead
_CAPTION_BODY_LIMIT = 900


def _trim_post_text(text: str, limit: int = _CAPTION_BODY_LIMIT) -> str:
    """
    Hard-trim generated post text so its total length (including HTML tags)
    stays within `limit` characters.  Trims at the last complete line that
    fits, then re-closes any open HTML tags.
    """
    if len(text) <= limit:
        return text
    # Try to trim at the last newline boundary that still fits
    cut = text[:limit]
    last_nl = cut.rfind("\n")
    if last_nl > limit // 2:
        cut = cut[:last_nl]
    # Re-sanitize to close any tags split by the cut
    return _sanitize_telegram_html(cut.rstrip())

SYSTEM_PROMPT = (
    "You are an expert gaming news editor writing for a Telegram channel. "
    "Transform Russian gaming news into punchy, engaging English Telegram posts."
)


async def adapt_article_ru(title: str, body: str) -> str:
    """
    Создаёт русскоязычный Telegram-пост из русской статьи.
    """
    user_message = (
        "Перепиши следующую игровую новость как пост для русскоязычного Telegram-канала.\n"
        "ВАЖНО: Весь ответ должен быть исключительно на русском языке.\n\n"
        "Структура поста (строго в таком порядке):\n"
        "1. <b>Заголовок</b> — цепляющий, с сутью новости. Можно добавить 1 эмодзи в конец заголовка.\n"
        "2. Лид — 1–2 строки: что произошло и почему важно.\n"
        "3. Детали списком через тире (—): что добавили/показали, дата, механики, платформы — только то что есть в статье.\n"
        "4. Реакции/факты (если есть): что заметили игроки, реакция комьюнити, утечки.\n"
        "5. Итог — 1 строка с ожиданиями или выводом.\n"
        "6. Пустая строка, затем 5–8 хэштегов через пробел.\n\n"
        "Правила:\n"
        "- Telegram HTML: <b>жирный</b> только для заголовка, <i>курсив</i> для одной ключевой детали\n"
        "- Максимум 750 символов включая теги (будет добавлен футер, итого не более 900)\n"
        "- Без упоминания источника\n\n"
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
        "options": {"num_predict": 1500, "num_ctx": 4096},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(data["message"]["content"].strip())))
                # Validate: must have more than just a headline + hashtags (>150 chars body)
                body_only = re.sub(r'<[^>]+>', '', post_text).strip()
                body_only = re.sub(r'#\S+', '', body_only).strip()
                if not post_text or len(body_only) < 100:
                    logger.warning("Gemma (RU) вернул только заголовок (%d симв.), повтор с упрощённым промптом", len(post_text))
                    raise ValueError("too short")
                logger.info("Gemma (RU) адаптировал статью: '%s' (%d симв.)", title[:60], len(post_text))
                return post_text
    except ValueError:
        # Retry with a simpler prompt that the model can't misinterpret
        simple_prompt = (
            f"Напиши Telegram-пост об этой игровой новости на русском языке.\n"
            f"Начни с <b>заголовка</b>, затем 3–4 предложения о сути новости, затем хэштеги.\n\n"
            f"Заголовок: {title}\n\nТекст: {body[:2000]}\n\nПост:"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": simple_prompt}],
                          "stream": False, "options": {"num_predict": 900, "num_ctx": 4096}},
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(data["message"]["content"].strip())))
                    if post_text:
                        logger.info("Gemma (RU) retry OK: '%s' (%d симв.)", title[:60], len(post_text))
                        return post_text
        except Exception as exc2:
            logger.error("Gemma (RU) retry failed: %s", exc2)
        return f"<b>{title}</b>\n\n#игры #новости #gaming"


async def adapt_article(title: str, body: str) -> str:
    """
    Translate, rephrase, and adapt the Russian article into an English Telegram post.
    Uses local Ollama — no API key required.
    """
    user_message = (
        "Transform the following Russian gaming news into an English Telegram post.\n"
        "IMPORTANT: Your entire response must be in English only.\n\n"
        "Post structure (follow this order):\n"
        "1. <b>Headline</b> — punchy, captures the news. You may add 1 emoji at the end of the headline.\n"
        "2. Lead — 1–2 lines: what happened and why it matters.\n"
        "3. Details as a bullet list with em-dashes (—): what was shown/added, release date, mechanics, platforms — only facts from the article.\n"
        "4. Reactions/facts (if available): what fans noticed, community reaction, leaks.\n"
        "5. Conclusion — 1 line with takeaway or expectations.\n"
        "6. Blank line, then 5–8 hashtags separated by spaces.\n\n"
        "Rules:\n"
        "- Telegram HTML: <b>bold</b> for headline only, <i>italic</i> for one key detail\n"
        "- Maximum 750 characters total including tags (a footer will be appended, total must stay under 900)\n"
        "- No emojis except in the headline. Do not mention the source website.\n\n"
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
        "options": {"num_predict": 1500, "num_ctx": 4096},
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
                post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(data["message"]["content"].strip())))
                # Validate: must have more than just a headline + hashtags (>150 chars body)
                body_only = re.sub(r'<[^>]+>', '', post_text).strip()
                body_only = re.sub(r'#\S+', '', body_only).strip()
                if not post_text or len(body_only) < 100:
                    logger.warning("Gemma (EN) вернул только заголовок (%d симв.), повтор с упрощённым промптом", len(post_text))
                    raise ValueError("too short")
                logger.info("Gemma адаптировал статью: '%s'  (%d симв.)", title[:60], len(post_text))
                return post_text
    except ValueError:
        simple_prompt = (
            f"Write a Telegram post in English about this gaming news.\n"
            f"Start with <b>headline in bold</b>, then 3–4 sentences about the news, then hashtags.\n\n"
            f"Title: {title}\n\nBody: {body[:2000]}\n\nPost:"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": simple_prompt}],
                          "stream": False, "options": {"num_predict": 1500, "num_ctx": 4096}},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(data["message"]["content"].strip())))
                    if post_text:
                        logger.info("Gemma (EN) retry OK: '%s' (%d симв.)", title[:60], len(post_text))
                        return post_text
        except Exception as exc2:
            logger.error("Gemma (EN) retry failed: %s", exc2)
        return f"{title}\n\n[AI processing failed. Please edit before publishing.]\n\n#gaming #news"


# Keywords that guarantee a title is gaming-related — bypass LLM entirely
_GAMING_KEYWORDS = [
    # Игровые термины (RU)
    "игр", "геймпл", "патч", "обновлени", "длс", "dlc", "моддинг", "мод ",
    "датамайн", "катсцен", "геймер", "игрок", "игровой", "игровая",
    "консол", "приставк", "релиз", "анонс", "трейлер", "геймплей",
    "разработчик", "издател", "студи", "esports", "киберспорт",
    "стример", "стрим",
    "ремейк", "ремастер", "сиквел", "приквел", "спин-офф", "аддон",
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
    "nintendo", "playstation", "xbox", "steam", "epic games",
    "black flag", "black ops", "modern warfare", "ghost recon",
    "far cry", "watch dogs", "rainbow six", "division", "crew",
    "remake", "remaster", "expansion",
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        raw = await _call_ollama_chat(messages, num_predict=1500, timeout=120)
        shortened = _sanitize_telegram_html(_md_to_html(raw))
        if len(shortened) < 50:
            logger.warning("shorten_post returned too-short result (%d chars), hard-truncating", len(shortened))
            return _hard_truncate(text, target_chars)
        logger.info("Gemma сократил пост: %d → %d симв.", len(text), len(shortened))
        return shortened
    except Exception as exc:
        logger.error("Ошибка Ollama shorten API: %s", exc)
        return _hard_truncate(text, target_chars)


def _hard_truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars at a word boundary, closing any open HTML tags."""
    if len(text) <= limit:
        return text
    truncated = text[:limit - 1].rstrip()
    # Don't cut mid-tag
    last_open = truncated.rfind("<")
    if last_open != -1 and ">" not in truncated[last_open:]:
        truncated = truncated[:last_open].rstrip()
    return _sanitize_telegram_html(truncated)


async def _call_ollama_chat(
    messages: list[dict],
    *,
    num_predict: int = 100,
    num_ctx: int = 2048,
    timeout: int = 60,
) -> str:
    """Send a chat request to Ollama and return the model's raw content string.
    Raises on network / HTTP errors so callers can handle them individually."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["message"]["content"].strip()


async def extract_game_name(article_title: str) -> str:
    """
    Ask Ollama to extract a concise YouTube search query from the article headline.
    For game news: returns the game title (e.g. "GTA 6", "Elden Ring 2").
    For other gaming/tech news: returns a short descriptive query suitable for
    finding relevant YouTube footage (e.g. "PC gaming setup", "game awards ceremony").
    Returns an English query string, or empty string on failure.
    """
    user_message = (
        "Given the following news headline, write a short YouTube search query (2-5 words in English) "
        "that would find relevant video footage for this story.\n\n"
        "PRIORITY RULES (follow in order):\n"
        "1. If the headline mentions a specific game title — return ONLY the game title (e.g. 'GTA 6', 'Silent Hill 2', 'Elden Ring'). "
        "Ignore any journalist names, company names, or studio names — the game title is always the priority.\n"
        "2. If there is no specific game title but there are gaming/tech topics — return a short descriptive query.\n"
        "3. Return ONLY the search query, nothing else — no punctuation, no explanation, no names of people or studios.\n\n"
        f"Headline: {article_title}\n\n"
        "Search query:"
    )
    try:
        name = (await _call_ollama_chat(
            [{"role": "user", "content": user_message}], num_predict=20, timeout=30
        )).strip('"\'')
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
    try:
        translated = (await _call_ollama_chat(
            [{"role": "user", "content": user_message}], num_predict=40, timeout=30
        )).strip('"\'')
        if translated and len(translated) <= 100:
            logger.info("Translated title: '%s' ← '%s'", translated, article_title[:60])
            return translated
    except Exception as exc:
        logger.warning("translate_title_to_english failed: %s", exc)
    return article_title


async def generate_thumbnail_hook(article_title: str, lang: str = "ru") -> str:
    """
    Generate a short, punchy, clickable thumbnail caption (2–3 words, ALL CAPS).
    lang: "ru" → Russian output, "en" → English output.
    Falls back to the original title on failure.
    """
    lang_instruction = (
        "Write the caption in RUSSIAN only. Use Cyrillic letters."
        if lang == "ru" else
        "Write the caption in ENGLISH only."
    )
    user_message = (
        "You are writing text for a gaming news video thumbnail.\n"
        "Create a SHORT, PUNCHY caption (2–3 words) that creates curiosity or urgency "
        "and fits the topic of the news headline below.\n"
        "Rules:\n"
        "- Maximum 3 words\n"
        f"- {lang_instruction}\n"
        "- No punctuation except ! or ?\n"
        "- Write in normal case (the system will uppercase it automatically)\n"
        "- Return ONLY the caption text, nothing else\n\n"
        f"Headline: {article_title}\n\n"
        "Caption:"
    )
    try:
        hook = (await _call_ollama_chat(
            [{"role": "user", "content": user_message}], num_predict=20, timeout=30
        )).strip('"\'').upper()
        if hook and len(hook) <= 60 and "\n" not in hook:
            logger.info("Thumbnail hook: '%s' ← '%s'", hook, article_title[:60])
            return hook
    except Exception as exc:
        logger.warning("generate_thumbnail_hook failed: %s", exc)
    return article_title


async def generate_video_script(post_text: str, article_title: str, lang: str = "en") -> str:
    """
    Generate a spoken narration script for TikTok/Reels/Shorts.
    lang='en' → English script; lang='ru' → Russian script.
    Target: 65–90 words (~35–45 seconds spoken at natural pace).
    """
    # Strip HTML, URLs, markdown and emoji from input text
    clean_text = re.sub(r"<[^>]+>", "", post_text)
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    clean_text = re.sub(r"[*_`#]", "", clean_text)
    clean_text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FEFF]", "", clean_text)
    clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

    if lang == "ru":
        system_content = (
            "Ты профессиональный редактор игровых новостей. "
            "Пишешь нейтральные, грамотные сценарии озвучки на русском литературном языке. "
            "Без сленга, мата и фамильярности. Только чистый информационный текст."
        )
        user_message = (
            "Напиши короткий сценарий озвучки для игровой новости (формат Shorts/Reels).\n\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            "- Язык: ТОЛЬКО русский\n"
            "- Длина: 65–90 слов МАКСИМУМ (35–45 секунд речи)\n"
            "- Только чистый текст — БЕЗ хэштегов, HTML, эмодзи, markdown\n"
            "- Стиль: нейтральный, информационный, без сленга, жаргона и бранных слов\n"
            "- Текст будет зачитан голосовым AI — пиши чёткими литературными предложениями\n"
            "- Следуй формуле (1-2 предложения на шаг):\n"
            "  1. [Что произошло] — цепляющее вступительное предложение\n"
            "  2. [Факт] — одна ключевая цифра, деталь или статистика\n"
            "  3. [Деталь] — интересная или неожиданная подробность\n"
            "  4. [Почему важно] — значимость для игрового сообщества\n"
            "  5. [Вопрос или вывод] — завершающий вопрос или ёмкая фраза\n\n"
            "- НЕ начинай с 'Сегодня в новостях', 'Привет всем' или шаблонных фраз\n"
            "- Начинай сразу с главного факта\n"
            "- ЗАПРЕЩЕНО: мат, сленг, грубые выражения, фамильярное обращение\n\n"
            f"Заголовок: {article_title}\n\n"
            f"Текст поста:\n{clean_text[:1800]}\n\n"
            "Напиши сценарий (нейтральный литературный текст, 65–90 слов):"
        )
    else:
        system_content = (
            "You are a viral TikTok script writer specializing in gaming news. "
            "You write punchy, engaging narration scripts that hook viewers in the first 3 seconds. "
            "You always write in plain English, no formatting, no symbols — just natural spoken words."
        )
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

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]
    try:
        script = await _call_ollama_chat(messages, num_predict=1500, num_ctx=2048, timeout=120)
        script = re.sub(r"<[^>]+>", "", script)
        script = re.sub(r"[*_`#]", "", script)
        script = script.strip()
        logger.info("%s video script generated: %d words", lang.upper(), len(script.split()))
        return script
    except Exception as exc:
        logger.error("Error generating %s video script: %s", lang.upper(), exc)
        return clean_text[:350].strip()
