import re
import json
import asyncio
import logging
import aiohttp
from config import (
    GEMINI_API_KEY,
    GEMINI_TEXT_MODEL,
    GEMINI_VIDEO_MODEL,
)
import gemini_keys

logger = logging.getLogger(__name__)


class _GeminiQuotaExceeded(Exception):
    """Raised when a Gemini API call returns 429 / RESOURCE_EXHAUSTED."""


async def _gemini_post_with_retry(url: str, body: dict, headers: dict, timeout: int) -> dict:
    """POST to the Gemini API with up to 3 attempts on 5xx errors.

    Raises ``_GeminiQuotaExceeded`` immediately (no retry) on 429 /
    RESOURCE_EXHAUSTED — the caller handles that by rotating API keys.
    """
    last_exc: Exception | None = None
    for attempt in range(1, 4):   # up to 3 attempts
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429:
                        text_body = await resp.text()
                        raise _GeminiQuotaExceeded(text_body[:500])
                    if resp.status >= 500 and attempt < 3:
                        last_exc = Exception(f"HTTP {resp.status}")
                        logger.warning("Gemini API %s (attempt %d/3), retrying…", resp.status, attempt)
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientResponseError as exc:
            if exc.status == 429:
                raise _GeminiQuotaExceeded(str(exc)) from exc
            if exc.status >= 500 and attempt < 3:
                last_exc = exc
                logger.warning("Gemini API %s (attempt %d/3), retrying…", exc.status, attempt)
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise last_exc  # type: ignore[misc]


async def _gemini_chat(payload: dict, timeout: int) -> dict:
    """Call Google's Generative Language API with a chat-style ``payload``
    (``{"messages": [...], "model": ..., "options": {...}}``).

    Returns a normalized dict ``{"message": {"content": str}}``.
    Retries up to 3 times on 5xx errors (gemma-4 preview is occasionally flaky).
    """
    messages = payload.get("messages", [])
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        g_role = "model" if role == "assistant" else "user"
        contents.append({"role": g_role, "parts": [{"text": m.get("content", "")}]})

    model = payload.get("model") or GEMINI_TEXT_MODEL
    body: dict = {"contents": contents}
    if system_parts:
        system_text = "\n\n".join(system_parts)
        # Gemma models (served via the same API) don't support systemInstruction —
        # fold the system prompt into the first user turn instead.
        if model.lower().startswith("gemma"):
            if contents and contents[0].get("parts"):
                first = contents[0]["parts"][0]
                first["text"] = f"{system_text}\n\n{first.get('text', '')}"
            else:
                contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})
        else:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
    opts = payload.get("options") or {}
    num_predict = opts.get("num_predict")
    if isinstance(num_predict, int) and num_predict > 0:
        body["generationConfig"] = {"maxOutputTokens": num_predict}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    max_key_attempts = max(gemini_keys.key_count(), 1)
    data: dict | None = None
    last_exc: Exception | None = None
    for key_attempt in range(1, max_key_attempts + 1):
        api_key = gemini_keys.get_current_key() or GEMINI_API_KEY
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        try:
            data = await _gemini_post_with_retry(url, body, headers, timeout)
            break
        except _GeminiQuotaExceeded as exc:
            last_exc = exc
            if key_attempt < max_key_attempts:
                logger.warning(
                    "Gemini API quota exceeded on key #%d/%d — rotating to next key",
                    key_attempt, max_key_attempts,
                )
                gemini_keys.rotate_key(reason="429 RESOURCE_EXHAUSTED")
                continue
            raise RuntimeError(
                f"All {max_key_attempts} Gemini API key(s) exhausted quota: {exc}"
            ) from exc
    if data is None:
        raise last_exc or RuntimeError("Gemini API request failed")

    try:
        parts = data["candidates"][0]["content"]["parts"]
        # Gemma/Gemini "thinking" models return reasoning parts flagged
        # ``"thought": true`` alongside the final answer — keep only the answer.
        text = "".join(
            p.get("text", "") for p in parts if not p.get("thought")
        )
    except (KeyError, IndexError):
        text = ""
    return {"message": {"content": text}}


def _render_prompt(template: str, **tokens) -> str:
    """Substitute {token} placeholders in a project-supplied prompt template."""
    out = template
    for k, v in tokens.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _prompt_parts(cfg, lang: str | None = None):
    """Normalize an AI-prompt override into ``(system, user)`` strings.

    Accepts a bare string (→ user), a ``{"system", "user"}`` dict, or a
    per-language ``{"en": {...}, "ru": {...}}`` mapping. Returns
    ``(None, None)`` when nothing usable is supplied so callers fall back to
    their built-in default prompts.
    """
    if cfg is None:
        return None, None
    if isinstance(cfg, str):
        return None, cfg
    if isinstance(cfg, dict):
        if lang and isinstance(cfg.get(lang), (str, dict)):
            return _prompt_parts(cfg[lang])
        return cfg.get("system"), cfg.get("user")
    return None, None



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


async def adapt_article_ru(title: str, body: str, prompt=None) -> str:
    """
    Создаёт русскоязычный Telegram-пост из русской статьи.

    ``prompt`` — необязательный переопределяющий промпт проекта (из projects.json).
    """
    _sys, _user = _prompt_parts(prompt)
    default_system = (
        "Ты опытный редактор игровых новостей. Пишешь живые, увлекательные посты "
        "для русскоязычной аудитории. Только русский язык в ответе."
    )
    if _user:
        user_message = _render_prompt(_user, title=title, body=body[:4000])
    else:
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

    messages = [
        {"role": "system", "content": (_sys or default_system)},
        {"role": "user", "content": user_message},
    ]

    try:
        content = await _call_llm_chat(messages, num_predict=8000, num_ctx=4096, timeout=300)
        post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(content)))
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
            content = await _call_llm_chat(
                [{"role": "user", "content": simple_prompt}],
                num_predict=6000, num_ctx=4096, timeout=300,
            )
            post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(content)))
            if post_text:
                logger.info("Gemma (RU) retry OK: '%s' (%d симв.)", title[:60], len(post_text))
                return post_text
        except Exception as exc2:
            logger.error("Gemma (RU) retry failed: %s", exc2)
        return f"<b>{title}</b>\n\n#игры #новости #gaming"


async def adapt_article(title: str, body: str, prompt=None) -> str:
    """
    Translate, rephrase, and adapt the Russian article into an English Telegram post.
    Uses the Gemini API (requires ``GEMINI_API_KEY``).

    ``prompt`` — optional per-project prompt override (from projects.json).
    """
    _sys, _user = _prompt_parts(prompt)
    if _user:
        user_message = _render_prompt(_user, title=title, body=body[:4000])
    else:
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

    messages = [
        {"role": "system", "content": (_sys or SYSTEM_PROMPT)},
        {"role": "user", "content": user_message},
    ]

    try:
        content = await _call_llm_chat(messages, num_predict=8000, num_ctx=4096, timeout=300)
        post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(content)))
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
            content = await _call_llm_chat(
                [{"role": "user", "content": simple_prompt}],
                num_predict=8000, num_ctx=4096, timeout=300,
            )
            post_text = _trim_post_text(_sanitize_telegram_html(_md_to_html(content)))
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


async def is_gaming_related(title: str, prompt=None) -> bool:
    """
    Ask Gemini whether a news article title is about video games.
    Returns True  → process the article.
    Returns False → skip it.
    Falls back to True (fail-open) if Gemini is unavailable.

    ``prompt`` — optional per-project prompt override (from projects.json).
    """
    title_lower = title.lower()
    for kw in _GAMING_KEYWORDS:
        if kw in title_lower:
            logger.info("AI фильтр [+] (keyword '%s'): '%s'", kw.strip(), title[:70])
            return True

    _sys, _user = _prompt_parts(prompt)
    if _user:
        user_message = _render_prompt(_user, title=title)
    else:
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
    _messages = [{"role": "user", "content": user_message}]
    if _sys:
        _messages.insert(0, {"role": "system", "content": _sys})
    try:
        answer = (await _call_llm_chat(
            _messages, num_predict=2000, num_ctx=2048, timeout=120
        )).upper()
        result = "YES" in answer
        logger.info("AI фильтр [%s]: '%s'", "+" if result else "-", title[:70])
        return result
    except Exception as exc:
        logger.warning("AI фильтр недоступен (%s), разрешаем статью: %s", exc, title[:70])
        return True  # fail-open: не блокируем если Gemini недоступен


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
        raw = await _call_llm_chat(messages, num_predict=8000, timeout=300)
        shortened = _sanitize_telegram_html(_md_to_html(raw))
        if len(shortened) < 50:
            logger.warning("shorten_post returned too-short result (%d chars), hard-truncating", len(shortened))
            return _hard_truncate(text, target_chars)
        logger.info("Gemma сократил пост: %d → %d симв.", len(text), len(shortened))
        return shortened
    except Exception as exc:
        logger.error("Ошибка Gemini shorten API: %s", exc)
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


async def _call_llm_chat(
    messages: list[dict],
    *,
    num_predict: int = 100,
    num_ctx: int = 16384,
    timeout: int = 60,
    model: str | None = None,
) -> str:
    """Send a chat request to Gemini and return the model's raw content string.
    Raises on network / HTTP errors so callers can handle them individually.

    ``model`` defaults to ``GEMINI_TEXT_MODEL``; pass ``GEMINI_VIDEO_MODEL``
    for the heavier reasoning-grade calls (e.g. video script generation).
    """
    payload = {
        "model": model or GEMINI_TEXT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }
    data = await _gemini_chat(payload, timeout=timeout)
    return data["message"]["content"].strip()


async def extract_game_name(article_title: str) -> str:
    """
    Ask Gemini to extract a concise YouTube search query from the article headline.
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
        name = (await _call_llm_chat(
            [{"role": "user", "content": user_message}], num_predict=2000, timeout=120
        )).strip('"\'')
        if name and len(name) <= 60:
            logger.info("AI game name: '%s' ← '%s'", name, article_title[:60])
            return name
    except Exception as exc:
        logger.warning("extract_game_name failed: %s", exc)
    return ""


async def extract_fighter_query(article_title: str) -> str:
    """
    Ask Gemini to extract the MMA/UFC fighter name(s) from a Russian headline
    and build a Russian-language YouTube search query for their fight footage.

    Returns a Russian query such as "Махачев против Гарри бой" or
    "Волков лучшие моменты", or an empty string on failure.
    """
    user_message = (
        "Дан заголовок новости о UFC/MMA на русском языке. Определи имя (или имена) "
        "бойцов, о которых идёт речь, и составь короткий поисковый запрос для YouTube "
        "на русском языке, чтобы найти видео их боёв.\n\n"
        "ПРАВИЛА (по порядку):\n"
        "1. Оставляй имена бойцов на русском языке — НЕ переводи и НЕ транслитерируй.\n"
        "2. Если в заголовке двое бойцов-соперников, верни '<Боец A> против <Боец B> бой'.\n"
        "3. Если упомянут только один боец, верни '<Боец> лучшие моменты'.\n"
        "4. Используй фамилию бойца (или полное имя, если оно известно).\n"
        "5. Верни ТОЛЬКО поисковый запрос на русском — без пояснений и лишних знаков.\n\n"
        f"Заголовок: {article_title}\n\n"
        "Поисковый запрос:"
    )
    try:
        query = (await _call_llm_chat(
            [{"role": "user", "content": user_message}], num_predict=2000, timeout=120
        )).strip('"\'')
        if query and len(query) <= 80:
            logger.info("AI fighter query: '%s' ← '%s'", query, article_title[:60])
            return query
    except Exception as exc:
        logger.warning("extract_fighter_query failed: %s", exc)
    return ""


async def extract_search_query(article_title: str, prompt=None) -> str:
    """
    Build a YouTube search query from a headline using a per-project prompt.

    ``prompt`` — optional prompt override (from projects.json). When absent,
    falls back to the default game-title extraction (gaming project behaviour).
    Returns the query string, or an empty string on failure.
    """
    _sys, _user = _prompt_parts(prompt)
    if not _user:
        # No project override → default gaming behaviour.
        return await extract_game_name(article_title)
    user_message = _render_prompt(_user, title=article_title)
    _messages = [{"role": "user", "content": user_message}]
    if _sys:
        _messages.insert(0, {"role": "system", "content": _sys})
    try:
        query = (await _call_llm_chat(
            _messages, num_predict=2000, timeout=120
        )).strip('"\'')
        if query and len(query) <= 80:
            logger.info("AI search query: '%s' ← '%s'", query, article_title[:60])
            return query
    except Exception as exc:
        logger.warning("extract_search_query failed: %s", exc)
    return ""


async def extract_entity_queries(script: str, prompt=None) -> list[str]:
    """
    Extract multiple short YouTube search queries — one per distinct person or
    subject actually mentioned in the narration *script* — so B-roll footage
    can be fetched to match what's being said at each point, instead of a
    single query derived only from the headline.

    ``prompt`` — required per-project override (projects.json → ai.entity_queries).
    Requires a ``{script}`` token. When absent, this feature is a no-op
    (returns an empty list) — callers should gate on the project config being
    present before calling this at all.
    Returns a list of query strings (may be empty on failure/no override).
    """
    _sys, _user = _prompt_parts(prompt)
    if not _user:
        return []
    user_message = _render_prompt(_user, script=script)
    _messages = [{"role": "user", "content": user_message}]
    if _sys:
        _messages.insert(0, {"role": "system", "content": _sys})
    try:
        raw = (await _call_llm_chat(_messages, num_predict=2000, timeout=120)).strip()
        queries: list[str] = []
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
                queries = [str(q).strip() for q in parsed if str(q).strip()]
            except Exception:
                queries = []
        if not queries:
            # Fallback: one query per non-empty line.
            queries = [ln.strip(" -•\t\"'") for ln in raw.splitlines() if ln.strip()]
        queries = [q for q in queries if q and len(q) <= 80][:4]
        if queries:
            logger.info("AI entity queries: %s", queries)
        return queries
    except Exception as exc:
        logger.warning("extract_entity_queries failed: %s", exc)
        return []


async def pick_video_segment(
    query: str, transcript_lines: list[str], duration: float,
    min_len: float = 6.0, max_len: float = 12.0,
) -> tuple[float, float] | None:
    """
    Given a timestamped YouTube auto-caption transcript and a topic/subject
    we need B-roll footage for, ask the LLM to pick the single best-matching
    segment — so only that portion needs to be downloaded, instead of the
    whole video.

    ``transcript_lines`` — list of "MM:SS text" strings (parsed captions,
    ordered by time). ``duration`` — total video duration in seconds.
    Returns (start_seconds, end_seconds), clamped to the video length and a
    sane segment length, or None on failure / no usable transcript.
    """
    if not transcript_lines or duration <= 0:
        return None
    transcript = "\n".join(transcript_lines[:500])
    user_message = (
        "You are given the auto-generated transcript of a YouTube video "
        "(timestamps in MM:SS), and a topic/subject we need short B-roll "
        "footage for.\n\n"
        f"Topic/subject: {query}\n"
        f"Video duration: {duration:.0f} seconds\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Find the single best {min_len:.0f}-{max_len:.0f} second segment of "
        "this video that best shows or discusses the topic above. Respond "
        'with ONLY a JSON object {"start": <seconds>, "end": <seconds>} — '
        "numbers only, no explanation."
    )
    try:
        raw = (await _call_llm_chat(
            [{"role": "user", "content": user_message}], num_predict=200, timeout=90
        )).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        start = float(data.get("start"))
        end = float(data.get("end"))
        if end <= start:
            return None
        start = max(0.0, min(start, max(0.0, duration - 1)))
        end = min(end, duration)
        if end - start < 5:
            return None
        if end - start > max_len + 10:
            end = start + max_len
        logger.info("AI picked video segment %.1f-%.1fs for '%s'", start, end, query)
        return (start, end)
    except Exception as exc:
        logger.warning("pick_video_segment failed: %s", exc)
        return None


async def translate_title_to_english(article_title: str) -> str:
    """
    Translate a Russian (or mixed) article headline to a concise English YouTube title.
    Returns the English title, or an empty string on failure (never returns Cyrillic).
    """
    system_content = (
        "You are a YouTube title translator. You ONLY write in ENGLISH. "
        "Never use Cyrillic. Never respond in Russian. Latin letters only."
    )
    user_message = (
        "Translate the following gaming news headline into English. "
        "Keep it concise (max 80 characters), catchy, and suitable as a YouTube video title. "
        "IMPORTANT: Your answer must be in English only. Use Latin letters only. Never use Cyrillic. "
        "Return ONLY the translated title, nothing else.\n\n"
        f"Headline: {article_title}\n\n"
        "English title:"
    )
    try:
        translated = (await _call_llm_chat(
            [{"role": "system", "content": system_content},
             {"role": "user", "content": user_message}], num_predict=2000, timeout=120
        )).strip('"\'')
        if translated and len(translated) <= 100 and not re.search(r'[а-яёА-ЯЁ]', translated):
            logger.info("Translated title: '%s' ← '%s'", translated, article_title[:60])
            return translated
        if translated:
            logger.warning("translate_title_to_english returned Cyrillic, discarding: '%s'", translated[:60])
    except Exception as exc:
        logger.warning("translate_title_to_english failed: %s", exc)
    return ""


async def generate_thumbnail_hook(article_title: str, lang: str = "ru", prompt=None) -> str:
    """
    Generate a short, punchy, clickable thumbnail caption (2–3 words, ALL CAPS).
    lang: "ru" → Russian output, "en" → English output.
    Falls back to the original title on failure.

    ``prompt`` — optional per-project prompt override (from projects.json),
    may be per-language (``{"en": {...}, "ru": {...}}``).
    """
    _sys, _user = _prompt_parts(prompt, lang=lang)
    if lang == "en":
        system_content = "You are a thumbnail copywriter. You ONLY write in ENGLISH. Never use Cyrillic."
        lang_instruction = "Write in ENGLISH only. Use Latin letters only. Never use Cyrillic or Russian."
    else:
        system_content = "Ты копирайтер для thumbnail. Пишешь ТОЛЬКО на русском языке кириллицей."
        lang_instruction = "Пиши ТОЛЬКО на русском. Используй только кириллицу."

    if _user:
        system_content = _sys or system_content
        user_message = _render_prompt(_user, title=article_title)
    else:
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
        hook = (await _call_llm_chat(
            [{"role": "system", "content": system_content},
             {"role": "user", "content": user_message}],
            num_predict=2000, timeout=120
        )).strip('"\'').upper()
        # Validate: for EN reject if result contains Cyrillic; for RU reject if only Latin
        if lang == "en" and re.search(r'[а-яёА-ЯЁ]', hook):
            logger.warning("Thumbnail hook for EN contained Cyrillic ('%s'), retrying…", hook)
            raise ValueError("cyrillic in en hook")
        if hook and len(hook) <= 60 and "\n" not in hook:
            logger.info("Thumbnail hook: '%s' ← '%s'", hook, article_title[:60])
            return hook
    except ValueError:
        # Retry with even more explicit prompt
        try:
            simple = f"Write a 2-word English gaming thumbnail caption for: {article_title}. English only, no Russian:"
            hook = (await _call_llm_chat(
                [{"role": "system", "content": "Reply in English only."},
                 {"role": "user", "content": simple}],
                num_predict=2000, timeout=120
            )).strip('"\'').upper()
            if hook and not re.search(r'[а-яёА-ЯЁ]', hook) and len(hook) <= 60:
                return hook
        except Exception:
            pass
    except Exception as exc:
        logger.warning("generate_thumbnail_hook failed: %s", exc)
    # For EN: never fall back to a Russian (Cyrillic) title — return empty string instead.
    if lang == "en" and re.search(r'[а-яёА-ЯЁ]', article_title):
        logger.warning("generate_thumbnail_hook EN: all attempts failed, refusing to return Russian title")
        return ""
    return article_title


async def generate_carousel_bullets(
    article_title: str,
    post_text: str,
    *,
    lang: str = "en",
    max_bullets: int = 6,
) -> list[str]:
    """
    Generate short factual bullets for an Instagram carousel.

    Each bullet is one self-contained sentence, ≤90 chars, no leading "•"/"-",
    no hashtags, no emoji. Returns up to *max_bullets* entries.
    """
    plain = re.sub(r'<[^>]+>', ' ', post_text or "")
    plain = re.sub(r'#\w+', '', plain)
    plain = re.sub(r'https?://\S+', '', plain)
    plain = re.sub(r'\s+', ' ', plain).strip()

    if lang == "en":
        system_content = (
            "You write SHORT factual bullet points for an Instagram carousel about gaming news. "
            "You write ONLY in ENGLISH. Never use Cyrillic. Latin letters only."
        )
        lang_hint = "English only. Latin letters only. No Cyrillic."
    else:
        system_content = (
            "Ты пишешь КОРОТКИЕ фактические тезисы для карусели в Instagram про игровые новости. "
            "Пиши ТОЛЬКО на русском кириллицей."
        )
        lang_hint = "Только на русском, кириллицей."

    user_message = (
        f"Headline: {article_title}\n\n"
        f"Article: {plain[:2500]}\n\n"
        f"Write up to {max_bullets} short bullet points summarising the key facts.\n"
        "Rules:\n"
        f"- {lang_hint}\n"
        "- One sentence per bullet, max 90 characters each.\n"
        "- No emoji, no hashtags, no quotation marks, no markdown.\n"
        "- Do NOT prefix lines with '-', '*', '•' or numbers.\n"
        "- Each bullet on its own line.\n"
        "- Cover different facts (release date, platforms, gameplay, reactions, numbers).\n"
        "- Return ONLY the bullet lines, nothing else."
    )
    try:
        raw = await _call_llm_chat(
            [{"role": "system", "content": system_content},
             {"role": "user", "content": user_message}],
            num_predict=4000, num_ctx=4096, timeout=180,
        )
    except Exception as exc:
        logger.warning("generate_carousel_bullets failed: %s", exc)
        return []

    bullets: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        # strip common list markers
        s = re.sub(r'^[\-\*•\u2022\u25CB\u25CF\d\.\)\s]+', '', s).strip()
        s = s.strip('"\'')
        if not s:
            continue
        if lang == "en" and re.search(r'[а-яёА-ЯЁ]', s):
            continue
        if lang == "ru" and not re.search(r'[а-яёА-ЯЁ]', s):
            continue
        # Soft length cap
        if len(s) > 110:
            s = s[:107].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    logger.info("carousel bullets [%s]: %d generated", lang, len(bullets))
    return bullets


# Cliché openers that should never start a narration script (RU + EN).
# Matches the opening sentence/clause and removes it so the script starts
# with the next sentence instead.
_CLICHE_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"забудьте(?:\s+обо?)?[^.!?…]*[.!?…]+"                       # «Забудьте всё, что вы знали…»
    r"|forget\s+(?:about\s+)?(?:everything|all)[^.!?…]*[.!?…]+"   # «Forget everything you knew…»
    r")\s*",
    re.IGNORECASE,
)


def _strip_cliche_opener(script: str) -> str:
    """Remove a tired clichéd opening sentence from a narration script.

    If, after removal, nothing meaningful remains, the original script is
    returned unchanged so we never produce an empty narration.
    """
    if not script:
        return script
    stripped = _CLICHE_OPENER_RE.sub("", script, count=1).lstrip()
    if stripped and len(stripped.split()) >= 10:
        if stripped != script:
            logger.info("Removed clichéd opener from narration script")
        return stripped
    return script


async def generate_video_script(post_text: str, article_title: str, lang: str = "en", prompt=None) -> str:
    """
    Generate a spoken narration script for TikTok/Reels/Shorts.
    lang='en' → English script; lang='ru' → Russian script.
    Target: 45–60 words (~18–25 seconds spoken at natural pace).
    Analytics show videos ≤22s average 84.8% retention vs 56.9% for ≥29s.

    ``prompt`` — optional per-project prompt override (from projects.json),
    may be per-language (``{"en": {...}, "ru": {...}}``).
    """
    # Strip HTML, URLs, markdown and emoji from input text
    clean_text = re.sub(r"<[^>]+>", "", post_text)
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    clean_text = re.sub(r"[*_`#]", "", clean_text)
    clean_text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FEFF]", "", clean_text)
    clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

    _sys_ovr, _user_ovr = _prompt_parts(prompt, lang=lang)

    if lang == "ru":
        system_content = (
            "Ты — сценарист вирусных YouTube Shorts в нише игровых новостей."
            "Твоя задача: создавать сверхдинамичные сценарии Shorts с высоким удержанием внимания."
            "Без сленга, мата и фамильярности."
        )
        user_message = (
            "Напиши ультракороткий сценарий озвучки для игровой новости (формат Shorts/Reels) "
            "по методу «лестницы интереса» — 5 блоков, где вопросы НЕ произносятся вслух, а сами "
            "возникают в голове зрителя из-за того, как подана информация (недосказанность, интригующая деталь, обрыв мысли).\n\n"
            "СТРУКТУРА (строго 5 блоков, идут один за другим БЕЗ заголовков/нумерации и БЕЗ единого вопросительного знака в тексте):\n"
            "- Блок 1: УТВЕРЖДЕНИЕ-крючок по сути новости, которое само по себе рождает у зрителя вопрос «а что дальше?/почему?/как это возможно?» — но формулируется как факт или интригующая деталь, а не как вопрос\n"
            "- Блок 2: закрывает вопрос, возникший после блока 1 (1 факт из новости), и сразу подаёт новую деталь, которая рождает следующий невысказанный вопрос\n"
            "- Блок 3: закрывает вопрос из блока 2 (следующий факт) и подаёт деталь, рождающую ещё более острый невысказанный вопрос\n"
            "- Блок 4: закрывает вопрос из блока 3 (следующий факт) и подаёт деталь, рождающую самый острый невысказанный вопрос\n"
            "- Блок 5: закрывает последний вопрос финальным фактом-выводом — завершённое утверждение, ставит точку в истории\n\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            "- Язык: ТОЛЬКО русский\n"
            "- Длина: СТРОГО 45–60 слов на весь сценарий (18–25 секунд речи) — НЕЛЬЗЯ БОЛЬШЕ\n"
            "- Только чистый текст — БЕЗ хэштегов, HTML, эмодзи, markdown, без пометок «Блок N»\n"
            "- ЗАПРЕЩЕНО использовать вопросительные знаки и прямые вопросительные конструкции («что если…», «а вы знали…», «почему…?») — только повествовательные утверждения, которые сами намекают на вопрос\n"
            "- каждый факт должен быть основан ТОЛЬКО на тексте новости\n"
            "- никакой воды, каждая фраза двигает историю вперёд и оставляет недосказанность до следующей фразы\n"
            "- Текст будет зачитан голосовым AI — пиши чёткими литературными предложениями\n"
            "- стиль эмоциональный, как у топовых gaming Shorts каналов\n"
            "- Блок 5 должен быть завершённым утвердительным предложением — НЕ обрывать на предлоге или союзе\n"
            "- ЗАПРЕЩЕНО: мат, сленг, грубые выражения, фамильярное обращение\n"
            "- ЗАПРЕЩЕНО начинать с избитых клише, особенно "
            "«Забудьте всё, что вы знали…», «Забудьте о…», «Представьте…». "
            "Крючок в блоке 1 должен быть оригинальным и конкретным по сути новости\n\n"
            f"Заголовок: {article_title}\n\n"
            f"Текст поста:\n{clean_text[:1800]}\n\n"
            "Напиши сценарий из 5 блоков подряд, слитным текстом, без вопросительных знаков (СТРОГО 45–60 слов):"
        )
    else:
        system_content = (
            "You are a scriptwriter for viral YouTube Shorts in the gaming news niche. "
            "Your task: create ultra-dynamic Shorts scripts with high viewer retention. "
            "No slang, no profanity, no familiar tone."
        )
        user_message = (
            "Write an ultra-short narration script for a gaming news piece (Shorts/Reels format) "
            "using the 'curiosity ladder' method — 5 blocks where questions are NEVER spoken out loud, "
            "but arise naturally in the viewer's mind because of how the information is withheld or teased.\n\n"
            "STRUCTURE (strictly 5 blocks, run one after another, NO labels/numbering and NOT A SINGLE question mark in the text):\n"
            "- Block 1: a HOOK STATEMENT about the news that naturally makes the viewer wonder 'what happens next / why / how' — phrased as a fact or intriguing detail, never as a question\n"
            "- Block 2: resolves the implicit question from block 1 (1 fact from the article), then immediately drops a new detail that plants the next unspoken question\n"
            "- Block 3: resolves the implicit question from block 2 (next fact), then plants an even sharper unspoken question\n"
            "- Block 4: resolves the implicit question from block 3 (next fact), then plants the sharpest unspoken question\n"
            "- Block 5: resolves the final implicit question with a closing fact — a complete statement that ends the story\n\n"
            "STRICT RULES:\n"
            "- Language: ENGLISH ONLY\n"
            "- Length: STRICTLY 45–60 words for the whole script (18–25 seconds of speech) — DO NOT EXCEED\n"
            "- Plain text only — NO hashtags, HTML, emojis, markdown, no 'Block N' labels\n"
            "- FORBIDDEN to use question marks or direct question phrasing ('what if...', 'did you know...', 'why...?') — only declarative statements that imply the question\n"
            "- every fact must be based ONLY on the article text\n"
            "- no filler, every phrase must move the story forward and leave something unsaid until the next phrase\n"
            "- The text will be read aloud by a voice AI — write clean, literary sentences\n"
            "- Emotional style, like top gaming Shorts channels\n"
            "- Block 5 must be a COMPLETE declarative sentence — NEVER cut off on a preposition or conjunction\n"
            "- FORBIDDEN: profanity, slang, rude expressions, familiar tone\n"
            "- FORBIDDEN to open with tired clichés, especially "
            "'Forget everything you knew...', 'Forget about...', 'Imagine...'. "
            "The hook in block 1 must be original and specific to the news itself\n\n"
            f"Article title: {article_title}\n\n"
            f"Post content:\n{clean_text[:1800]}\n\n"
            "Write the 5-block script as one continuous flow of text, with no question marks (STRICTLY 45–60 words):"
        )

    # Per-project override (projects.json) takes precedence over the defaults.
    if _user_ovr:
        user_message = _render_prompt(_user_ovr, title=article_title, post_text=clean_text[:1800])
    if _sys_ovr:
        system_content = _sys_ovr

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]
    try:
        script = await _call_llm_chat(
            messages,
            num_predict=8000,
            num_ctx=16384,
            timeout=300,
            model=GEMINI_VIDEO_MODEL,
        )
        # --- Extract answer from thinking / reasoning blocks ---
        # Strategy: prefer content AFTER the closing thinking tag; fall back to
        # content INSIDE the block (gemma4 sometimes puts the answer there when
        # thinking suppression is incomplete).

        # Gemma 4 native format: <|channel>thought\n[reasoning]<channel|>[answer]
        _g4 = re.search(r"<\|channel>thought\n(.*?)<channel\|>(.*)", script, flags=re.DOTALL)
        if _g4:
            after = _g4.group(2).strip()
            inside = _g4.group(1).strip()
            script = after if len(after.split()) >= 5 else inside
            logger.debug("Gemma4 channel block: after=%d words, inside=%d words → used %s",
                         len(after.split()), len(inside.split()), "after" if after else "inside")

        # Standard <think>…</think> (qwen3, deepseek-r1, gemma4 via Gemini API)
        _th = re.search(r"<think>(.*?)</think>(.*)", script, flags=re.DOTALL | re.IGNORECASE)
        if _th:
            after = _th.group(2).strip()
            inside = _th.group(1).strip()
            script = after if len(after.split()) >= 5 else inside
            logger.debug("think block: after=%d words, inside=%d words → used %s",
                         len(after.split()), len(inside.split()), "after" if after else "inside")
        elif "</think>" in script.lower():
            # Closing tag without opening — drop prefix up to last </think>
            script = re.split(r"</think>", script, maxsplit=1, flags=re.IGNORECASE)[-1].strip()

        # Remove only known safe HTML tags; avoid stripping partial words that
        # happen to contain angle brackets (e.g. "mechan<ics>" → "mechan").
        script = re.sub(r"<(?:br|p|div|span|b|i|em|strong|ul|li|ol|h[1-6])[^>]*/?>", "", script, flags=re.IGNORECASE)
        script = re.sub(r"[*_`#]", "", script)
        script = script.strip()
        # Strip the overused "Forget everything you knew…" clichéd opener if the
        # model used it anyway, despite the prompt forbidding it.
        script = _strip_cliche_opener(script)
        word_count = len(script.split())
        logger.info("%s video script generated: %d words", lang.upper(), word_count)
        if word_count > 90:
            # Trim to first 75 words to stay within target duration
            script = " ".join(script.split()[:75])
            logger.info("Script trimmed to 75 words for retention target")
            word_count = 75
        # Drop a trailing incomplete sentence (model often gets cut mid-word
        # when it exceeds num_predict). Trim back to the last sentence-ending
        # punctuation if the script doesn't already end on one.
        _sentence_end = '.!?…»"\')'
        if script and script[-1] not in _sentence_end:
            last_idx = max(script.rfind(c) for c in '.!?…')
            if last_idx > 0:
                trimmed = script[: last_idx + 1].rstrip()
                trimmed_words = len(trimmed.split())
                logger.info(
                    "%s script: dropped incomplete trailing sentence (%d → %d words)",
                    lang.upper(), word_count, trimmed_words,
                )
                script = trimmed

        # Also catch "complete-looking" but unfinished cliffhangers — the model
        # sometimes ends on a short trailing clause like "This is why..." or
        # "Because..." whose LAST char *is* a period/ellipsis (so the check
        # above doesn't fire) but which never delivers the payoff it promises.
        # Heuristic: strip the trailing ellipsis itself, then look at the
        # sentence before it — if what remains after that sentence is short
        # (≤8 words), it's a dangling cliffhanger — drop it and fall back to
        # the previous (already complete) sentence.
        if re.search(r"\.{2,}\s*$", script):
            core = re.sub(r"\.{2,}\s*$", "", script).rstrip()
            prior_ends = [core.rfind(c) for c in ".!?"]
            prior_end = max((i for i in prior_ends if i >= 0), default=-1)
            tail = core[prior_end + 1:].strip() if prior_end >= 0 else core
            if prior_end >= 0 and len(tail.split()) <= 8:
                trimmed = core[: prior_end + 1].rstrip()
                if len(trimmed.split()) >= 5:
                    logger.info(
                        "%s script: dropped dangling ellipsis cliffhanger ('%s') — %d → %d words",
                        lang.upper(), tail[:40], word_count, len(trimmed.split()),
                    )
                    script = trimmed
        return script
    except Exception as exc:
        logger.error("Error generating %s video script: %s", lang.upper(), exc)
        return clean_text[:280].strip()
