import json
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID: str = os.getenv("TELEGRAM_CHANNEL_ID", "@playitnews")
TELEGRAM_SECOND_CHANNEL_ID: str = os.getenv("TELEGRAM_SECOND_CHANNEL_ID", "@readitgames")
TELEGRAM_ADMIN_CHAT_ID: int = int(os.getenv("TELEGRAM_ADMIN_CHAT_ID", "0"))

# --- Local Bot API server (optional) ---
# Telegram's official cloud Bot API caps uploads at 50 MB. Running your own
# `telegram-bot-api` server (https://github.com/tdlib/telegram-bot-api) raises
# that limit to 2000 MB. To enable:
#   1) Get api_id / api_hash on https://my.telegram.org/apps
#   2) Run: telegram-bot-api --local --api-id=XXXX --api-hash=YYYY --dir=/tmp/tgbotapi
#      (or use ./start_tg_api.sh — see start.sh)
#   3) One-time switch the bot from cloud to local:
#        curl https://api.telegram.org/bot<TOKEN>/logOut
#   4) Set TELEGRAM_LOCAL_API_URL in .env (default below points at localhost).
# Leave TELEGRAM_LOCAL_API_URL empty to keep using the official cloud API.
TELEGRAM_LOCAL_API_URL: str = os.getenv("TELEGRAM_LOCAL_API_URL", "").rstrip("/")
# File-download endpoint of the same server. Defaults are derived from the API URL.
TELEGRAM_LOCAL_API_FILE_URL: str = os.getenv("TELEGRAM_LOCAL_API_FILE_URL", "").rstrip("/")
if TELEGRAM_LOCAL_API_URL and not TELEGRAM_LOCAL_API_FILE_URL:
    # Convention: same host, path /file/bot<token>/...
    TELEGRAM_LOCAL_API_FILE_URL = TELEGRAM_LOCAL_API_URL.replace("/bot", "/file/bot") \
        if "/bot" in TELEGRAM_LOCAL_API_URL else TELEGRAM_LOCAL_API_URL + "/file"

TELEGRAM_LOCAL_MODE: bool = bool(TELEGRAM_LOCAL_API_URL)

# Telegram upload size cap. 50 MB on cloud, 2000 MB on local Bot API server.
TG_MAX_BYTES: int = (2000 if TELEGRAM_LOCAL_MODE else 50) * 1024 * 1024

# Credentials for the local Bot API server itself (not the bot token).
TELEGRAM_API_ID: str = os.getenv("TELEGRAM_API_ID", "")
TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")

# Ollama (local)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Default model — used for Telegram post generation, translations, filters.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:latest")
# Separate model for video script generation (reasoning-grade for better hooks/structure).
OLLAMA_VIDEO_MODEL: str = os.getenv("OLLAMA_VIDEO_MODEL", "gemma4:31b")

# LLM backend for text generation: "ollama" (local) or "gemini" (Google cloud).
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").strip().lower()
# Google Gemini API (https://aistudio.google.com/apikey)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Cloud model for text generation. Google's API serves Gemma open models
# (e.g. "gemma-4-31b-it") as well as "gemini-*" models.
GEMINI_TEXT_MODEL: str = os.getenv("GEMINI_TEXT_MODEL", "gemma-4-31b-it")
# Optional separate model for video-script generation (defaults to the text model).
GEMINI_VIDEO_MODEL: str = os.getenv("GEMINI_VIDEO_MODEL", GEMINI_TEXT_MODEL)

# Monitoring
CHECK_INTERVAL_MINUTES: int = int(os.getenv("CHECK_INTERVAL_MINUTES", "30"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DB_PATH = os.path.join(BASE_DIR, "data.db")

# Instagram Graph API — English account
# Required: Business/Creator account connected to a Facebook Page
# Graph API app must have instagram_content_publish permission
INSTAGRAM_USER_ID: str = os.getenv("INSTAGRAM_USER_ID", "")
INSTAGRAM_ACCESS_TOKEN: str = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")

# Instagram Graph API — Russian account (separate channel)
INSTAGRAM_USER_ID_RU: str = os.getenv("INSTAGRAM_USER_ID_RU", "")
INSTAGRAM_ACCESS_TOKEN_RU: str = os.getenv("INSTAGRAM_ACCESS_TOKEN_RU", "")

# YouTube Data API v3
# client_secrets.json path (downloaded from Google Cloud Console)
YOUTUBE_CLIENT_SECRETS: str = os.getenv("YOUTUBE_CLIENT_SECRETS", os.path.join(BASE_DIR, "client_secrets.json"))
# OAuth token file (auto-created by get_youtube_token.py)
YOUTUBE_TOKEN_FILE: str = os.getenv("YOUTUBE_TOKEN_FILE", os.path.join(BASE_DIR, "youtube_token.json"))
# Category ID: 20 = Gaming
YOUTUBE_CATEGORY_ID: str = os.getenv("YOUTUBE_CATEGORY_ID", "20")

# YouTube — Russian channel (separate Google account / channel)
# Run get_youtube_token.py once logged in as the RU account, then point this variable at the saved token.
YOUTUBE_TOKEN_FILE_RU: str = os.getenv("YOUTUBE_TOKEN_FILE_RU", os.path.join(BASE_DIR, "youtube_token_ru.json"))

# TikTok browser session (persistent Chromium profile, created by get_tiktok_session.py)
TIKTOK_SESSION_DIR: str = os.getenv("TIKTOK_SESSION_DIR", os.path.join(BASE_DIR, "tiktok_session"))

# VK API — загрузка коротких видео (VK Клипы)
# Требуется access token со scope `video` (выдаётся по запросу в devsupport@corp.vk.com).
# VK_GROUP_ID — числовой ID сообщества без минуса; если пусто, видео грузится в профиль владельца токена.
VK_ACCESS_TOKEN: str = os.getenv("VK_ACCESS_TOKEN", "")
VK_GROUP_ID: str = os.getenv("VK_GROUP_ID", "")
# Версия VK API
VK_API_VERSION: str = os.getenv("VK_API_VERSION", "5.199")
# Публиковать ли запись с клипом на стене сообщества после загрузки (1 — да, 0 — нет).
VK_WALLPOST: bool = os.getenv("VK_WALLPOST", "1") == "1"

# Source
NEWS_URL = "https://www.playground.ru/news"

# --- Content filtering ---
# Skip articles whose URL path contains any of these segments.
# Covers entire categories (movies, trailers, etc.).
BLOCKED_URL_CATEGORIES: list[str] = [
    "/news/movies/",
]

# Video generation — YouTube clip settings
YT_CLIP_DURATION: int = int(os.getenv("YT_CLIP_DURATION", "8"))    # seconds per clip
YT_CLIP_SKIP: int = int(os.getenv("YT_CLIP_SKIP", "5"))            # skip intro seconds
YT_MAX_CLIPS: int = int(os.getenv("YT_MAX_CLIPS", "5"))             # max clips to download
YT_MAX_FILESIZE: int = int(os.getenv("YT_MAX_FILESIZE", "1500"))      # MB per clip (yt-dlp limit)


def setup_dirs() -> None:
    """Create required runtime directories. Call once at application startup."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Multi-project configuration
# ---------------------------------------------------------------------------
# Each "project" (gaming / ufc / movies …) has its own news source and its own
# set of social platforms. The structure lives in projects.json; secrets stay
# in environment variables (projects.json only references their names / paths).

PROJECTS_FILE: str = os.path.join(BASE_DIR, "projects.json")
DEFAULT_PROJECT: str = "gaming"


def _load_projects() -> tuple[dict, str]:
    """Load projects.json → (projects_dict, default_project_name).

    Falls back to a minimal single-project config if the file is missing or
    malformed, so the app keeps working exactly as before.
    """
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {DEFAULT_PROJECT: {}}, DEFAULT_PROJECT
    projects = data.get("projects", {}) if isinstance(data, dict) else {}
    default = data.get("default_project", DEFAULT_PROJECT) if isinstance(data, dict) else DEFAULT_PROJECT
    if default not in projects:
        default = next(iter(projects), DEFAULT_PROJECT)
    return projects or {DEFAULT_PROJECT: {}}, default


PROJECTS, DEFAULT_PROJECT = _load_projects()


def get_project(name: str | None) -> dict:
    """Return a project's config dict (with its name injected as ``name``).

    Unknown / missing names fall back to the default project.
    """
    if name and name in PROJECTS:
        return {"name": name, **PROJECTS[name]}
    return {"name": DEFAULT_PROJECT, **PROJECTS.get(DEFAULT_PROJECT, {})}


def project_names() -> list[str]:
    """Ordered list of configured project names."""
    return list(PROJECTS.keys())


def project_platforms(name: str | None) -> dict:
    """Return the ``platforms`` mapping of a project (key → config dict)."""
    return get_project(name).get("platforms", {}) or {}


def required_platforms(name: str | None) -> set[str]:
    """Platform keys that must be published for a post to count as fully done.

    A platform counts unless its config sets ``"counts_as_published": false``
    (used for Telegram channels, which are tracked via the post's ``sent``
    status rather than the ``published_platforms`` list).
    """
    return {
        key
        for key, cfg in project_platforms(name).items()
        if (cfg or {}).get("counts_as_published", True)
    }


def platform_credentials(project: str | None, platform_key: str) -> dict:
    """Resolve a project's platform config (projects.json) into concrete values.

    Mapping rules for each key in the platform's config object:
      - keys ending in ``_env`` are read from the environment and the ``_env``
        suffix is dropped (``user_id_env`` → ``user_id``, ``token_env`` →
        ``token``, ``channel_env`` → ``channel``, ``group_env`` → ``group``).
      - ``token_file`` is resolved to an absolute path relative to BASE_DIR.
      - any other key (``label``, ``footer``, ``counts_as_published``) is passed
        through unchanged.

    Returns an empty dict if the project/platform is unknown.
    """
    cfg = project_platforms(project).get(platform_key, {}) or {}
    out: dict = {}
    for key, value in cfg.items():
        if isinstance(key, str) and key.endswith("_env"):
            out[key[:-4]] = os.getenv(value, "") if isinstance(value, str) else ""
        elif key == "token_file":
            sval = str(value)
            out["token_file"] = sval if os.path.isabs(sval) else os.path.join(BASE_DIR, sval)
        else:
            out[key] = value
    return out


def project_ai(project: str | None, key: str, default=None):
    """Return the per-project AI prompt config for a step, from projects.json.

    Looks up ``projects.<name>.ai.<key>``. The value may be a string (a bare
    user-prompt template) or an object (e.g. ``{"system": ..., "user": ...}`` or
    a per-language mapping ``{"en": {...}, "ru": {...}}``). Returns ``default``
    when the project has no ``ai`` section or no such key, so callers fall back
    to their built-in default prompts.
    """
    ai = (get_project(project).get("ai") or {})
    val = ai.get(key)
    return val if val is not None else default



