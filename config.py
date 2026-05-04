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
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Monitoring
CHECK_INTERVAL_MINUTES: int = int(os.getenv("CHECK_INTERVAL_MINUTES", "30"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DB_PATH = os.path.join(BASE_DIR, "data.db")

# Pixabay (free stock images/videos for video generation)
# Get a free API key at https://pixabay.com/api/docs/
PIXABAY_API_KEY: str = os.getenv("PIXABAY_API_KEY", "")

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
