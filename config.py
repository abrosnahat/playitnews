import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID: str = os.getenv("TELEGRAM_CHANNEL_ID", "@playitnews")
TELEGRAM_ADMIN_CHAT_ID: int = int(os.getenv("TELEGRAM_ADMIN_CHAT_ID", "0"))

# Ollama (local)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# Monitoring
CHECK_INTERVAL_MINUTES: int = int(os.getenv("CHECK_INTERVAL_MINUTES", "30"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DB_PATH = os.path.join(BASE_DIR, "data.db")

# Source
NEWS_URL = "https://www.playground.ru/news"

# --- Content filtering ---
# Skip articles whose URL path contains any of these segments.
# Covers entire categories (movies, trailers, etc.).
BLOCKED_URL_CATEGORIES: list[str] = [
    "/news/movies/",
]


os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
