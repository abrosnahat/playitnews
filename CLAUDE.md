# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**PlayItNews** is a Python-based news automation pipeline that monitors [playground.ru/news](https://www.playground.ru/news), uses AI to translate/adapt gaming news (RU→EN and RU→RU), and publishes to multiple platforms:

- Telegram channels (`@playitnews` EN, `@readitgames` RU)
- Instagram (Reels, EN + RU accounts)
- YouTube (Shorts, EN + RU channels)
- TikTok (via persistent browser session)

The pipeline scrapes articles, downloads images and video clips (HLS / YouTube embeds via `yt-dlp`), generates short videos with TTS narration (`edge-tts`) and subtitles (`faster-whisper`), and routes everything through an admin approval flow on Telegram.

## Architecture

### Entry points
- `main.py` — orchestrator. Boots Telegram bot, schedules periodic news checks, runs the pipeline: scrape → adapt → schedule → publish.
- `webapp.py` — Flask dashboard on port `5003` (optionally exposed via `cloudflared`).
- `start.sh` — launcher that kills stale processes, starts `main.py` + `webapp.py` + Cloudflare tunnel.

### Modules
| File | Responsibility |
| --- | --- |
| `scraper.py` | playground.ru parsing, image and HLS video download, YouTube embed extraction |
| `ai_adapter.py` | LLM calls (Ollama / Claude) for translation, adaptation, gaming-relevance check |
| `bot.py` | Telegram handlers — Approve / Edit / Cancel admin buttons |
| `database.py` | SQLite (`data.db`) — seen articles, scheduled posts, publication state |
| `config.py` | Loads `.env`, defines paths, constants, content filters |
| `video_generator.py` | Composes vertical video (clips + TTS + subtitles + music) |
| `thumbnail_generator.py` | YouTube thumbnail rendering |
| `analyze_video.py` | Helpers for clip selection / scoring |
| `instagram_publisher.py` | Instagram Graph API Reels publishing |
| `youtube_publisher.py` | YouTube Data API v3 upload |
| `github_uploader.py` | Uploads media to GitHub for public hosting (used by Instagram/TikTok flows) |
| `get_*_token.py` | One-shot OAuth / session helpers (Instagram, YouTube, TikTok) |

### Runtime directories (auto-created)
- `images/` — downloaded article images
- `videos/clips_*/` — per-article working dirs for downloaded YouTube clips and assembled video
- `music/` — background music pool
- `static/` — dashboard frontend (`index.html`)

### Persistent state
- `data.db` (SQLite) — never delete in production
- `playitnews.log` — rolling log
- `youtube_token.json`, `youtube_token_ru.json`, `client_secrets.json` — OAuth tokens (do **not** commit)
- `tiktok_session/` — persistent Chromium profile for TikTok

## Development

### Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.11+. Dependencies pinned in `requirements.txt` (note: `python-telegram-bot[job-queue]==21.9`).

### Run
```bash
# Full stack (bot + dashboard + tunnel)
./start.sh

# Bot only
python3 main.py

# Dashboard only
python3 webapp.py
```

### Configuration
All secrets live in `.env` (loaded by `config.py` via `python-dotenv`). Key vars:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_CHAT_ID`, `TELEGRAM_CHANNEL_ID`, `TELEGRAM_SECOND_CHANNEL_ID`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (local LLM for adaptation)
- `INSTAGRAM_USER_ID[_RU]`, `INSTAGRAM_ACCESS_TOKEN[_RU]`
- `YOUTUBE_TOKEN_FILE[_RU]`, `YOUTUBE_CLIENT_SECRETS`
- `PIXABAY_API_KEY`
- `CHECK_INTERVAL_MINUTES`, `YT_CLIP_DURATION`, `YT_CLIP_SKIP`, `YT_MAX_CLIPS`, `YT_MAX_FILESIZE`
- `BLOCKED_URL_CATEGORIES` (in `config.py`) — URL path segments to skip (e.g. `/news/movies/`).

## Conventions

- **Async-first**: scraping, downloads, Telegram I/O all use `asyncio` + `aiohttp`. Use `aiofiles` for disk I/O inside coroutines.
- **Logging**: get a module logger via `logging.getLogger(__name__)`. Logs go to stdout + `playitnews.log`. Third-party loggers (`httpx`, `apscheduler`, `telegram*`) are silenced in `main.py` — preserve that.
- **No new top-level scripts** unless they serve a clear pipeline role; prefer extending an existing module.
- **SQLite schema changes**: update `database.py` migrations carefully; `data.db` is shared with the running service.
- **Russian comments / log messages** are common in this codebase — keep them when editing nearby code unless the user asks otherwise.
- **`bot.py.bak`** is a manual backup; do not edit.

## Things to avoid

- Don't commit `.env`, `*.json` token files, `data.db`, `playitnews.log`, `tiktok_session/`, or media under `images/` / `videos/` / `music/`.
- Don't hardcode credentials — always read through `config.py`.
- Don't run the bot twice against the same Telegram token (causes `getUpdates` conflicts). `start.sh` handles cleanup; respect it.
- Don't switch the LLM backend without updating `ai_adapter.py` and the `OLLAMA_*` env vars together.
