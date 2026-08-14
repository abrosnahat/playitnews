# PlayItNews — Multi-Project News-to-Video Automation Pipeline

Monitors news sources, uses AI to translate/rewrite articles, generates short
vertical videos (TikTok/Reels/Shorts style) with TTS narration and burned-in
subtitles, and publishes everything across Telegram, Instagram, YouTube and
VK — with every post gated behind a human approval step in a local web
dashboard.

The pipeline is **multi-project**: each project (currently `gaming` — branded
*WatchItNews*, movies/TV series news — and `ufc` — MMA/UFC news) has its own
news source, its own AI prompts, and its own set of social-media targets, all
declared in [projects.json](projects.json).

---

## What it does

1. **Scrape** — periodically checks each project's news source for new
   articles (`sources.py` → `scraper.py`).
2. **Adapt** — an LLM (Google Gemini/Gemma, see
   [ai_adapter.py](ai_adapter.py)) translates/rewrites the article into EN and
   RU Telegram posts, checks topical relevance, and later generates a short
   video narration script.
3. **Schedule + notify** — the post is stored in SQLite
   ([database.py](database.py)) and a Telegram notification is sent to the
   admin ([bot.py](bot.py)).
4. **Review** — all actual moderation (approve/edit/cancel, generate video,
   publish to each platform) happens in the local **web dashboard**
   ([webapp.py](webapp.py) + [static/index.html](static/index.html)), not via
   Telegram buttons — the bot is notification-only.
5. **Video generation** ([video_generator.py](video_generator.py)) — builds a
   1080×1920 vertical video per post: TTS narration (`edge-tts` or Gemini
   TTS), `faster-whisper` word-level subtitle timing, YouTube/Pixabay
   b-roll clips selected and ordered to match the script (via Gemini Vision
   frame tagging), optional per-project overlays (e.g. UFC mid-roll ad / banner), burned-in subtitles.
6. **Publish** — pushes the finished post/video to whichever platforms the
   project enables: Telegram channel, Instagram (Reels + carousels),
   YouTube Shorts, VK Clips. TikTok has a persistent-browser session helper
   ([get_tiktok_session.py](get_tiktok_session.py)) but automated TikTok
   upload is not wired in yet.

---

## Architecture

### Entry points
| File | Role |
| --- | --- |
| [main.py](main.py) | Orchestrator — boots the Telegram bot, schedules a periodic news check per project, runs scrape → adapt → schedule for each new article. |
| [webapp.py](webapp.py) | Flask dashboard (port `5003`) — the primary control surface: review/edit posts, generate videos, publish to each platform. |
| [start.ps1](start.ps1) | Windows launcher — kills stale processes, starts `main.py` + `webapp.py` (+ optional Cloudflare tunnel / local Telegram Bot API server). |
| [start.sh](start.sh) | macOS/Linux equivalent of `start.ps1`. |

### Core modules
| File | Responsibility |
| --- | --- |
| [sources.py](sources.py) | Pluggable per-project news sources (`playground`, `playground_movies`, `fighttime`) — each exposes `get_latest_links` / `scrape_article`. |
| [scraper.py](scraper.py) | playground.ru parsing, image download, HLS/YouTube-embed video download (`yt-dlp`). |
| [ai_adapter.py](ai_adapter.py) | All LLM calls — translation/adaptation, relevance filter, search-query extraction, thumbnail hook text, video scripts. Calls the Google Gemini/Gemma API; per-project prompt overrides come from `projects.json`. |
| [gemini_keys.py](gemini_keys.py) | Rotates across multiple `GEMINI_API_KEY_N` keys on 429/quota errors, persists the active key across the `main.py`/`webapp.py` processes. |
| [bot.py](bot.py) | Telegram — sends admin notifications with a link to the dashboard (no inline moderation buttons). |
| [database.py](database.py) | SQLite (`data.db`) — seen articles, scheduled posts, publication state, per-post project tag. |
| [config.py](config.py) | Loads `.env`, path constants, and the `projects.json` loader (`get_project`, `project_platforms`, `required_platforms`, `platform_credentials`, `project_ai`). |
| [projects.json](projects.json) | Per-project source, check interval, platform credentials/labels, and AI prompt overrides. |
| [video_generator.py](video_generator.py) | Builds the short video: TTS, subtitles, clip selection/ordering, project-specific overlays, final ffmpeg assembly. |
| [thumbnail_generator.py](thumbnail_generator.py) | Renders the YouTube thumbnail (hook text over a frame). |
| [carousel_builder.py](carousel_builder.py) | Builds Instagram carousel slide images. |
| [analyze_video.py](analyze_video.py) | Clip scoring/selection helpers (scene-change + text-overlay detection) used by `video_generator.py`. |
| [instagram_publisher.py](instagram_publisher.py) / [instagram_carousel_publisher.py](instagram_carousel_publisher.py) | Instagram Graph API — Reels / carousel publishing. |
| [youtube_publisher.py](youtube_publisher.py) | YouTube Data API v3 upload (Shorts). |
| [vk_publisher.py](vk_publisher.py) | VK API — Clips upload + optional wall post. |
| [github_uploader.py](github_uploader.py) | Uploads media to GitHub for public hosting (used where a public URL is required). |
| [get_instagram_token.py](get_instagram_token.py), [get_youtube_token.py](get_youtube_token.py), [get_vk_token.py](get_vk_token.py), [get_tiktok_session.py](get_tiktok_session.py) | One-shot OAuth/session helpers, run manually once per account. |
| [redownload_active.py](redownload_active.py) | Utility to re-download missing/expired media for still-active posts. |

### Runtime directories / files (auto-created, gitignored)
- `images/` — downloaded article images (`images/carousels/` for carousel slides)
- `videos/` — per-article working dirs (`clips_*/`, `gen_*/`) with downloaded clips and assembled videos; `tts_manual/` for the dashboard's standalone TTS tool
- `music/` — background music pool for video generation
- `assets/` — static assets checked into the repo: EAST text-detection model, optional mid-roll ad/banner overlays for video generation
- `data.db` (SQLite, never delete in production), `playitnews.log`, `gemini_key_state.json`
- `youtube_token.json`, `youtube_token_ru.json`, `client_secrets.json` — OAuth tokens (**not committed**)
- `tiktok_session/` — persistent Chromium profile for TikTok (session capture only)

---

## Setup

### 1. Python environment
Python 3.11+ (dependencies pinned in [requirements.txt](requirements.txt)).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

`ffmpeg` and `yt-dlp` must be available on `PATH` for video generation.

### 2. Telegram bot
1. `@BotFather` → `/newbot`, copy the token.
2. Add the bot as **Administrator** with *Post Messages* permission in each channel referenced by your projects.
3. Message `@userinfobot` to get your numeric admin chat ID.

### 3. LLM (Gemini)
All AI text generation ([ai_adapter.py](ai_adapter.py)) goes through Google's Gemini/Gemma API ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). Set `GEMINI_API_KEY` in `.env` (or `GEMINI_API_KEY_1`, `_2`, `_3`, … for multiple keys rotated automatically on quota errors — see [gemini_keys.py](gemini_keys.py)). `GEMINI_TEXT_MODEL` / `GEMINI_VIDEO_MODEL` select the models used for regular text vs. the heavier video-script generation.

### 4. Publishing platform credentials
Configure whichever platforms a project needs. Values referenced from
`projects.json` (via `*_env` keys or inline) come from `.env`:
- **Instagram Graph API**: Business/Creator account + Facebook Page, app with `instagram_content_publish` permission → `INSTAGRAM_USER_ID` / `INSTAGRAM_ACCESS_TOKEN` (+ `_RU` variants). Helper: `get_instagram_token.py`.
- **YouTube Data API v3**: `client_secrets.json` from Google Cloud Console + `get_youtube_token.py` to produce `youtube_token.json` (`_ru` variant for a second channel).
- **VK API**: `VK_ACCESS_TOKEN` with `video` scope + `VK_GROUP_ID`. Helper: `get_vk_token.py`.
- **TikTok**: `get_tiktok_session.py` opens a browser to log in and saves the session to `tiktok_session/` (upload automation not yet implemented).

### 5. Configure `.env`
There is no `.env.example` in this repo — create `.env` in the project root and fill in the variables referenced in [config.py](config.py): Telegram tokens/IDs, `GEMINI_API_KEY` (+ platform credentials above), plus optional tuning knobs (`CHECK_INTERVAL_MINUTES`, `YT_CLIP_DURATION`, `YT_CLIP_SKIP`, `YT_MAX_CLIPS`, `YT_MAX_FILESIZE`, `PIXABAY_API_KEY`, `SMART_FRAME_MATCH`, `TTS_BACKEND`, `TTS_PODCAST_POLISH`, etc. — see comments in `config.py` / `video_generator.py`).

### 6. Configure `projects.json`
Add/edit a project block: `title`, `source` (a key from `sources.py`'s
`_SOURCES`), `check_interval_minutes`, `platforms` (one entry per publish
target, referencing env vars or inline credentials, `counts_as_published`
to exclude Telegram from the "fully published" count), and optional `ai`
prompt overrides (`search_query`, `relevance`, `post_text_en`/`post_text_ru`,
`video_script`, `thumbnail_hook`).

---

## Running

```powershell
# Full stack: Telegram bot + dashboard (+ Cloudflare tunnel if configured)
.\start.ps1

# Or individually
.\.venv\Scripts\python.exe main.py      # bot only
.\.venv\Scripts\python.exe webapp.py    # dashboard only, http://localhost:5003
```

`start.ps1` kills any stale `main.py`/`webapp.py`/`cloudflared` processes
first to avoid Telegram `getUpdates` conflicts and port clashes on `:5003`.

---

## Dashboard workflow

Open `http://localhost:5003`. Each detected article becomes a post card:
review/edit the EN and RU text, generate a short video (with optional
project-specific overlays),
preview it, then publish per-platform (Telegram / Instagram / Instagram
carousel / YouTube / VK) — separately for EN and RU where applicable. A
project tab bar filters posts by project. A standalone "paste text → TTS
audio" tool is also available from the dashboard header.

The Telegram bot only sends a notification when a new article is queued —
approval/editing/publishing all happens in the dashboard.

---

## Conventions

- **Async-first**: scraping, downloads, Telegram I/O use `asyncio` + `aiohttp`; `aiofiles` for disk I/O inside coroutines.
- **Logging**: `logging.getLogger(__name__)` per module → stdout + `playitnews.log`. Noisy third-party loggers (`httpx`, `apscheduler`, `telegram*`) are silenced in `main.py` — preserve that.
- **Multi-project**: never hardcode project-specific values (channel names, prompts, credentials) in code — add them to `projects.json` / `.env` and read them via `config.py`.
- **SQLite schema changes**: add migrations carefully in `database.py`; `data.db` is shared with the running service.
- Russian comments/log messages are common in this codebase — keep them when editing nearby code unless asked otherwise.
- `bot.py.bak` is a manual backup; do not edit.

## Things to avoid

- Don't commit `.env`, `*.json` token files, `data.db`, `playitnews.log`, `gemini_key_state.json`, `tiktok_session/`, or media under `images/` / `videos/` / `music/`.
- Don't hardcode credentials — always read through `config.py` / `projects.json`.
- Don't run the bot twice against the same Telegram token (causes `getUpdates` conflicts); `start.ps1`/`start.sh` handle cleanup — respect it.
- Don't switch the LLM backend without keeping `ai_adapter.py` and the `GEMINI_*` env vars consistent (Ollama support has been removed — Gemini/Gemma is the only backend).

