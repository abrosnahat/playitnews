"""
Local web dashboard for PlayItNews post management.
Run:  python webapp.py
Open: http://localhost:5000
"""
import asyncio
import json
import logging
import os
import queue
import re
import shutil
import threading
from typing import Optional

from flask import Flask, Response, abort, jsonify, request, send_file
import mimetypes

import database as db
import ai_adapter
import video_generator
import thumbnail_generator
import instagram_publisher
import youtube_publisher
from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
    TELEGRAM_SECOND_CHANNEL_ID,
    INSTAGRAM_USER_ID_RU,
    INSTAGRAM_ACCESS_TOKEN_RU,
)

app = Flask(__name__, static_folder="static", static_url_path="")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)-12s  %(message)s",
    datefmt="%d.%m %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("playitnews.log", encoding="utf-8"),
    ],
)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# DB migration: add published_platforms column if missing
try:
    with db.get_conn() as _conn:
        _conn.execute("ALTER TABLE scheduled_posts ADD COLUMN published_platforms TEXT DEFAULT '[]'")
except Exception:
    pass  # column already exists

@app.after_request
def _ngrok_headers(response):
    response.headers["ngrok-skip-browser-warning"] = "1"
    return response

# ---------------------------------------------------------------------------
# In-memory task state: post_id -> Queue of SSE event dicts (None = sentinel)
# ---------------------------------------------------------------------------
_task_queues: dict[int, queue.Queue] = {}
_task_logs:   dict[int, list] = {}  # post_id -> [{type, message}, ...]
_cancel_flags: set = set()  # post_ids requested to cancel
_yt_queries:  dict[int, str]  = {}  # post_id -> custom YT search query override
_pub_queues:  dict[int, queue.Queue] = {}  # post_id -> publish progress queue
_pub_logs:    dict[int, list] = {}  # post_id -> publish log entries
_tasks_lock = threading.Lock()

YT_SKIP_STEP = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously in a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _clean_text(text: str) -> str:
    """Strip Telegram HTML and Markdown formatting for plain-text captions."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[*_`~]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) → text
    return text.strip()


def _push_pub(post_id: int, message: str, type_: str = "progress") -> None:
    """Push a publish-progress SSE event."""
    event = {"type": type_, "message": message}
    with _tasks_lock:
        q = _pub_queues.get(post_id)
        _pub_logs.setdefault(post_id, []).append(event)
    if q is not None:
        q.put(event)


def _push(post_id: int, message: str, type_: str = "progress") -> None:
    """Push an SSE event into the post's active queue (if any listener) and persist to log."""
    event = {"type": type_, "message": message}
    with _tasks_lock:
        q = _task_queues.get(post_id)
        _task_logs.setdefault(post_id, []).append(event)
    if q is not None:
        q.put(event)


# ---------------------------------------------------------------------------
# Routes: pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


# ---------------------------------------------------------------------------
# Routes: posts API
# ---------------------------------------------------------------------------

@app.route("/api/posts")
def api_posts():
    status   = request.args.get("status", "active")
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    offset   = (page - 1) * per_page

    with db.get_conn() as conn:
        if status == "all":
            total = conn.execute("SELECT COUNT(*) FROM scheduled_posts").fetchone()[0]
            rows  = conn.execute(
                "SELECT * FROM scheduled_posts ORDER BY id ASC LIMIT ? OFFSET ?",
                (per_page, offset),
            ).fetchall()
        elif status == "active":
            # pending always shown; sent only if published to fewer than 4 platforms
            total = conn.execute(
                """SELECT COUNT(*) FROM scheduled_posts
                   WHERE status = 'pending'
                      OR (status = 'sent'
                          AND json_array_length(COALESCE(published_platforms, '[]')) < 4)"""
            ).fetchone()[0]
            rows  = conn.execute(
                """SELECT * FROM scheduled_posts
                   WHERE status = 'pending'
                      OR (status = 'sent'
                          AND json_array_length(COALESCE(published_platforms, '[]')) < 4)
                   ORDER BY id ASC LIMIT ? OFFSET ?""",
                (per_page, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) FROM scheduled_posts WHERE status = ?", (status,)
            ).fetchone()[0]
            rows  = conn.execute(
                "SELECT * FROM scheduled_posts WHERE status = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (status, per_page, offset),
            ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["image_paths"] = json.loads(d.get("image_paths") or "[]")
        d["video_paths"] = json.loads(d.get("video_paths") or "[]")
        d["image_paths"] = [p for p in d["image_paths"] if os.path.exists(p)]
        d["video_paths"] = [p for p in d["video_paths"] if os.path.exists(p)]
        en_vid = d.get("generated_video_path")
        ru_vid = d.get("generated_video_path_ru")
        d["en_video_exists"] = bool(en_vid and os.path.exists(en_vid))
        d["ru_video_exists"] = bool(ru_vid and os.path.exists(ru_vid))
        d["published_platforms"] = json.loads(d.get("published_platforms") or "[]")
        result.append(d)

    return jsonify({
        "posts":    result,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, -(-total // per_page)),  # ceil division
    })


@app.route("/api/posts/<int:post_id>")
def api_post(post_id: int):
    post = db.get_scheduled_post(post_id)
    if not post:
        abort(404)
    post["generated_video_path"] = db.get_generated_video_path(post_id)
    post["generated_video_path_ru"] = db.get_generated_video_path_ru(post_id)
    post["image_paths"] = [p for p in post.get("image_paths", []) if os.path.exists(p)]
    post["video_paths"] = [p for p in post.get("video_paths", []) if os.path.exists(p)]
    en_vid = post.get("generated_video_path")
    ru_vid = post.get("generated_video_path_ru")
    post["en_video_exists"] = bool(en_vid and os.path.exists(en_vid))
    post["ru_video_exists"] = bool(ru_vid and os.path.exists(ru_vid))
    return jsonify(post)


# ---------------------------------------------------------------------------
# Routes: approve / cancel
# ---------------------------------------------------------------------------

@app.route("/api/posts/<int:post_id>/approve", methods=["POST"])
def api_approve(post_id: int):
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    if post["status"] != "pending":
        return jsonify({"error": f"Post is already '{post['status']}'"}), 400
    try:
        _run_async(_do_publish_telegram(post_id))
        updated = db.get_scheduled_post(post_id)
        return jsonify({"success": True, "status": updated["status"]})
    except Exception as exc:
        logger.exception("Telegram publish failed for post #%d", post_id)
        return jsonify({"error": str(exc)}), 500


async def _do_publish_telegram(post_id: int) -> None:
    from telegram import Bot, InputMediaPhoto, InputMediaVideo
    from telegram.error import TimedOut, TelegramError

    post = db.get_scheduled_post(post_id)
    if not post:
        raise ValueError(f"Post #{post_id} not found")

    text       = post["post_text"]
    ru_text    = post.get("ru_post_text")
    images     = [p for p in post.get("image_paths", [])  if os.path.exists(p)]
    TG_MAX_BYTES = 50 * 1024 * 1024  # Telegram Bot API hard limit

    def _compress_for_tg(src: str) -> str:
        """Compress video to fit under 50 MB using ffmpeg bitrate targeting. Returns path to compressed file."""
        import subprocess as _sp, json as _json
        # Probe duration
        r = _sp.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "json", src],
            capture_output=True, text=True, timeout=15,
        )
        duration = float(_json.loads(r.stdout).get("format", {}).get("duration", 0) or 0)
        if duration <= 0:
            return src  # can't probe, send as-is
        target_mb = 45  # conservative target to stay safely under 50MB
        audio_kbps = 128
        total_kbits = target_mb * 8 * 1024
        video_kbps = max(200, int(total_kbits / duration) - audio_kbps)
        out = src.rsplit(".", 1)[0] + "_tg.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-c:v", "libx264", "-preset", "fast",
            "-b:v", f"{video_kbps}k",
            "-maxrate", f"{int(video_kbps * 1.5)}k",
            "-bufsize", f"{video_kbps * 3}k",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k",
            "-movflags", "+faststart",
            out,
        ]
        result = _sp.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(out) and os.path.getsize(out) < TG_MAX_BYTES:
            logger.info("Compressed %s → %s (%.1fMB)", src, out, os.path.getsize(out)/1024/1024)
            return out
        logger.warning("Compression failed or still too large for %s, returncode=%s stderr=%s",
                       src, result.returncode, result.stderr[-500:] if result.stderr else "")
        return src  # fallback to original

    raw_videos = [p for p in post.get("video_paths", []) if os.path.exists(p)]
    videos = []
    for vp in raw_videos:
        if os.path.getsize(vp) > TG_MAX_BYTES:
            logger.info("Post #%d: video %s is %.1fMB, compressing…", post_id, vp, os.path.getsize(vp)/1024/1024)
            vp = _compress_for_tg(vp)
        if os.path.getsize(vp) <= TG_MAX_BYTES:
            videos.append(vp)
        else:
            logger.warning("Post #%d: skipping video %s — still >50MB after compression", post_id, vp)

    def _probe_dims(path: str):
        """Return (width, height) of a video using ffprobe, or (None, None) on failure."""
        try:
            import subprocess as _sp, json as _json
            r = _sp.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "json", path],
                capture_output=True, text=True, timeout=10,
            )
            d = _json.loads(r.stdout)
            s = d.get("streams", [{}])[0]
            return s.get("width"), s.get("height")
        except Exception:
            return None, None

    async def _send(chat_id, caption, footer=None):
        full_caption = caption
        if footer:
            full_caption = caption.rstrip() + f"\n\n{footer}"
        # Hard-trim to Telegram caption limit (1024) at last newline boundary
        if len(full_caption) > 1024:
            cut = full_caption[:1024]
            last_nl = cut.rfind("\n")
            full_caption = cut[:last_nl].rstrip() if last_nl > 512 else cut.rstrip()
        from telegram.constants import ParseMode
        if images or videos:
            media = []
            for i, p in enumerate(images):
                media.append(InputMediaPhoto(
                    media=open(p, "rb"),
                    caption=full_caption if i == 0 else None,
                    parse_mode=ParseMode.HTML if i == 0 else None,
                ))
            for j, p in enumerate(videos):
                w, h = _probe_dims(p)
                media.append(InputMediaVideo(
                    media=open(p, "rb"),
                    caption=full_caption if not images and j == 0 else None,
                    parse_mode=ParseMode.HTML if not images and j == 0 else None,
                    width=w, height=h,
                    supports_streaming=True,
                ))
            if len(media) == 1:
                if images:
                    await bot.send_photo(chat_id=chat_id, photo=open(images[0], "rb"),
                                         caption=full_caption, parse_mode=ParseMode.HTML)
                else:
                    w, h = _probe_dims(videos[0])
                    await bot.send_video(chat_id=chat_id, video=open(videos[0], "rb"),
                                         caption=full_caption, parse_mode=ParseMode.HTML,
                                         width=w, height=h, supports_streaming=True)
            else:
                await bot.send_media_group(chat_id=chat_id, media=media)
        else:
            await bot.send_message(chat_id=chat_id, text=full_caption,
                                   parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
        try:
            await _send(TELEGRAM_CHANNEL_ID, text, footer="@playitgamesnews")
            if TELEGRAM_SECOND_CHANNEL_ID and ru_text:
                try:
                    await _send(TELEGRAM_SECOND_CHANNEL_ID, ru_text, footer="@readitgames")
                except TelegramError as e:
                    logger.error("Second channel publish failed for #%d: %s", post_id, e)
            db.update_post_status(post_id, "sent")
        except TimedOut:
            db.update_post_status(post_id, "sent")  # Telegram accepted it
        except TelegramError as exc:
            raise


@app.route("/api/posts/<int:post_id>/cancel", methods=["POST"])
def api_cancel(post_id: int):
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    if post["status"] not in ("pending", "approved"):
        return jsonify({"error": f"Cannot cancel post with status '{post['status']}'"}), 400
    db.update_post_status(post_id, "cancelled")
    return jsonify({"success": True, "status": "cancelled"})


@app.route("/api/posts/<int:post_id>/mark-done", methods=["POST"])
def api_mark_done(post_id: int):
    """Mark a sent post as fully published (all 4 platforms) so it leaves the active feed."""
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    all_platforms = json.dumps(["instagram", "instagram-ru", "youtube", "youtube-ru"])
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE scheduled_posts SET published_platforms = ? WHERE id = ?",
            (all_platforms, post_id),
        )
    return jsonify({"success": True})


@app.route("/api/posts/<int:post_id>/republish-ru", methods=["POST"])
def api_republish_ru(post_id: int):
    """Re-send ru_post_text to @readitgames (useful when the second channel failed silently)."""
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    if post["status"] != "sent":
        return jsonify({"error": f"Post status is '{post['status']}', expected 'sent'"}), 400
    ru_text = post.get("ru_post_text")
    if not ru_text:
        return jsonify({"error": "No ru_post_text for this post"}), 400
    try:
        _run_async(_do_republish_ru(post_id))
        return jsonify({"success": True})
    except Exception as exc:
        logger.exception("Republish RU failed for post #%d", post_id)
        return jsonify({"error": str(exc)}), 500


async def _do_republish_ru(post_id: int) -> None:
    from telegram import Bot, InputMediaPhoto, InputMediaVideo
    from telegram.error import TelegramError

    post = db.get_scheduled_post(post_id)
    ru_text = post["ru_post_text"]
    images  = [p for p in post.get("image_paths", [])  if os.path.exists(p)]
    videos  = [p for p in post.get("video_paths", [])  if os.path.exists(p)]
    if post.get("generated_video_path_ru") and os.path.exists(post["generated_video_path_ru"]):
        videos = [post["generated_video_path_ru"]]
        images = []

    full_caption = ru_text.rstrip() + "\n\n@readitgames"
    if len(full_caption) > 1024:
        cut = full_caption[:1024]
        last_nl = cut.rfind("\n")
        full_caption = cut[:last_nl].rstrip() if last_nl > 512 else cut.rstrip()

    from telegram.constants import ParseMode

    def _probe_dims(p):
        try:
            import subprocess, json as _j
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", p],
                capture_output=True, text=True, timeout=10,
            )
            s = next((s for s in _j.loads(r.stdout).get("streams", []) if s.get("codec_type") == "video"), {})
            return s.get("width"), s.get("height")
        except Exception:
            return None, None

    async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
        if images or videos:
            media = []
            for i, p in enumerate(images):
                media.append(InputMediaPhoto(
                    media=open(p, "rb"),
                    caption=full_caption if i == 0 else None,
                    parse_mode=ParseMode.HTML if i == 0 else None,
                ))
            for j, p in enumerate(videos):
                w, h = _probe_dims(p)
                media.append(InputMediaVideo(
                    media=open(p, "rb"),
                    caption=full_caption if not images and j == 0 else None,
                    parse_mode=ParseMode.HTML if not images and j == 0 else None,
                    width=w, height=h, supports_streaming=True,
                ))
            if len(media) == 1:
                if images:
                    await bot.send_photo(chat_id=TELEGRAM_SECOND_CHANNEL_ID, photo=open(images[0], "rb"),
                                         caption=full_caption, parse_mode=ParseMode.HTML)
                else:
                    w, h = _probe_dims(videos[0])
                    await bot.send_video(chat_id=TELEGRAM_SECOND_CHANNEL_ID, video=open(videos[0], "rb"),
                                         caption=full_caption, parse_mode=ParseMode.HTML,
                                         width=w, height=h, supports_streaming=True)
            else:
                await bot.send_media_group(chat_id=TELEGRAM_SECOND_CHANNEL_ID, media=media)
        else:
            await bot.send_message(chat_id=TELEGRAM_SECOND_CHANNEL_ID, text=full_caption,
                                   parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    logger.info("Post #%d republished to @readitgames", post_id)


# ---------------------------------------------------------------------------
# Route: serve local media files (images / videos)
# ---------------------------------------------------------------------------

@app.route("/api/media")
def api_media():
    path = request.args.get("path", "")
    base = os.path.abspath(os.path.dirname(__file__))
    full = os.path.abspath(path)
    # Security: only serve files within the project directory
    if not full.startswith(base):
        abort(403)
    if not os.path.isfile(full):
        abort(404)

    mime = mimetypes.guess_type(full)[0] or "application/octet-stream"
    file_size = os.path.getsize(full)
    range_header = request.headers.get("Range")

    if range_header:
        # Parse "bytes=start-end"
        try:
            byte_range = range_header.replace("bytes=", "").strip()
            start_str, _, end_str = byte_range.partition("-")
            start = int(start_str) if start_str else 0
            end   = int(end_str)   if end_str   else file_size - 1
        except ValueError:
            abort(400)
        end = min(end, file_size - 1)
        length = end - start + 1
        with open(full, "rb") as f:
            f.seek(start)
            data = f.read(length)
        resp = Response(data, 206, mimetype=mime, direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    resp = send_file(full, mimetype=mime)
    resp.headers["Accept-Ranges"]  = "bytes"
    resp.headers["Content-Length"] = str(file_size)
    return resp


# ---------------------------------------------------------------------------
# Routes: video generation (background thread + SSE progress)
# ---------------------------------------------------------------------------

@app.route("/api/posts/<int:post_id>/generate-video", methods=["POST"])
def api_generate_video(post_id: int):
    body = request.get_json(silent=True) or {}
    lang = body.get("lang", "both")
    if lang not in ("en", "ru", "both"):
        return jsonify({"error": "lang must be 'en', 'ru', or 'both'"}), 400

    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    # Optional custom YT search query override
    yt_query_override = (body.get("yt_query") or "").strip() or None
    if yt_query_override:
        with _tasks_lock:
            _yt_queries[post_id] = yt_query_override

    with _tasks_lock:
        if post_id in _task_queues:
            # Check if the queue has a sentinel already (task finished but not cleaned up)
            q_existing = _task_queues[post_id]
            # Drain to see if it's stale (has None sentinel or items but no active thread)
            active_threads = {t.name for t in threading.enumerate()}
            if f"vidgen-{post_id}" not in active_threads:
                # Thread is gone — stale queue, remove it
                _task_queues.pop(post_id, None)
            else:
                return jsonify({"error": "Video generation already in progress for this post"}), 409
        q: queue.Queue = queue.Queue()
        _task_queues[post_id] = q
        _task_logs[post_id] = []  # reset log for new task

    def _run():
        try:
            _run_async(_generate_video(post_id, lang))
        except Exception as exc:
            _push(post_id, f"Fatal error: {exc}", "error")
        finally:
            # Put sentinel so SSE generator exits
            with _tasks_lock:
                q_ref = _task_queues.get(post_id)
            if q_ref is not None:
                q_ref.put(None)

    threading.Thread(target=_run, daemon=True, name=f"vidgen-{post_id}").start()
    return jsonify({"success": True, "message": "Video generation started"})


@app.route("/api/posts/<int:post_id>/task-status")
def api_task_status(post_id: int):
    """Return saved log entries and whether task is still running."""
    with _tasks_lock:
        running = post_id in _task_queues
        active_threads = {t.name for t in threading.enumerate()}
        if running and f"vidgen-{post_id}" not in active_threads:
            running = False
            _task_queues.pop(post_id, None)
        logs = list(_task_logs.get(post_id, []))
    return jsonify({"running": running, "logs": logs})


@app.route("/api/posts/<int:post_id>/reset-yt-skip", methods=["POST"])
def api_reset_yt_skip(post_id: int):
    """Reset the YouTube skip counter to 0."""
    with db.get_conn() as conn:
        conn.execute("UPDATE scheduled_posts SET yt_skip_count = 0 WHERE id = ?", (post_id,))
    return jsonify({"success": True})


@app.route("/api/posts/<int:post_id>/cancel-video", methods=["POST"])
def api_cancel_video(post_id: int):
    """Request cancellation of an in-progress video generation task."""
    with _tasks_lock:
        _cancel_flags.add(post_id)
    _push(post_id, "⛔ Cancellation requested…", "progress")
    return jsonify({"success": True})


@app.route("/api/posts/<int:post_id>/reset-task", methods=["POST"])
def api_reset_task(post_id: int):
    """Force-clear a stuck video generation task."""
    with _tasks_lock:
        _task_queues.pop(post_id, None)
    return jsonify({"success": True})


@app.route("/api/posts/<int:post_id>/video-stream")
def api_video_stream(post_id: int):
    """SSE endpoint — streams video generation progress to the browser."""
    with _tasks_lock:
        q = _task_queues.get(post_id)

    if q is None:
        def _empty():
            yield f"data: {json.dumps({'type': 'done', 'message': 'No active task'})}\n\n"
        return Response(_empty(), mimetype="text/event-stream")

    def _generate():
        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            if event is None:
                # Sentinel: task finished — clean up and close
                with _tasks_lock:
                    _task_queues.pop(post_id, None)
                yield f"data: {json.dumps({'type': 'closed'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _generate_video(post_id: int, lang: str) -> None:
    """Core video generation logic (mirrors handle_create_video in bot.py)."""
    post = db.get_scheduled_post(post_id)
    if not post:
        _push(post_id, "Post not found", "error")
        return

    def progress(msg: str, type_: str = "progress") -> None:
        _push(post_id, msg, type_)
        logger.info("[vidgen #%d] %s", post_id, msg)

    def check_cancel():
        with _tasks_lock:
            if post_id in _cancel_flags:
                _cancel_flags.discard(post_id)
                return True
        return False

    try:
        do_en = lang in ("en", "both")
        do_ru = lang in ("ru", "both")

        # --- Step 1: Write scripts ---
        progress("Step 1/4: Writing scripts…")
        if check_cancel(): raise InterruptedError("Cancelled")
        en_script: Optional[str] = None
        ru_script: Optional[str] = None

        if do_en:
            en_title = await ai_adapter.translate_title_to_english(post["article_title"])
            en_script = await ai_adapter.generate_video_script(
                post_text=post["post_text"], article_title=en_title, lang="en"
            )
            en_script = re.sub(r'<[^>]+>', '', en_script or '').strip()
            fallback_en = re.sub(r'<[^>]+>', '', post["post_text"][:350]).strip()
            if len(en_script.split()) < 20:
                en_script = fallback_en
            progress(f"🇬🇧 EN script: {len(en_script.split())} words")

        if do_ru:
            ru_script = await ai_adapter.generate_video_script(
                post_text=post.get("ru_post_text") or post["post_text"],
                article_title=post["article_title"],
                lang="ru",
            )
            ru_script = re.sub(r'<[^>]+>', '', ru_script or '').strip()
            fallback_ru = re.sub(r'<[^>]+>', '', (post.get("ru_post_text") or post["post_text"])[:350]).strip()
            if len(ru_script.split()) < 20:
                ru_script = fallback_ru
            progress(f"🇷🇺 RU script: {len(ru_script.split())} words")

        # --- Step 2: Determine search query ---
        progress("Step 2/4: Synthesizing voices…")
        if check_cancel(): raise InterruptedError("Cancelled")
        title = post.get("article_title", "")
        with _tasks_lock:
            search_query = _yt_queries.pop(post_id, None)
        user_query = bool(search_query)
        if not search_query:
            search_query = await ai_adapter.extract_game_name(title)
        if not search_query:
            first_chunk = re.split(r"[:–—|]", title)[0].strip()
            latin_tokens = [t for t in first_chunk.split() if re.match(r"^[A-Za-z0-9'&.]+$", t)]
            search_query = " ".join(latin_tokens).strip() or title[:50]

        # --- Step 3: Fetch gameplay clips ---
        progress(f"Step 3/4: Searching YouTube for '{search_query}'…")
        if check_cancel(): raise InterruptedError("Cancelled")
        yt_skip = db.increment_yt_skip(post_id, YT_SKIP_STEP) - YT_SKIP_STEP
        article_videos, yt_clips, clips_workdir = await video_generator.fetch_gameplay_clips(
            post=post, search_query=search_query, yt_skip=yt_skip, user_query=user_query,
        )
        shared_clips = article_videos + yt_clips
        progress(f"Found {len(shared_clips)} video clips. Rendering…")

        # --- Step 4: Render ---
        if check_cancel(): raise InterruptedError("Cancelled")
        en_path: Optional[str] = None
        ru_path: Optional[str] = None
        try:
            if do_en:
                progress("Rendering 🇬🇧 EN video…")
                en_path = await video_generator.create_short_video(
                    post=post,
                    script=en_script,
                    search_query=search_query,
                    yt_skip=yt_skip,
                    lang="en",
                    prefetched_clips=shared_clips,
                    n_article_clips=len(article_videos),
                )
                if en_path:
                    db.set_generated_video_path(post_id, en_path)
                    progress(f"✅ EN video: {os.path.basename(en_path)}")
                else:
                    progress("❌ EN video generation failed", "error")

            if do_ru:
                progress("Rendering 🇷🇺 RU video…")
                ru_path = await video_generator.create_short_video(
                    post=post,
                    script=ru_script,
                    search_query=search_query,
                    yt_skip=yt_skip,
                    lang="ru",
                    prefetched_clips=shared_clips,
                    n_article_clips=len(article_videos),
                )
                if ru_path:
                    db.set_generated_video_path_ru(post_id, ru_path)
                    progress(f"✅ RU video: {os.path.basename(ru_path)}")
                else:
                    progress("❌ RU video generation failed", "error")

        finally:
            shutil.rmtree(clips_workdir, ignore_errors=True)

        progress("Step 4/4: Done!", "done")
        _push(post_id, json.dumps({
            "en_path": en_path,
            "ru_path": ru_path,
        }), "video_ready")

    except InterruptedError:
        progress("⛔ Generation cancelled", "error")
    except Exception as exc:
        progress(f"Error: {exc}", "error")
        raise


# ---------------------------------------------------------------------------
# Route: publish to social media
# ---------------------------------------------------------------------------

@app.route("/api/posts/<int:post_id>/publish/<platform>", methods=["POST"])
def api_publish(post_id: int, platform: str):
    valid = {"instagram", "instagram-ru", "youtube", "youtube-ru", "all", "all-ru", "all-combined"}
    if platform not in valid:
        return jsonify({"error": f"Unknown platform '{platform}'"}), 400
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    with _tasks_lock:
        q: queue.Queue = queue.Queue()
        _pub_queues[post_id] = q
        _pub_logs[post_id] = []

    def _run():
        def progress(msg: str, type_: str = "progress"):
            _push_pub(post_id, msg, type_)
            logger.info("[publish #%d %s] %s", post_id, platform, msg)
        try:
            progress(f"Publishing to {platform}…")
            result = _run_async(_do_publish_social(post_id, platform, post, progress_cb=progress))
            progress(f"✅ Done: {result}", "done")
            # Determine which individual platforms actually succeeded
            _PLATFORM_KEYS = {
                "instagram": "instagram", "instagram-ru": "instagram-ru",
                "youtube": "youtube", "youtube-ru": "youtube-ru",
            }
            if platform in _PLATFORM_KEYS:
                succeeded = [platform]
            else:
                # all / all-ru / all-combined — parse result string for ✅ lines
                succeeded = []
                label_map = {
                    "Instagram EN": "instagram", "Instagram RU": "instagram-ru",
                    "YouTube EN":   "youtube",   "YouTube RU":   "youtube-ru",
                }
                for part in result.split(" | "):
                    if part.startswith("✅"):
                        for label, key in label_map.items():
                            if label in part:
                                succeeded.append(key)
                                break
            if succeeded:
                with db.get_conn() as conn:
                    existing = conn.execute(
                        "SELECT published_platforms FROM scheduled_posts WHERE id = ?", (post_id,)
                    ).fetchone()
                    platforms_list = json.loads((existing[0] if existing else None) or "[]")
                    for s in succeeded:
                        if s not in platforms_list:
                            platforms_list.append(s)
                    conn.execute(
                        "UPDATE scheduled_posts SET published_platforms = ? WHERE id = ?",
                        (json.dumps(platforms_list), post_id),
                    )
        except Exception as exc:
            progress(f"❌ Error: {exc}", "error")
            logger.exception("Social publish failed: post #%d platform %s", post_id, platform)
        finally:
            with _tasks_lock:
                q_ref = _pub_queues.get(post_id)
            if q_ref is not None:
                q_ref.put(None)

    threading.Thread(target=_run, daemon=True, name=f"publish-{post_id}").start()
    return jsonify({"success": True})


@app.route("/api/posts/<int:post_id>/publish-stream")
def api_publish_stream(post_id: int):
    """SSE endpoint for publish progress."""
    with _tasks_lock:
        q = _pub_queues.get(post_id)
    if q is None:
        def _empty():
            yield f"data: {json.dumps({'type': 'done', 'message': 'No active publish'})}\n\n"
        return Response(_empty(), mimetype="text/event-stream")

    def _generate():
        while True:
            try:
                event = q.get(timeout=60)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            if event is None:
                with _tasks_lock:
                    _pub_queues.pop(post_id, None)
                yield f"data: {json.dumps({'type': 'closed'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _do_publish_social(post_id: int, platform: str, post: dict, progress_cb=None) -> str:
    def progress(msg: str, type_: str = "progress"):
        if progress_cb:
            progress_cb(msg, type_)
    en_path = db.get_generated_video_path(post_id)
    ru_path = db.get_generated_video_path_ru(post_id)

    def _caption_en() -> str:
        raw = _clean_text(post.get("post_text", ""))
        ht = " ".join(re.findall(r'#\w+', re.sub(r'<[^>]+>', '', post.get("post_text", ""))))
        if ht and ht not in raw:
            raw = raw.rstrip() + "\n\n" + ht
        return "More news in the telegram channel, link in the bio\n\n" + raw

    def _caption_ru() -> str:
        raw = _clean_text(post.get("ru_post_text") or post.get("post_text", ""))
        src = post.get("ru_post_text") or post.get("post_text", "")
        ht = " ".join(re.findall(r'#\w+', re.sub(r'<[^>]+>', '', src)))
        if ht and ht not in raw:
            raw = raw.rstrip() + "\n\n" + ht
        return "Больше новостей в telegram-канале, ссылка в био\n\n" + raw

    def _tags(key: str = "post_text") -> list[str]:
        src = re.sub(r'<[^>]+>', '', post.get(key) or post.get("post_text", ""))
        return [h.lstrip("#") for h in re.findall(r'#\w+', src)]

    async def _make_thumb(path: str, lang: str = "en", title: Optional[str] = None) -> Optional[str]:
        try:
            image_paths: list[str] = [p for p in post.get("image_paths", []) if os.path.exists(p)]
            if not image_paths:
                return None
            source = image_paths[0]
            raw_title = title or post.get("article_title", "")
            if lang == "en" and (not title or re.search(r'[а-яёА-ЯЁ]', raw_title)):
                translated = await ai_adapter.translate_title_to_english(raw_title)
                if translated and not re.search(r'[а-яёА-ЯЁ]', translated):
                    raw_title = translated
            hook = await ai_adapter.generate_thumbnail_hook(raw_title, lang=lang)
            # Final safety net: never burn Cyrillic text onto an EN thumbnail.
            if lang == "en" and (not hook or re.search(r'[а-яёА-ЯЁ]', hook)):
                logger.warning("_make_thumb EN: hook is empty or contains Cyrillic ('%s') — skipping thumbnail", hook[:60])
                return None
            base = os.path.splitext(path)[0]
            out_path = f"{base}_thumb_{lang}.jpg"
            ok = thumbnail_generator.generate_instagram_thumbnail(source, hook, out_path)
            return out_path if ok else None
        except Exception:
            return None

    if platform == "instagram":
        if not en_path or not os.path.exists(en_path):
            raise ValueError("EN video not found — generate video first.")
        progress("Translating title to English…")
        en_title = await ai_adapter.translate_title_to_english(post.get("article_title", ""))
        # Only pass en_title if it's actually in English (no Cyrillic).
        _en_title_clean = en_title if en_title and not re.search(r'[а-яёА-ЯЁ]', en_title) else None
        progress("Generating thumbnail…")
        thumb = await _make_thumb(en_path, lang="en", title=_en_title_clean)
        progress("Uploading reel to Instagram…")
        media_id = await instagram_publisher.publish_reel(
            video_path=en_path, caption=_caption_en(), cover_image_path=thumb,
        )
        return f"Instagram EN — Media ID: {media_id}"

    elif platform == "instagram-ru":
        if not ru_path or not os.path.exists(ru_path):
            raise ValueError("RU video not found — generate video first.")
        progress("Generating thumbnail…")
        thumb = await _make_thumb(ru_path, lang="ru")
        progress("Uploading reel to Instagram RU…")
        media_id = await instagram_publisher.publish_reel(
            video_path=ru_path, caption=_caption_ru(),
            user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
            cover_image_path=thumb,
        )
        return f"Instagram RU — Media ID: {media_id}"

    elif platform == "youtube":
        if not en_path or not os.path.exists(en_path):
            raise ValueError("EN video not found — generate video first.")
        progress("Translating title to English…")
        _raw_yt_title = await ai_adapter.translate_title_to_english(
            post.get("article_title", "") or f"Gaming news #{post_id}"
        )
        yt_title = (_raw_yt_title if _raw_yt_title and not re.search(r'[а-яёА-ЯЁ]', _raw_yt_title) else "Gaming News")[:100]
        progress("Uploading Short to YouTube…")
        desc = "More news in the telegram channel, link in the bio\n\n" + _clean_text(post.get("post_text", ""))
        video_id = await youtube_publisher.upload_short(
            video_path=en_path, title=yt_title, description=desc, tags=_tags(),
        )
        return f"YouTube EN — https://youtu.be/{video_id}"

    elif platform == "youtube-ru":
        if not ru_path or not os.path.exists(ru_path):
            raise ValueError("RU video not found — generate video first.")
        yt_title = (post.get("article_title") or f"Gaming news #{post_id}")[:100]
        progress("Uploading Short to YouTube RU…")
        desc = "Больше новостей в telegram-канале, ссылка в bio\n\n" + _clean_text(
            post.get("ru_post_text") or post.get("post_text", "")
        )
        video_id = await youtube_publisher.upload_short_ru(
            video_path=ru_path, title=yt_title, description=desc, tags=_tags("ru_post_text"),
        )
        return f"YouTube RU — https://youtu.be/{video_id}"

    elif platform in ("all", "all-ru", "all-combined"):
        tasks: dict[str, asyncio.Task] = {}
        results: list[str] = []

        if platform in ("all", "all-combined") and en_path and os.path.exists(en_path):
            progress("Translating EN title…")
            _raw_en_title = (await ai_adapter.translate_title_to_english(
                post.get("article_title", "") or f"Gaming news #{post_id}"
            ))[:100]
            # Only pass as title if it's actually English (no Cyrillic).
            en_title = (_raw_en_title if _raw_en_title and not re.search(r'[а-яёА-ЯЁ]', _raw_en_title) else "Gaming News")[:100]
            progress("Generating EN thumbnail…")
            en_thumb = await _make_thumb(en_path, lang="en", title=en_title)
            en_desc = "More news in the telegram channel, link in the bio\n\n" + _clean_text(post.get("post_text", ""))
            if instagram_publisher.is_configured():
                progress("Starting Instagram EN upload…")
                tasks["Instagram EN"] = asyncio.create_task(
                    instagram_publisher.publish_reel(
                        video_path=en_path, caption=_caption_en(), cover_image_path=en_thumb,
                    )
                )
            if youtube_publisher.is_configured():
                progress("Starting YouTube EN upload…")
                tasks["YouTube EN"] = asyncio.create_task(
                    youtube_publisher.upload_short(
                        video_path=en_path, title=en_title, description=en_desc, tags=_tags(),
                    )
                )

        if platform in ("all-ru", "all-combined") and ru_path and os.path.exists(ru_path):
            ru_title = (post.get("article_title") or f"Gaming news #{post_id}")[:100]
            progress("Generating RU thumbnail…")
            ru_thumb = await _make_thumb(ru_path, lang="ru")
            if instagram_publisher.is_configured_ru():
                progress("Starting Instagram RU upload…")
                tasks["Instagram RU"] = asyncio.create_task(
                    instagram_publisher.publish_reel(
                        video_path=ru_path, caption=_caption_ru(),
                        user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
                        cover_image_path=ru_thumb,
                    )
                )
            if youtube_publisher.is_configured_ru():
                progress("Starting YouTube RU upload…")
                tasks["YouTube RU"] = asyncio.create_task(
                    youtube_publisher.upload_short_ru(
                        video_path=ru_path, title=ru_title,
                        description=_caption_ru(), tags=_tags("ru_post_text"),
                    )
                )

        for name, task in tasks.items():
            try:
                r = await task
                if "YouTube" in name and isinstance(r, str):
                    results.append(f"✅ {name}: https://youtu.be/{r}")
                else:
                    results.append(f"✅ {name}: {r}")
            except Exception as exc:
                results.append(f"❌ {name}: {exc}")
                logger.error("%s failed for post #%d: %s", name, post_id, exc)

        return " | ".join(results) or "No platforms configured"

    return "Done"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db.init_db()
    os.makedirs("static", exist_ok=True)
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print("=" * 50)
    print(f"  PlayItNews Dashboard")
    print(f"  Local:   http://localhost:5001")
    print(f"  Network: http://{local_ip}:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
