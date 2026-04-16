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

import database as db
import ai_adapter
import video_generator
import instagram_publisher
import youtube_publisher
from config import (
    TELEGRAM_BOT_TOKEN,
    INSTAGRAM_USER_ID_RU,
    INSTAGRAM_ACCESS_TOKEN_RU,
)

app = Flask(__name__, static_folder="static", static_url_path="")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
)

# ---------------------------------------------------------------------------
# In-memory task state: post_id -> Queue of SSE event dicts (None = sentinel)
# ---------------------------------------------------------------------------
_task_queues: dict[int, queue.Queue] = {}
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


def _push(post_id: int, message: str, type_: str = "progress") -> None:
    """Push an SSE event into the post's active queue (if any listener)."""
    with _tasks_lock:
        q = _task_queues.get(post_id)
    if q is not None:
        q.put({"type": type_, "message": message})


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
    status = request.args.get("status", "all")
    with db.get_conn() as conn:
        if status == "all":
            rows = conn.execute(
                "SELECT * FROM scheduled_posts ORDER BY id DESC LIMIT 500"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM scheduled_posts WHERE status = ? ORDER BY id DESC",
                (status,),
            ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["image_paths"] = json.loads(d.get("image_paths") or "[]")
        d["video_paths"] = json.loads(d.get("video_paths") or "[]")
        d["image_paths"] = [p for p in d["image_paths"] if os.path.exists(p)]
        d["video_paths"] = [p for p in d["video_paths"] if os.path.exists(p)]
        result.append(d)
    return jsonify(result)


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
    from telegram import Bot
    import bot as bot_module
    async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
        await bot_module.publish_post(bot, post_id)


@app.route("/api/posts/<int:post_id>/cancel", methods=["POST"])
def api_cancel(post_id: int):
    post = db.get_scheduled_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    if post["status"] not in ("pending", "approved"):
        return jsonify({"error": f"Cannot cancel post with status '{post['status']}'"}), 400
    db.update_post_status(post_id, "cancelled")
    return jsonify({"success": True, "status": "cancelled"})


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
    return send_file(full)


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

    with _tasks_lock:
        if post_id in _task_queues:
            return jsonify({"error": "Video generation already in progress for this post"}), 409
        q: queue.Queue = queue.Queue()
        _task_queues[post_id] = q

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

    try:
        do_en = lang in ("en", "both")
        do_ru = lang in ("ru", "both")

        # --- Step 1: Write scripts ---
        progress("Step 1/4: Writing scripts…")
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
        title = post.get("article_title", "")
        search_query = await ai_adapter.extract_game_name(title)
        if not search_query:
            first_chunk = re.split(r"[:–—|]", title)[0].strip()
            latin_tokens = [t for t in first_chunk.split() if re.match(r"^[A-Za-z0-9'&.]+$", t)]
            search_query = " ".join(latin_tokens).strip() or title[:50]

        # --- Step 3: Fetch gameplay clips ---
        progress(f"Step 3/4: Searching YouTube for '{search_query}'…")
        yt_skip = db.increment_yt_skip(post_id, YT_SKIP_STEP) - YT_SKIP_STEP
        article_videos, yt_clips, clips_workdir = await video_generator.fetch_gameplay_clips(
            post=post, search_query=search_query, yt_skip=yt_skip,
        )
        shared_clips = article_videos + yt_clips
        progress(f"Found {len(shared_clips)} video clips. Rendering…")

        # --- Step 4: Render ---
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
    try:
        result = _run_async(_do_publish_social(post_id, platform, post))
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        logger.exception("Social publish failed: post #%d platform %s", post_id, platform)
        return jsonify({"error": str(exc)}), 500


async def _do_publish_social(post_id: int, platform: str, post: dict) -> str:
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
            from bot import _make_thumbnail
            return await _make_thumbnail(post, path, title=title, lang=lang)
        except Exception:
            return None

    if platform == "instagram":
        if not en_path or not os.path.exists(en_path):
            raise ValueError("EN video not found — generate video first.")
        en_title = await ai_adapter.translate_title_to_english(post.get("article_title", ""))
        thumb = await _make_thumb(en_path, lang="en", title=en_title or None)
        media_id = await instagram_publisher.publish_reel(
            video_path=en_path, caption=_caption_en(), cover_image_path=thumb,
        )
        return f"Instagram EN — Media ID: {media_id}"

    elif platform == "instagram-ru":
        if not ru_path or not os.path.exists(ru_path):
            raise ValueError("RU video not found — generate video first.")
        thumb = await _make_thumb(ru_path, lang="ru")
        media_id = await instagram_publisher.publish_reel(
            video_path=ru_path, caption=_caption_ru(),
            user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
            cover_image_path=thumb,
        )
        return f"Instagram RU — Media ID: {media_id}"

    elif platform == "youtube":
        if not en_path or not os.path.exists(en_path):
            raise ValueError("EN video not found — generate video first.")
        yt_title = (await ai_adapter.translate_title_to_english(
            post.get("article_title", "") or f"Gaming news #{post_id}"
        ))[:100]
        desc = "More news in the telegram channel, link in the bio\n\n" + _clean_text(post.get("post_text", ""))
        video_id = await youtube_publisher.upload_short(
            video_path=en_path, title=yt_title, description=desc, tags=_tags(),
        )
        return f"YouTube EN — https://youtu.be/{video_id}"

    elif platform == "youtube-ru":
        if not ru_path or not os.path.exists(ru_path):
            raise ValueError("RU video not found — generate video first.")
        yt_title = (post.get("article_title") or f"Gaming news #{post_id}")[:100]
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
            en_title = (await ai_adapter.translate_title_to_english(
                post.get("article_title", "") or f"Gaming news #{post_id}"
            ))[:100]
            en_thumb = await _make_thumb(en_path, lang="en", title=en_title)
            en_desc = "More news in the telegram channel, link in the bio\n\n" + _clean_text(post.get("post_text", ""))
            if instagram_publisher.is_configured():
                tasks["Instagram EN"] = asyncio.create_task(
                    instagram_publisher.publish_reel(
                        video_path=en_path, caption=_caption_en(), cover_image_path=en_thumb,
                    )
                )
            if youtube_publisher.is_configured():
                tasks["YouTube EN"] = asyncio.create_task(
                    youtube_publisher.upload_short(
                        video_path=en_path, title=en_title, description=en_desc, tags=_tags(),
                    )
                )

        if platform in ("all-ru", "all-combined") and ru_path and os.path.exists(ru_path):
            ru_title = (post.get("article_title") or f"Gaming news #{post_id}")[:100]
            ru_thumb = await _make_thumb(ru_path, lang="ru")
            if instagram_publisher.is_configured_ru():
                tasks["Instagram RU"] = asyncio.create_task(
                    instagram_publisher.publish_reel(
                        video_path=ru_path, caption=_caption_ru(),
                        user_id=INSTAGRAM_USER_ID_RU, access_token=INSTAGRAM_ACCESS_TOKEN_RU,
                        cover_image_path=ru_thumb,
                    )
                )
            if youtube_publisher.is_configured_ru():
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
    print("=" * 50)
    print("  PlayItNews Dashboard  →  http://localhost:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
