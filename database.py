import sqlite3
import json
from datetime import datetime
from typing import Optional
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS seen_articles (
                url TEXT PRIMARY KEY,
                title TEXT,
                seen_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scheduled_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_url TEXT,
                article_title TEXT,
                post_text TEXT,
                image_paths TEXT,        -- JSON array of local file paths
                video_paths TEXT DEFAULT '[]', -- JSON array of local video file paths
                scheduled_at TEXT,       -- ISO datetime
                status TEXT DEFAULT 'pending',  -- pending | approved | cancelled | sent
                notification_message_id INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        # Migration: add video_paths column to existing databases
        try:
            conn.execute("ALTER TABLE scheduled_posts ADD COLUMN video_paths TEXT DEFAULT '[]'")
        except Exception:
            pass  # column already exists
        # Migration: add generated_video_path column
        try:
            conn.execute("ALTER TABLE scheduled_posts ADD COLUMN generated_video_path TEXT DEFAULT NULL")
        except Exception:
            pass  # column already exists


# --- seen articles ---

def is_article_seen(url: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM seen_articles WHERE url = ?", (url,)).fetchone()
        return row is not None


def mark_article_seen(url: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (url, title) VALUES (?, ?)",
            (url, title),
        )


# --- scheduled posts ---

def create_scheduled_post(
    article_url: str,
    article_title: str,
    post_text: str,
    image_paths: list[str],
    scheduled_at: datetime,
    video_paths: list[str] | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO scheduled_posts
               (article_url, article_title, post_text, image_paths, video_paths, scheduled_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                article_url,
                article_title,
                post_text,
                json.dumps(image_paths),
                json.dumps(video_paths or []),
                scheduled_at.isoformat(),
            ),
        )
        return cur.lastrowid


def get_scheduled_post(post_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM scheduled_posts WHERE id = ?", (post_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["image_paths"] = json.loads(d["image_paths"])
        d["video_paths"] = json.loads(d.get("video_paths") or "[]")
        return d


def update_post_status(post_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE scheduled_posts SET status = ? WHERE id = ?", (status, post_id)
        )


def update_post_text(post_id: int, new_text: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE scheduled_posts SET post_text = ? WHERE id = ?",
            (new_text, post_id),
        )


def set_generated_video_path(post_id: int, path: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE scheduled_posts SET generated_video_path = ? WHERE id = ?",
            (path, post_id),
        )


def get_generated_video_path(post_id: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT generated_video_path FROM scheduled_posts WHERE id = ?",
            (post_id,),
        ).fetchone()
    return row[0] if row else None


def set_notification_message_id(post_id: int, message_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE scheduled_posts SET notification_message_id = ? WHERE id = ?",
            (message_id, post_id),
        )


def get_pending_posts_due(now: datetime) -> list[dict]:
    """Return pending posts whose scheduled_at <= now."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM scheduled_posts
               WHERE status = 'pending' AND scheduled_at <= ?""",
            (now.isoformat(),),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["image_paths"] = json.loads(d["image_paths"])
            result.append(d)
        return result
