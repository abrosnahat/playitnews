"""
One-shot tool: re-download images and videos for all "active" posts.

"Active" matches the dashboard filter (webapp.py /api/posts?status=active):
  status = 'pending'
  OR (status = 'sent' AND fewer than 4 published platforms).

For each active post:
  1. Delete cached media files referenced by image_paths / video_paths
     (including the generated short video, so video_generator rebuilds it).
  2. Re-scrape the article to get fresh image URLs and pg-embeds.
  3. Re-download images and videos via scraper.download_images / download_videos.
  4. Update scheduled_posts: image_paths, video_paths, clear generated_video_path*.

Usage (from project root, .venv activated):
    python redownload_active.py            # rewrite all active posts
    python redownload_active.py --dry-run  # only print what would be done
    python redownload_active.py --id 42 --id 51  # limit to specific post ids
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import aiohttp

import database as db
from config import setup_dirs
from scraper import download_images, download_videos, scrape_article

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)-12s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("redownload")


_ACTIVE_SQL = """
    SELECT * FROM scheduled_posts
    WHERE status = 'pending'
       OR (status = 'sent'
           AND json_array_length(COALESCE(published_platforms, '[]')) < 4)
    ORDER BY id ASC
"""


def _active_posts(ids: list[int] | None) -> list[dict[str, Any]]:
    with db.get_conn() as conn:
        rows = conn.execute(_ACTIVE_SQL).fetchall()
    posts = [dict(r) for r in rows]
    if ids:
        wanted = set(ids)
        posts = [p for p in posts if p["id"] in wanted]
    # Decode JSON columns
    for p in posts:
        p["image_paths"] = json.loads(p.get("image_paths") or "[]")
        p["video_paths"] = json.loads(p.get("video_paths") or "[]")
    return posts


def _delete(paths: list[str], dry: bool) -> int:
    n = 0
    for p in paths:
        if p and os.path.exists(p):
            if dry:
                logger.info("  [dry] would delete %s", p)
            else:
                try:
                    os.remove(p)
                    n += 1
                except OSError as exc:
                    logger.warning("  failed to delete %s: %s", p, exc)
    return n


def _update_post_media(
    post_id: int,
    image_paths: list[str],
    video_paths: list[str],
) -> None:
    with db.get_conn() as conn:
        conn.execute(
            """UPDATE scheduled_posts
                  SET image_paths           = ?,
                      video_paths           = ?,
                      generated_video_path  = NULL,
                      generated_video_path_ru = NULL
                WHERE id = ?""",
            (json.dumps(image_paths), json.dumps(video_paths), post_id),
        )


async def _redownload_one(
    session: aiohttp.ClientSession,
    post: dict[str, Any],
    dry: bool,
) -> tuple[int, int]:
    pid = post["id"]
    url = post["article_url"]
    title = (post.get("article_title") or "")[:80]
    logger.info("Post #%s — %s", pid, title)
    logger.info("  url: %s", url)

    # 1) drop cached files so download_videos won't reuse them
    deleted_imgs = _delete(post["image_paths"], dry)
    deleted_vids = _delete(post["video_paths"], dry)
    for col_attr in ("generated_video_path", "generated_video_path_ru"):
        gv = post.get(col_attr)
        if gv:
            _delete([gv], dry)
    if deleted_imgs or deleted_vids:
        logger.info("  cleaned: %d image(s), %d video(s)", deleted_imgs, deleted_vids)

    # 2) re-scrape article
    article = await scrape_article(session, url)
    if article is None:
        logger.warning("  scrape failed, skip")
        return (0, 0)
    logger.info("  scraped: %d image url(s), %d pg-embed(s)",
                len(article.image_urls), len(article.pg_embeds))

    if dry:
        return (len(article.image_urls), len(article.pg_embeds))

    # 3) download
    image_paths = await download_images(session, article.image_urls)
    _, video_paths = await download_videos(session, article.pg_embeds)
    logger.info("  downloaded: %d image(s), %d video(s)",
                len(image_paths), len(video_paths))

    # 4) persist
    _update_post_media(pid, image_paths, video_paths)
    return (len(image_paths), len(video_paths))


async def main_async(ids: list[int] | None, dry: bool) -> None:
    setup_dirs()
    db.init_db()

    posts = _active_posts(ids)
    if not posts:
        logger.info("No active posts found.")
        return
    logger.info("Found %d active post(s) to refresh%s",
                len(posts), " (dry-run)" if dry else "")

    total_imgs = total_vids = 0
    async with aiohttp.ClientSession() as session:
        for post in posts:
            try:
                imgs, vids = await _redownload_one(session, post, dry)
                total_imgs += imgs
                total_vids += vids
            except Exception:
                logger.exception("  unexpected error on post #%s", post["id"])

    logger.info("Done. Total images: %d, videos: %d", total_imgs, total_vids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, action="append", default=None,
                        help="Limit to specific post id(s). Repeatable.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be done.")
    args = parser.parse_args()

    asyncio.run(main_async(args.id, args.dry_run))


if __name__ == "__main__":
    main()
