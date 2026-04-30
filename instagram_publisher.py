"""
Instagram Reels publisher via Instagram Graph API.

Flow:
  1. Upload MP4 (and optional cover image) to GitHub via github_uploader.
  2. Create a REELS media container with the resulting raw.githubusercontent.com URLs.
  3. Poll until container status == FINISHED.
  4. Publish the container.
  5. Delete the GitHub assets (if GITHUB_MEDIA_DELETE_AFTER_PUBLISH=1).

Requirements:
  - pip install aiohttp certifi
  - Instagram Business or Creator account.
  - Meta app with Instagram product and instagram_business_content_publish permission.
  - Public GitHub repo + PAT (GITHUB_MEDIA_REPO / GITHUB_MEDIA_TOKEN in .env).

Environment variables (see config.py):
  INSTAGRAM_USER_ID      — Instagram account numeric ID
  INSTAGRAM_ACCESS_TOKEN — long-lived token
"""
import asyncio
import logging
import os
import ssl

import aiohttp
import certifi

import github_uploader
from config import (
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_USER_ID,
    INSTAGRAM_ACCESS_TOKEN_RU,
    INSTAGRAM_USER_ID_RU,
)

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.instagram.com/v21.0"
_CONTAINER_POLL_INTERVAL = 5    # seconds between status checks
_CONTAINER_POLL_TIMEOUT  = 300  # give up after 5 minutes


# ---------------------------------------------------------------------------
# Instagram API helpers
# ---------------------------------------------------------------------------

def _aiohttp_session() -> aiohttp.ClientSession:
    """aiohttp session with certifi SSL context (required on macOS)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ctx)
    return aiohttp.ClientSession(connector=connector)


async def _ig_create_container(
    session: aiohttp.ClientSession,
    video_url: str,
    caption: str,
    user_id: str,
    access_token: str,
    cover_url: str | None = None,
) -> str:
    """Step 1: Create a REELS media container. Returns container ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/media"
    payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": access_token,
    }
    if cover_url:
        payload["cover_url"] = cover_url
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Instagram create container error: {data['error']}")
        return data["id"]


async def _ig_wait_for_container(
    session: aiohttp.ClientSession,
    container_id: str,
    access_token: str,
) -> None:
    """Step 2: Poll until Instagram has finished processing the video."""
    url = f"{GRAPH_API_BASE}/{container_id}"
    params = {
        "fields": "status_code,status,error_message",
        "access_token": access_token,
    }
    elapsed = 0
    while elapsed < _CONTAINER_POLL_TIMEOUT:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"Instagram container status error: {data['error']}")
            status = data.get("status_code") or data.get("status", "")
            logger.debug("Instagram container %s status: %s", container_id, status)
            if status == "FINISHED":
                return
            if status == "ERROR":
                err_msg = data.get("error_message", "no details")
                raise RuntimeError(
                    f"Instagram container {container_id} ERROR: {err_msg} | raw={data}"
                )
        await asyncio.sleep(_CONTAINER_POLL_INTERVAL)
        elapsed += _CONTAINER_POLL_INTERVAL

    raise TimeoutError(
        f"Instagram container {container_id} did not finish within {_CONTAINER_POLL_TIMEOUT}s"
    )


async def _ig_publish_container(
    session: aiohttp.ClientSession,
    container_id: str,
    user_id: str,
    access_token: str,
) -> str:
    """Step 3: Publish the container. Returns the new media ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/media_publish"
    payload = {
        "creation_id": container_id,
        "access_token": access_token,
    }
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Instagram publish error: {data['error']}")
        return data["id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """Return True if English Instagram credentials are set."""
    return bool(INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN)


def is_configured_ru() -> bool:
    """Return True if Russian Instagram credentials are set."""
    return bool(INSTAGRAM_USER_ID_RU and INSTAGRAM_ACCESS_TOKEN_RU)


async def publish_reel(
    video_path: str,
    caption: str,
    *,
    user_id: str | None = None,
    access_token: str | None = None,
    cover_image_path: str | None = None,
) -> str:
    """
    Publish *video_path* as an Instagram Reel.

    Pass *user_id* / *access_token* to publish to a non-default account;
    otherwise defaults to INSTAGRAM_USER_ID / INSTAGRAM_ACCESS_TOKEN.
    Pass *cover_image_path* to set a custom Reels cover. Returns the
    published media ID on success.
    """
    uid = user_id or INSTAGRAM_USER_ID
    tok = access_token or INSTAGRAM_ACCESS_TOKEN
    if not (uid and tok):
        raise RuntimeError(
            "Instagram credentials not configured. "
            "Set INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN in .env"
        )
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")

    # 1. Upload video to GitHub (acts as a Meta-friendly CDN).
    logger.info("Uploading video to GitHub...")
    video_url, video_repo_path = await asyncio.to_thread(github_uploader.upload, video_path)
    logger.info("Public video URL: %s", video_url)

    cover_url: str | None = None
    cover_repo_path: str | None = None
    if cover_image_path and os.path.exists(cover_image_path):
        try:
            cover_url, cover_repo_path = await asyncio.to_thread(
                github_uploader.upload, cover_image_path
            )
            logger.info("Public cover URL: %s", cover_url)
        except Exception as exc:
            logger.warning("Cover upload failed (continuing without cover): %s", exc)

    last_exc: Exception = RuntimeError("No attempts made")
    media_id: str | None = None
    try:
        for attempt in range(1, 4):
            async with _aiohttp_session() as session:
                try:
                    logger.info("Creating Instagram container (attempt %d/3)...", attempt)
                    container_id = await _ig_create_container(
                        session, video_url, caption, uid, tok, cover_url
                    )
                    logger.info("Container ID: %s — waiting for processing...", container_id)

                    await _ig_wait_for_container(session, container_id, tok)

                    logger.info("Publishing container %s...", container_id)
                    media_id = await _ig_publish_container(session, container_id, uid, tok)
                    logger.info("Published Instagram Reel, media_id=%s", media_id)
                    return media_id

                except RuntimeError as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    # Auth errors are never retriable.
                    if "oauthexception" in msg or "access token" in msg or "error_subcode" in msg:
                        raise
                    # Retry on transient Meta errors that suggest recreating the container.
                    if "something went wrong" in msg or "please retry" in msg or "container" in msg:
                        logger.warning("Instagram transient error (attempt %d/3): %s", attempt, exc)
                        if attempt < 3:
                            await asyncio.sleep(15 * attempt)  # 15s, 30s
                        continue
                    raise  # non-retriable error (bad video, etc.)

        raise last_exc
    finally:
        # Best-effort cleanup of GitHub assets after a successful publish.
        if media_id is not None and os.getenv("GITHUB_MEDIA_DELETE_AFTER_PUBLISH", "1") == "1":
            await asyncio.to_thread(github_uploader.delete, video_repo_path)
            if cover_repo_path is not None:
                await asyncio.to_thread(github_uploader.delete, cover_repo_path)
