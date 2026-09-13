"""
Threads posting via Meta's official Threads API (graph.threads.net).

Posts are TEXT or IMAGE+TEXT here (no video upload) — the post text is
expected to already be adapted for Threads by ai_adapter.adapt_post_for_threads().
When an article image is available it's uploaded to GitHub (same public-hosting
trick used by instagram_publisher.py) and posted as an IMAGE container with the
adapted text as its caption; otherwise a plain TEXT post is published.

Flow:
  1. (If an image is given) upload it to GitHub via github_uploader, get a
     public URL, rewritten to raw.githubusercontent.com when small enough.
  2. Create a Threads media container: POST /{threads_user_id}/threads with
     media_type=IMAGE (image_url + text) or media_type=TEXT (text only).
  3. Poll GET /{container_id}?fields=status,error_message until FINISHED.
  4. Publish: POST /{threads_user_id}/threads_publish with creation_id.
  5. Delete the GitHub image asset (if GITHUB_MEDIA_DELETE_AFTER_PUBLISH=1).

Setup required (one-time, cannot be done from code):
  - Create a Meta app with the "Threads" use case
    (https://developers.facebook.com/documentation/development/create-an-app/threads-use-case).
  - Get a Threads user access token with threads_basic + threads_content_publish
    scopes for the account you want to post as — use get_threads_token.py.
  - Set THREADS_USER_ID / THREADS_ACCESS_TOKEN (and _RU for the second
    account) in .env.

Docs: https://developers.facebook.com/docs/threads
"""
import asyncio
import logging
import os
import ssl

import aiohttp
import certifi

import github_uploader
from config import (
    THREADS_ACCESS_TOKEN,
    THREADS_USER_ID,
    THREADS_ACCESS_TOKEN_RU,
    THREADS_USER_ID_RU,
)

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.threads.net/v1.0"
_CONTAINER_POLL_INTERVAL = 3    # seconds between status checks
_CONTAINER_POLL_TIMEOUT = 180   # give up after 3 minutes (images take a bit longer than text)

# Threads text posts are capped at 500 characters (emoji count as UTF-8
# byte-groups, but a plain character cap is a safe conservative margin).
_TEXT_MAX_LEN = 480

# Same rationale as instagram_publisher.py: raw.githubusercontent.com serves
# application/octet-stream, which some Meta ingesters reject above ~10 MB;
# fall back to jsDelivr (image/* content-type) for bigger files. Threads'
# own image spec caps at 8 MB anyway, so this is mostly a safety net.
_RAW_GITHUB_OCTET_LIMIT = 10 * 1024 * 1024  # 10 MB


def _jsdelivr_to_raw(url: str) -> str:
    """Rewrite a jsDelivr URL produced by github_uploader to raw.githubusercontent.com."""
    prefix = "https://cdn.jsdelivr.net/gh/"
    if not url.startswith(prefix):
        return url
    rest = url[len(prefix):]
    if "@" not in rest or "/" not in rest:
        return url
    owner_repo, _, after = rest.partition("@")
    if "/" not in owner_repo or "/" not in after:
        return url
    branch, _, path = after.partition("/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"


def _make_session() -> aiohttp.ClientSession:
    ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ctx)
    return aiohttp.ClientSession(connector=connector)


def _truncate_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _TEXT_MAX_LEN:
        return text
    return text[:_TEXT_MAX_LEN - 1].rstrip() + "…"


async def _create_text_container(
    session: aiohttp.ClientSession,
    text: str,
    user_id: str,
    access_token: str,
) -> str:
    """Create a Threads TEXT media container. Returns container ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/threads"
    payload = {
        "media_type": "TEXT",
        "text": _truncate_text(text),
        "access_token": access_token,
    }
    async with session.post(url, data=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Threads create container error: {data['error']}")
        return data["id"]


async def _create_image_container(
    session: aiohttp.ClientSession,
    image_url: str,
    text: str,
    user_id: str,
    access_token: str,
) -> str:
    """Create a Threads IMAGE media container (with caption text). Returns container ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/threads"
    payload = {
        "media_type": "IMAGE",
        "image_url": image_url,
        "text": _truncate_text(text),
        "access_token": access_token,
    }
    async with session.post(url, data=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Threads create container error: {data['error']}")
        return data["id"]


async def _wait_for_container(
    session: aiohttp.ClientSession,
    container_id: str,
    access_token: str,
) -> None:
    """Step 2: Poll until Threads has finished processing the container."""
    url = f"{GRAPH_API_BASE}/{container_id}"
    params = {"fields": "status,error_message", "access_token": access_token}
    elapsed = 0
    while elapsed < _CONTAINER_POLL_TIMEOUT:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"Threads container status error: {data['error']}")
            status = data.get("status", "")
            logger.debug("Threads container %s status: %s", container_id, status)
            if status == "FINISHED":
                return
            if status in ("ERROR", "EXPIRED"):
                err_msg = data.get("error_message", "no details")
                raise RuntimeError(f"Threads container {container_id} {status}: {err_msg} | raw={data}")
        await asyncio.sleep(_CONTAINER_POLL_INTERVAL)
        elapsed += _CONTAINER_POLL_INTERVAL

    raise TimeoutError(f"Threads container {container_id} did not finish within {_CONTAINER_POLL_TIMEOUT}s")


async def _publish_container(
    session: aiohttp.ClientSession,
    container_id: str,
    user_id: str,
    access_token: str,
) -> str:
    """Step 3: Publish the container. Returns the new Threads media ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/threads_publish"
    payload = {"creation_id": container_id, "access_token": access_token}
    async with session.post(url, data=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Threads publish error: {data['error']}")
        return data["id"]


def is_configured() -> bool:
    """Return True if English Threads credentials are set."""
    return bool(THREADS_USER_ID and THREADS_ACCESS_TOKEN)


def is_configured_ru() -> bool:
    """Return True if Russian Threads credentials are set."""
    return bool(THREADS_USER_ID_RU and THREADS_ACCESS_TOKEN_RU)


async def publish_post(
    text: str,
    *,
    image_path: str | None = None,
    user_id: str | None = None,
    access_token: str | None = None,
) -> str:
    """
    Publish a Threads post.

    If *image_path* points to an existing file, it's uploaded and published
    as an IMAGE post with *text* as the caption. Otherwise a plain TEXT post
    is published. Pass *user_id* / *access_token* to publish to a non-default
    account; otherwise defaults to THREADS_USER_ID / THREADS_ACCESS_TOKEN.
    Returns the published Threads media ID on success.
    """
    uid = user_id or THREADS_USER_ID
    tok = access_token or THREADS_ACCESS_TOKEN
    if not (uid and tok):
        raise RuntimeError(
            "Threads credentials not configured. "
            "Set THREADS_USER_ID and THREADS_ACCESS_TOKEN in .env"
        )
    has_image = bool(image_path and os.path.exists(image_path))
    if not (text or "").strip() and not has_image:
        raise RuntimeError("Threads post text is empty")

    image_repo_path: str | None = None
    try:
        async with _make_session() as session:
            if has_image:
                logger.info("Uploading image to GitHub...")
                jsd_url, image_repo_path = await asyncio.to_thread(github_uploader.upload, image_path)
                img_size = os.path.getsize(image_path)
                image_url = (
                    _jsdelivr_to_raw(jsd_url) if img_size <= _RAW_GITHUB_OCTET_LIMIT else jsd_url
                )
                logger.info("Public image URL: %s", image_url)
                logger.info("Creating Threads image container...")
                container_id = await _create_image_container(session, image_url, text, uid, tok)
            else:
                logger.info("Creating Threads text container...")
                container_id = await _create_text_container(session, text, uid, tok)

            logger.info("Container ID: %s — waiting for processing…", container_id)
            await _wait_for_container(session, container_id, tok)

            logger.info("Publishing container %s...", container_id)
            media_id = await _publish_container(session, container_id, uid, tok)
            logger.info("Published Threads post, media_id=%s", media_id)
            return media_id
    finally:
        if image_repo_path is not None and os.getenv("GITHUB_MEDIA_DELETE_AFTER_PUBLISH", "1") == "1":
            await asyncio.to_thread(github_uploader.delete, image_repo_path)
