"""
Instagram Reels publisher via Instagram API (Meta, 2024+).

Flow:
  1. Upload MP4 to a free temporary public host (tries multiple services).
  2. Create Instagram media container (REELS) with that URL.
  3. Poll until container status == FINISHED.
  4. Publish the container.
  5. File on free host expires automatically.

Requirements:
  - pip install aiohttp
  - Instagram Business or Creator account.
  - Meta app with "Instagram" product and instagram_business_content_publish
    permission (Instagram Login OAuth — no Facebook Page required).

How to get INSTAGRAM_ACCESS_TOKEN: run  python get_instagram_token.py

Environment variables (see config.py):
  INSTAGRAM_USER_ID      — your Instagram account numeric ID
  INSTAGRAM_ACCESS_TOKEN — long-lived token (60 days)
"""
import asyncio
import json
import logging
import os
import uuid
import urllib.request

import aiohttp

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
# Upload backends (tried in order until one succeeds)
# ---------------------------------------------------------------------------

def _ssl_ctx():
    """Return an SSL context that validates certificates via certifi."""
    import ssl, certifi
    return ssl.create_default_context(cafile=certifi.where())


def _upload_catbox(local_path: str) -> str:
    """catbox.moe — free, no account, files kept indefinitely."""
    boundary = uuid.uuid4().hex
    filename  = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reqtype"\r\n\r\nfileupload\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="fileToUpload"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://catbox.moe/user/api.php",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
        url = resp.read().decode().strip()
    if not url.startswith("http"):
        raise RuntimeError(f"catbox.moe unexpected response: {url!r}")
    return url


def _upload_0x0(local_path: str) -> str:
    """0x0.st — free, no account, files expire after ~1 year."""
    boundary = uuid.uuid4().hex
    filename  = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://0x0.st",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
        return resp.read().decode().strip()


def _upload_transfer_sh(local_path: str) -> str:
    """transfer.sh — free, no account, files auto-deleted after 1 day."""
    filename = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        req = urllib.request.Request(
            f"https://transfer.sh/{filename}", data=f, method="PUT",
            headers={"Content-Type": "video/mp4", "Max-Days": "1"},
        )
        with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
            return resp.read().decode().strip()


def _upload_pixeldrain(local_path: str) -> str:
    """pixeldrain.com — free, no account, direct MP4 link."""
    filename = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        req = urllib.request.Request(
            f"https://pixeldrain.com/api/file/{filename}",
            data=f, method="PUT",
            headers={"Content-Type": "video/mp4"},
        )
        with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read())
    file_id = data.get("id")
    if not file_id:
        raise RuntimeError(f"pixeldrain error: {data}")
    return f"https://pixeldrain.com/api/file/{file_id}"


def _upload_litterbox(local_path: str) -> str:
    """litterbox.catbox.moe — free, no account, files expire after 72 h. Direct MP4 link."""
    boundary = uuid.uuid4().hex
    filename  = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reqtype"\r\n\r\nfileupload\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="time"\r\n\r\n72h\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="fileToUpload"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
        url = resp.read().decode().strip()
    if not url.startswith("http"):
        raise RuntimeError(f"litterbox unexpected response: {url!r}")
    return url


def _upload_uguu(local_path: str) -> str:
    """uguu.se — free, no account, files expire after ~48 h. Direct MP4 link."""
    boundary = uuid.uuid4().hex
    filename  = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files[]"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://uguu.se/upload",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
        data = json.loads(resp.read())
    files = data.get("files", [])
    if not files or not files[0].get("url"):
        raise RuntimeError(f"uguu.se unexpected response: {data}")
    return files[0]["url"]


async def _upload_video(local_path: str) -> str:
    """
    Try free upload backends in order until one succeeds.
    Returns a publicly accessible URL Instagram can fetch.
    """
    backends = [
        ("litterbox.catbox.moe", _upload_litterbox),
        ("uguu.se",             _upload_uguu),
        ("catbox.moe",          _upload_catbox),
        ("pixeldrain.com",      _upload_pixeldrain),
        ("0x0.st",              _upload_0x0),
        ("transfer.sh",         _upload_transfer_sh),
    ]
    errors: list[str] = []
    for name, fn in backends:
        try:
            logger.info("Uploading to %s...", name)
            url = await asyncio.to_thread(fn, local_path)
            logger.info("%s → %s", name, url)
            return url
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")

    raise RuntimeError(
        "All upload backends failed:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Instagram API helpers
# ---------------------------------------------------------------------------

def _aiohttp_session() -> aiohttp.ClientSession:
    """aiohttp session with certifi SSL context (required on macOS)."""
    import ssl, certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ctx)
    return aiohttp.ClientSession(connector=connector)


async def _ig_create_container(
    session: aiohttp.ClientSession,
    video_url: str,
    caption: str,
    user_id: str,
    access_token: str,
) -> str:
    """Step 1: Create a REELS media container. Returns container ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/media"
    payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": access_token,
    }
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
) -> str:
    """
    Upload *video_path*, post as Instagram Reel, then clean up.
    Pass *user_id* / *access_token* to publish to a non-default account;
    otherwise defaults to INSTAGRAM_USER_ID / INSTAGRAM_ACCESS_TOKEN.
    Returns the published media ID on success.
    Raises RuntimeError / TimeoutError on failure.
    """
    uid = user_id or INSTAGRAM_USER_ID
    tok = access_token or INSTAGRAM_ACCESS_TOKEN
    if not (uid and tok):
        raise RuntimeError(
            "Instagram credentials not configured. "
            "Set INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN in .env"
        )

    # 1. Upload video to temporary public host (once — reused across retries)
    video_url = await _upload_video(video_path)

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, 4):
        async with _aiohttp_session() as session:
            try:
                # 2. Create container
                logger.info("Creating Instagram container (attempt %d/3)...", attempt)
                container_id = await _ig_create_container(session, video_url, caption, uid, tok)
                logger.info("Container ID: %s — waiting for processing...", container_id)

                # 3. Wait for processing
                await _ig_wait_for_container(session, container_id, tok)

                # 4. Publish
                logger.info("Publishing container %s...", container_id)
                media_id = await _ig_publish_container(session, container_id, uid, tok)
                logger.info("Published Instagram Reel, media_id=%s", media_id)
                return media_id

            except RuntimeError as exc:
                last_exc = exc
                msg = str(exc).lower()
                # Auth errors are never retriable
                if "oauthexception" in msg or "access token" in msg or "error_subcode" in msg:
                    raise
                # Retry on transient Meta errors that suggest recreating the container
                if "something went wrong" in msg or "please retry" in msg or "container" in msg:
                    logger.warning("Instagram transient error (attempt %d/3): %s", attempt, exc)
                    if attempt < 3:
                        await asyncio.sleep(15 * attempt)  # 15s, 30s
                    continue
                raise  # non-retriable error (bad video, etc.)

    raise last_exc
