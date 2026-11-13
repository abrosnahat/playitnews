"""
YouTube Shorts publisher via YouTube Data API v3.

Flow:
  1. Load OAuth2 credentials from youtube_token.json (created by get_youtube_token.py).
  2. Upload MP4 as a YouTube Short (vertical video ≤60s is auto-classified as Short).
  3. Set title, description, tags, category=Gaming, privacy=public.
  4. Optionally set a custom thumbnail via thumbnails.set (works for Shorts too;
     requires custom-thumbnail permission on the channel — verified account).

Requirements:
  pip install google-auth-oauthlib google-api-python-client

Environment / files (see config.py):
  client_secrets.json  — OAuth2 client credentials from Google Cloud Console
  youtube_token.json   — OAuth2 token (created by get_youtube_token.py, auto-refreshes)
  YOUTUBE_CATEGORY_ID  — YouTube category (default: 20 = Gaming)
"""
import asyncio
import logging
import os

from config import YOUTUBE_CATEGORY_ID, YOUTUBE_CLIENT_SECRETS, YOUTUBE_TOKEN_FILE, YOUTUBE_TOKEN_FILE_RU

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Return True if EN token file exists."""
    return os.path.exists(YOUTUBE_TOKEN_FILE)


def is_configured_ru() -> bool:
    """Return True if RU token file exists."""
    return os.path.exists(YOUTUBE_TOKEN_FILE_RU)


def _build_youtube_client(token_file: str = YOUTUBE_TOKEN_FILE):
    """Build an authenticated YouTube API client. Blocking."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "pip install google-auth-oauthlib google-api-python-client"
        ) from exc

    if not os.path.exists(token_file):
        raise RuntimeError(
            f"YouTube token not found: {token_file}\n"
            "Run:  python get_youtube_token.py"
        )

    creds = Credentials.from_authorized_user_file(
        token_file,
        scopes=[
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
        ],
    )
    # Refresh if expired
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            if "invalid_grant" in str(exc):
                raise RuntimeError(
                    "YouTube refresh token has been revoked or expired (invalid_grant). "
                    "Re-run:  python get_youtube_token.py"
                ) from exc
            raise
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        logger.info("YouTube token refreshed (%s)", token_file)

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _wait_until_processed(youtube, video_id: str, timeout: int = 180, interval: int = 10) -> bool:
    """
    Poll videos.list until processingDetails.processingStatus == 'succeeded'.

    Per the thumbnails.set docs, for Shorts the method must be called only after
    processing finished, otherwise the thumbnail may not be applied. Blocking.
    """
    import time

    deadline = time.monotonic() + timeout
    while True:
        items = youtube.videos().list(part="processingDetails", id=video_id).execute().get("items", [])
        status = (items[0].get("processingDetails", {}) if items else {}).get("processingStatus")
        if status == "succeeded":
            return True
        if status in ("failed", "terminated") or time.monotonic() >= deadline:
            logger.warning("YouTube %s processing status: %s", video_id, status)
            return False
        time.sleep(interval)


def _set_thumbnail_blocking(youtube, video_id: str, thumbnail_path: str) -> None:
    """
    Set a custom thumbnail on *video_id* using thumbnails.set.

    NOTE: for Shorts the API returns HTTP 200 but Google confirms the custom
    cover is ignored/overwritten ("Won't Fix (Intended Behavior)", issue tracker
    561284868 / 494845784) — Studio uses a private endpoint. The image may still
    show on some surfaces (search/API thumbnails). Requires youtube.force-ssl and
    a verified channel (else 403). Blocking — run in thread.
    """
    from googleapiclient.http import MediaFileUpload

    _wait_until_processed(youtube, video_id)
    mimetype = "image/png" if thumbnail_path.lower().endswith(".png") else "image/jpeg"
    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(thumbnail_path, mimetype=mimetype),
    ).execute()
    logger.info("YouTube thumbnail set for %s", video_id)


def _upload_blocking(
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    token_file: str = YOUTUBE_TOKEN_FILE,
    thumbnail_path: str | None = None,
) -> str:
    """Upload video to YouTube. Returns video ID. Blocking — run in thread."""
    from googleapiclient.http import MediaFileUpload

    youtube = _build_youtube_client(token_file)

    body = {
        "snippet": {
            "title": title[:100],          # YouTube title limit
            "description": description[:5000],
            "tags": tags[:500],
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=4 * 1024 * 1024,   # 4 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    logger.info("Uploading to YouTube: %s", title[:60])
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            logger.info("YouTube upload: %d%%", pct)

    video_id = response["id"]
    logger.info("YouTube upload complete: https://youtu.be/%s", video_id)

    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            _set_thumbnail_blocking(youtube, video_id, thumbnail_path)
        except Exception as exc:
            # Don't fail the whole upload if only the thumbnail step fails
            # (e.g. channel without custom-thumbnail permission → 403).
            logger.warning("YouTube thumbnail set failed for %s: %s", video_id, exc)

    return video_id


async def upload_short(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    token_file: str = YOUTUBE_TOKEN_FILE,
    thumbnail_path: str | None = None,
) -> str:
    """
    Upload *video_path* as a YouTube Short.
    Returns the YouTube video ID (e.g. 'dQw4w9WgXcQ').
    If *thumbnail_path* is given, sets it as the Short's custom thumbnail
    after upload (best-effort — a thumbnail failure won't fail the upload).
    Raises RuntimeError if credentials are missing or upload fails.
    """
    if not os.path.exists(token_file):
        raise RuntimeError(
            f"YouTube token not found: {token_file}. Run:  python get_youtube_token.py"
        )
    return await asyncio.to_thread(
        _upload_blocking, video_path, title, description, tags or [], token_file, thumbnail_path
    )


async def upload_short_ru(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    thumbnail_path: str | None = None,
) -> str:
    """Convenience wrapper that uploads to the RU YouTube channel."""
    return await upload_short(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        token_file=YOUTUBE_TOKEN_FILE_RU,
        thumbnail_path=thumbnail_path,
    )


async def set_thumbnail(
    video_id: str,
    thumbnail_path: str,
    token_file: str = YOUTUBE_TOKEN_FILE,
) -> None:
    """
    Set a custom thumbnail on an already-published video/Short.
    Raises on failure (unlike the best-effort path inside upload_short).
    """
    if not os.path.exists(thumbnail_path):
        raise RuntimeError(f"Thumbnail not found: {thumbnail_path}")

    def _run() -> None:
        youtube = _build_youtube_client(token_file)
        _set_thumbnail_blocking(youtube, video_id, thumbnail_path)

    await asyncio.to_thread(_run)
