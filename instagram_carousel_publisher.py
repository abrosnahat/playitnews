"""
Instagram CAROUSEL publisher via Instagram Graph API.

Flow (same building blocks as `instagram_publisher.publish_reel`, but for
image carousels — up to 10 photos):

  1. Optimise each slide → JPEG, 1080×1350 (4:5), sRGB, ≤ 8 MB.
  2. Upload all slides to GitHub via github_uploader.
  3. Rewrite each jsDelivr URL to raw.githubusercontent.com (origin, no
     CDN propagation lag, and slides are tiny so the 10 MB octet-stream
     limit is irrelevant).
  4. Create a child media container per slide
     (`media_type=IMAGE`, `is_carousel_item=true`) — done concurrently.
  5. Create the parent CAROUSEL container with `children=<comma joined ids>`.
  6. Poll parent until status_code == FINISHED.
  7. Publish the parent. Return the published media ID.
  8. Delete the GitHub assets in a `finally` block (when
     GITHUB_MEDIA_DELETE_AFTER_PUBLISH=1).

Endpoint, version, and token type are identical to the Reels publisher,
so no extra Meta-side configuration is required.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Sequence

import aiohttp

import github_uploader
from config import (
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_USER_ID,
    INSTAGRAM_ACCESS_TOKEN_RU,
    INSTAGRAM_USER_ID_RU,
)
from instagram_publisher import (
    GRAPH_API_BASE,
    _CONTAINER_POLL_INTERVAL,
    _CONTAINER_POLL_TIMEOUT,
    _jsdelivr_to_raw,
    _make_session,
)

logger = logging.getLogger(__name__)

CAR_W, CAR_H = 1080, 1350     # 4:5 portrait (Instagram-recommended)
MAX_SLIDES = 10               # Instagram hard limit
MIN_SLIDES = 2                # Instagram requires at least 2 children


# ---------------------------------------------------------------------------
# Per-slide optimisation
# ---------------------------------------------------------------------------

def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".mov", ".m4v"))


def _optimize_slide_for_ig(src_path: str) -> str:
    """
    Normalise *src_path* for IG. Images → 1080×1350 baseline JPEG. Videos
    are passed through unchanged (carousel_builder already encodes them as
    1080×1350 H.264/AAC/yuv420p +faststart). If ffmpeg is missing, the
    original path is returned.
    """
    if _is_video(src_path):
        return src_path
    if shutil.which("ffmpeg") is None:
        return src_path
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="ig_slide_")
    os.close(fd)
    # Scale-to-fill then centre-crop to exactly 1080×1350.
    vf = (
        f"scale={CAR_W}:{CAR_H}:force_original_aspect_ratio=increase,"
        f"crop={CAR_W}:{CAR_H},format=yuvj420p"
    )
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
             "-vf", vf, "-q:v", "3", tmp_path],
            check=False, capture_output=True,
        )
        if proc.returncode != 0 or not os.path.getsize(tmp_path):
            logger.warning("IG slide re-encode failed: %s",
                           proc.stderr.decode(errors="replace")[:300])
            try: os.remove(tmp_path)
            except OSError: pass
            return src_path
        return tmp_path
    except Exception as exc:
        logger.warning("IG slide re-encode exception: %s", exc)
        try: os.remove(tmp_path)
        except OSError: pass
        return src_path


# ---------------------------------------------------------------------------
# Graph API calls
# ---------------------------------------------------------------------------

async def _create_child_container(
    session: aiohttp.ClientSession,
    media_url: str,
    user_id: str,
    access_token: str,
    *,
    is_video: bool = False,
) -> str:
    """Create a single carousel-child container.

    For images: media_type=IMAGE is implicit when image_url is set.
    For videos: must pass media_type=VIDEO + video_url + is_carousel_item.
    Returns the child container id.
    """
    url = f"{GRAPH_API_BASE}/{user_id}/media"
    if is_video:
        payload = {
            "media_type": "VIDEO",
            "video_url": media_url,
            "is_carousel_item": "true",
            "access_token": access_token,
        }
    else:
        payload = {
            "image_url": media_url,
            "is_carousel_item": "true",
            "access_token": access_token,
        }
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"IG carousel child error: {data['error']}")
        return data["id"]


async def _create_parent_container(
    session: aiohttp.ClientSession,
    children_ids: Sequence[str],
    caption: str,
    user_id: str,
    access_token: str,
) -> str:
    """Create the CAROUSEL container that references all children."""
    url = f"{GRAPH_API_BASE}/{user_id}/media"
    payload = {
        "media_type": "CAROUSEL",
        "children": ",".join(children_ids),
        "caption": caption,
        "access_token": access_token,
    }
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"IG carousel parent error: {data['error']}")
        return data["id"]


async def _wait_for_container(
    session: aiohttp.ClientSession,
    container_id: str,
    access_token: str,
) -> None:
    """Poll the container until status_code == FINISHED."""
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
                raise RuntimeError(f"IG carousel status error: {data['error']}")
            status = data.get("status_code") or data.get("status", "")
            logger.debug("IG carousel container %s status: %s", container_id, status)
            if status == "FINISHED":
                return
            if status == "ERROR":
                err = data.get("error_message", "no details")
                raise RuntimeError(f"IG carousel container {container_id} ERROR: {err}")
        await asyncio.sleep(_CONTAINER_POLL_INTERVAL)
        elapsed += _CONTAINER_POLL_INTERVAL
    raise TimeoutError(
        f"IG carousel container {container_id} did not finish within "
        f"{_CONTAINER_POLL_TIMEOUT}s"
    )


async def _publish_container(
    session: aiohttp.ClientSession,
    container_id: str,
    user_id: str,
    access_token: str,
) -> str:
    """Publish the container. Returns the new media ID."""
    url = f"{GRAPH_API_BASE}/{user_id}/media_publish"
    payload = {"creation_id": container_id, "access_token": access_token}
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"IG carousel publish error: {data['error']}")
        return data["id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    return bool(INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN)


def is_configured_ru() -> bool:
    return bool(INSTAGRAM_USER_ID_RU and INSTAGRAM_ACCESS_TOKEN_RU)


async def publish_carousel(
    image_paths: list[str],
    caption: str,
    *,
    user_id: str | None = None,
    access_token: str | None = None,
) -> str:
    """
    Publish *image_paths* (in order) as an Instagram carousel post.

    Returns the published media ID.
    """
    uid = user_id or INSTAGRAM_USER_ID
    tok = access_token or INSTAGRAM_ACCESS_TOKEN
    if not (uid and tok):
        raise RuntimeError(
            "Instagram credentials not configured "
            "(INSTAGRAM_USER_ID / INSTAGRAM_ACCESS_TOKEN)."
        )

    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < MIN_SLIDES:
        raise RuntimeError(
            f"Need at least {MIN_SLIDES} slides for a carousel, got {len(paths)}."
        )
    if len(paths) > MAX_SLIDES:
        logger.warning("Trimming carousel from %d to %d slides (IG limit)",
                       len(paths), MAX_SLIDES)
        paths = paths[:MAX_SLIDES]

    # 1. Optimise each slide to 1080×1350 JPEG (videos pass through)
    optimised: list[str] = []
    tmp_files: list[str] = []
    is_video_flags: list[bool] = []
    for p in paths:
        opt = await asyncio.to_thread(_optimize_slide_for_ig, p)
        optimised.append(opt)
        is_video_flags.append(_is_video(opt))
        if opt != p:
            tmp_files.append(opt)

    # 2. Upload each slide to GitHub (raw.github URL — slides are tiny)
    upload_paths: list[str] = []
    media_urls:   list[str] = []
    # Meta refuses application/octet-stream payloads above ~10 MB. raw.github
    # serves with that Content-Type; jsDelivr serves video/mp4 + image/jpeg
    # and lifts the limit. Slides almost always fit, but a 5 s 1080×1350
    # video can occasionally exceed this on very high-motion clips.
    _RAW_OCTET_LIMIT = 9 * 1024 * 1024
    try:
        for i, op in enumerate(optimised, start=1):
            is_vid = is_video_flags[i - 1]
            kind = "video" if is_vid else "image"
            size = os.path.getsize(op)
            logger.info("Uploading carousel slide %d/%d (%s, %.2f MB) to GitHub…",
                        i, len(optimised), kind, size / (1024 * 1024))
            jsd_url, repo_path = await asyncio.to_thread(github_uploader.upload, op)
            upload_paths.append(repo_path)
            # Big videos → keep the jsDelivr URL (sets the right Content-Type).
            if is_vid and size > _RAW_OCTET_LIMIT:
                media_urls.append(jsd_url)
            else:
                media_urls.append(_jsdelivr_to_raw(jsd_url))

        media_id: str | None = None
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(1, 4):
            async with _make_session() as session:
                try:
                    # 3. Create child containers (concurrently)
                    logger.info("Creating %d carousel children (attempt %d/3)…",
                                len(media_urls), attempt)
                    child_ids: list[str] = await asyncio.gather(*[
                        _create_child_container(
                            session, url, uid, tok, is_video=is_vid,
                        )
                        for url, is_vid in zip(media_urls, is_video_flags)
                    ])

                    # 3b. Video children need to finish processing BEFORE we
                    # reference them from the parent CAROUSEL container,
                    # otherwise the parent creation fails with
                    # "Media ID is not available". Image children finish
                    # instantly so no polling is required for them.
                    video_child_ids = [
                        cid for cid, is_vid in zip(child_ids, is_video_flags) if is_vid
                    ]
                    if video_child_ids:
                        logger.info("Waiting for %d video child(ren) to finish…",
                                    len(video_child_ids))
                        await asyncio.gather(*[
                            _wait_for_container(session, cid, tok)
                            for cid in video_child_ids
                        ])

                    # 4. Create parent CAROUSEL container
                    parent_id = await _create_parent_container(
                        session, child_ids, caption, uid, tok,
                    )
                    logger.info("Carousel parent container: %s — waiting…", parent_id)

                    # 5. Poll parent until FINISHED
                    await _wait_for_container(session, parent_id, tok)

                    # 6. Publish
                    media_id = await _publish_container(session, parent_id, uid, tok)
                    logger.info("Published IG carousel, media_id=%s", media_id)
                    return media_id

                except RuntimeError as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "oauthexception" in msg or "access token" in msg:
                        raise
                    transient = (
                        "something went wrong", "please retry",
                        "could not be fetched", "media could not be fetched",
                        "timeout", "connection", "fwdproxy", "fetch headers",
                        "http error code 4", "http error code 5",
                    )
                    if any(m in msg for m in transient):
                        logger.warning("IG carousel transient error (%d/3): %s",
                                       attempt, exc)
                        if attempt < 3:
                            await asyncio.sleep(10 * attempt)
                        continue
                    raise
        raise last_exc

    finally:
        # 7. Clean up GitHub assets (best-effort, only after publish)
        if os.getenv("GITHUB_MEDIA_DELETE_AFTER_PUBLISH", "1") == "1":
            for rp in upload_paths:
                try:
                    await asyncio.to_thread(github_uploader.delete, rp)
                except Exception as exc:
                    logger.warning("GitHub delete failed for %s: %s", rp, exc)
        # Drop optimised temp files
        for tmp in tmp_files:
            try: os.remove(tmp)
            except OSError: pass
