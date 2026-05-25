"""
Build an Instagram carousel slide deck for a scheduled post.

Slide layout (per language):
  1. COVER          — ALL-CAPS hook over the first article image (1080×1350)
  2..N-1. BULLET    — fact bullets from ai_adapter.generate_carousel_bullets
                      rendered on remaining article images. If we run out
                      of images, frames are extracted from any local
                      `video_paths` for that post. As a last resort, the
                      bullet is rendered on a brand-coloured gradient.
  N. CTA            — "More news in the Telegram channel, link in bio"
                      (or RU counterpart) on a gradient background.

Output is written to ``images/carousels/post_<id>/<lang>/slide_<n>.jpg``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from typing import Iterable

import ai_adapter
import thumbnail_generator as tg

logger = logging.getLogger(__name__)

CAROUSELS_ROOT = os.path.join("images", "carousels")
MAX_SLIDES = 10               # Instagram hard limit
MIN_TARGET_SLIDES = 5         # aim for at least this many slides
# Instagram requires carousel videos to be ≥ 3 s and ≤ 60 s.
VIDEO_SLIDE_DURATION = 5.0    # seconds per video slide
VIDEO_SLIDE_W = 1080
VIDEO_SLIDE_H = 1350

_CTA_TEXT = {
    "en": "More gaming news\nin our Telegram channel\n— link in bio",
    "ru": "Больше игровых новостей\nв нашем Telegram-канале\n— ссылка в био",
}


def _post_workdir(post_id: int, lang: str) -> str:
    d = os.path.join(CAROUSELS_ROOT, f"post_{post_id}", lang)
    os.makedirs(d, exist_ok=True)
    return d


def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _ffprobe_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.0


def _extract_video_frames(video_paths: list[str], needed: int, out_dir: str) -> list[str]:
    """
    Extract up to *needed* still frames from the supplied videos, evenly
    spaced across each clip's duration. Returns paths to JPEG frames.
    """
    if needed <= 0 or not shutil.which("ffmpeg"):
        return []
    frames: list[str] = []
    for vp in video_paths:
        if len(frames) >= needed:
            break
        if not os.path.exists(vp):
            continue
        dur = _ffprobe_duration(vp)
        if dur <= 0:
            continue
        # Take up to 3 frames per clip, depending on how many are still needed
        per_clip = min(3, needed - len(frames))
        # Sample at 20%, 50%, 80% of clip duration (or fewer for short clips)
        ratios = [0.2, 0.5, 0.8][:per_clip] if per_clip > 1 else [0.5]
        for i, r in enumerate(ratios):
            ts = max(0.1, dur * r)
            out = os.path.join(
                out_dir, f"_frame_{os.path.basename(vp)}_{i}.jpg",
            )
            try:
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ts:.2f}",
                     "-i", vp, "-frames:v", "1", "-q:v", "3", out],
                    capture_output=True, timeout=30,
                )
                if proc.returncode == 0 and os.path.getsize(out) > 1024:
                    frames.append(out)
                    if len(frames) >= needed:
                        break
            except Exception as exc:
                logger.warning("frame extract failed for %s @%.1fs: %s", vp, ts, exc)
    logger.info("Extracted %d frames from %d video(s)", len(frames), len(video_paths))
    return frames


def _plan_video_segments(
    video_paths: list[str],
    needed: int,
    clip_duration: float = VIDEO_SLIDE_DURATION,
) -> list[tuple[str, float]]:
    """
    Plan up to *needed* (source_path, start_seconds) segments evenly spread
    across the supplied videos. Each segment is *clip_duration* seconds long
    and shifted away from the very beginning/end of the source.
    """
    if needed <= 0:
        return []
    segments: list[tuple[str, float]] = []
    # First pass: compute how many segments each clip can give without overlap.
    capacities: list[tuple[str, float, int]] = []
    for vp in video_paths:
        if not os.path.exists(vp):
            continue
        dur = _ffprobe_duration(vp)
        usable = dur - 1.0  # leave 0.5s padding on either end
        if usable < clip_duration:
            continue
        cap = max(1, int(usable // clip_duration))
        # Cap per-clip contribution so segments are spread across sources.
        capacities.append((vp, dur, min(cap, 3)))
    if not capacities:
        return []

    # Round-robin until we have enough segments or all clips are exhausted.
    per_clip_used: dict[str, int] = {vp: 0 for vp, _, _ in capacities}
    while len(segments) < needed:
        progressed = False
        for vp, dur, cap in capacities:
            if len(segments) >= needed:
                break
            used = per_clip_used[vp]
            if used >= cap:
                continue
            # Evenly distribute *cap* starts within usable range.
            usable_start = 0.5
            usable_end   = max(usable_start, dur - clip_duration - 0.5)
            if cap == 1:
                start = (usable_start + usable_end) / 2
            else:
                step = (usable_end - usable_start) / max(1, cap - 1)
                start = usable_start + step * used
            segments.append((vp, max(0.0, start)))
            per_clip_used[vp] = used + 1
            progressed = True
        if not progressed:
            break
    return segments


def _build_video_slide(
    source_path: str,
    start_sec: float,
    text: str,
    out_path: str,
    overlay_png: str,
    duration: float = VIDEO_SLIDE_DURATION,
) -> bool:
    """
    Cut a *duration*-second clip from *source_path* starting at *start_sec*,
    scale+crop it to 1080×1350, overlay the pre-rendered transparent
    *overlay_png* on top, and re-encode to H.264 / yuv420p / AAC.
    Returns True on success.
    """
    if shutil.which("ffmpeg") is None:
        return False
    vf_video = (
        f"scale={VIDEO_SLIDE_W}:{VIDEO_SLIDE_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_SLIDE_W}:{VIDEO_SLIDE_H},setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.2f}", "-i", source_path,
        "-i", overlay_png,
        "-t", f"{duration:.2f}",
        "-filter_complex",
        f"[0:v]{vf_video}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "44100",
        "-movflags", "+faststart",
        "-shortest",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 10_000:
            logger.warning("Video slide encode failed for %s @ %.1fs: %s",
                           source_path, start_sec,
                           proc.stderr.decode(errors="replace")[:300] if proc.stderr else "")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Video slide encode timed out for %s @ %.1fs", source_path, start_sec)
        return False
    except Exception as exc:
        logger.warning("Video slide encode exception: %s", exc)
        return False


def _strip_html(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or "")).strip()


async def build_slides(post: dict, lang: str, *, progress=lambda *_: None) -> list[str]:
    """
    Build a carousel slide deck for *post* in *lang* ("en" or "ru").

    Returns a list of slide file paths in display order. Always writes a
    fresh deck (clears the previous lang directory first).
    """
    post_id = post["id"]
    work = _post_workdir(post_id, lang)
    _clear_dir(work)

    article_title = post.get("article_title") or ""
    if lang == "en":
        post_text = post.get("post_text") or ""
        # Prefer an English title for the hook; reject Cyrillic outright.
        if re.search(r'[а-яёА-ЯЁ]', article_title):
            try:
                en_title = await ai_adapter.translate_title_to_english(article_title)
                if en_title:
                    article_title = en_title
            except Exception as exc:
                logger.warning("translate_title_to_english failed: %s", exc)
    else:
        post_text = post.get("ru_post_text") or post.get("post_text") or ""

    article_images = [p for p in post.get("image_paths", []) if os.path.exists(p)]
    article_videos = [p for p in post.get("video_paths", []) if os.path.exists(p)]
    logger.info(
        "build_slides post #%d lang=%s — images=%d videos=%d",
        post_id, lang, len(article_images), len(article_videos),
    )

    # Bullets — aim for enough to reach MIN_TARGET_SLIDES (cover + bullets + CTA)
    desired_bullets = min(MAX_SLIDES - 2, max(3, MIN_TARGET_SLIDES - 2))
    progress(f"Generating {desired_bullets} bullet points ({lang})…")
    bullets = await ai_adapter.generate_carousel_bullets(
        article_title or "", _strip_html(post_text),
        lang=lang, max_bullets=desired_bullets,
    )
    if not bullets:
        # Fallback: split the post body into sentences
        text = _strip_html(post_text)
        parts = re.split(r'(?<=[.!?])\s+', text)
        bullets = [p.strip() for p in parts if len(p.strip()) > 20][:desired_bullets]
        logger.info("Fell back to sentence split: %d bullets", len(bullets))

    # Determine how many backgrounds we need for the bullet slides
    n_bullet_slides = min(len(bullets), MAX_SLIDES - 2)
    bullets = bullets[:n_bullet_slides]

    # Background slots for bullets. Each slot is one of:
    #   ("img", path)                    — JPEG slide with this image as bg
    #   ("video", source_path, start_s)  — MP4 slide cut from this source
    #   ("none", None)                   — text-on-gradient fallback
    # Strategy:
    #   1. Use article images first (one per slot).
    #   2. If we still need more, plan video segments from article_videos.
    #   3. If we still need more, cycle through whatever images we have.
    #   4. Otherwise fall back to gradient slides.
    available_imgs_for_bullets = max(0, len(article_images) - 1)  # cover reserves [0]
    img_slots: list[str] = list(article_images[1:1 + n_bullet_slides])
    bullet_slots: list[tuple] = [("img", p) for p in img_slots]

    short_by = n_bullet_slides - len(bullet_slots)
    if short_by > 0 and article_videos:
        progress(f"Not enough images ({len(article_images)}); planning {short_by} video clip(s)…")
        segments = _plan_video_segments(article_videos, short_by)
        for vp, start_s in segments:
            bullet_slots.append(("video", vp, start_s))
        logger.info("Planned %d video segments for post #%d", len(segments), post_id)

    if len(bullet_slots) < n_bullet_slides and article_images:
        reusable = article_images
        progress(
            f"Reusing {len(reusable)} image(s) to fill remaining "
            f"{n_bullet_slides - len(bullet_slots)} slot(s)…"
        )
        i = 0
        while len(bullet_slots) < n_bullet_slides:
            bullet_slots.append(("img", reusable[i % len(reusable)]))
            i += 1
    while len(bullet_slots) < n_bullet_slides:
        bullet_slots.append(("none", None))

    slides: list[str] = []

    # --- Slide 1: COVER ---
    progress("Rendering cover slide…")
    cover_src = article_images[0] if article_images else None
    try:
        hook = await ai_adapter.generate_thumbnail_hook(article_title or "Gaming News", lang=lang)
    except Exception:
        hook = ""
    if not hook:
        hook = (article_title or "Gaming News")
    cover_out = os.path.join(work, "slide_01_cover.jpg")
    if cover_src:
        ok = tg.render_carousel_cover(cover_src, hook, cover_out)
    else:
        ok = tg.render_carousel_text_slide(hook, cover_out)
    if ok:
        slides.append(cover_out)

    # --- Middle slides: BULLETS ---
    for i, (bullet, slot) in enumerate(zip(bullets, bullet_slots), start=2):
        kind = slot[0]
        progress(f"Rendering slide {i}/{n_bullet_slides + 2} ({kind})…")
        ok = False
        if kind == "video":
            _, src_path, start_s = slot
            # Render a transparent text overlay first, then ffmpeg-composite.
            overlay_png = os.path.join(work, f"_overlay_{i:02d}.png")
            ovl_ok = tg.render_carousel_video_overlay(bullet, overlay_png)
            if ovl_ok:
                out = os.path.join(work, f"slide_{i:02d}.mp4")
                ok = await asyncio.to_thread(
                    _build_video_slide, src_path, start_s, bullet, out, overlay_png,
                )
                if ok:
                    slides.append(out)
                # Overlay is no longer needed once the video is encoded
                try: os.remove(overlay_png)
                except OSError: pass
            if not ok:
                # Fall back to a still frame from the same source if encoding failed
                logger.warning("Video slide failed at slot %d — falling back to still frame", i)
                frame_out = os.path.join(work, f"slide_{i:02d}.jpg")
                fallback_frame = _extract_video_frames([src_path], 1, work)
                if fallback_frame and tg.render_carousel_image_slide(fallback_frame[0], bullet, frame_out):
                    slides.append(frame_out)
                    ok = True
        elif kind == "img":
            out = os.path.join(work, f"slide_{i:02d}.jpg")
            ok = tg.render_carousel_image_slide(slot[1], bullet, out)
            if ok:
                slides.append(out)
        else:
            out = os.path.join(work, f"slide_{i:02d}.jpg")
            ok = tg.render_carousel_text_slide(bullet, out)
            if ok:
                slides.append(out)

    # --- Final slide: CTA ---
    progress("Rendering CTA slide…")
    cta_out = os.path.join(work, f"slide_{len(slides) + 1:02d}_cta.jpg")
    cta_text = _CTA_TEXT.get(lang, _CTA_TEXT["en"])
    # Reuse an article image as the CTA background. Prefer one we have NOT
    # already used as the cover, falling back to the cover image.
    cta_bg: str | None = None
    if len(article_images) > 1:
        cta_bg = article_images[-1]
    elif article_images:
        cta_bg = article_images[0]
    if tg.render_carousel_text_slide(cta_text, cta_out, bg_image=cta_bg):
        slides.append(cta_out)

    logger.info("Built %d slides for post #%d lang=%s", len(slides), post_id, lang)
    return slides


# ---------------------------------------------------------------------------
# CLI: python carousel_builder.py <post_id> [en|ru]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python carousel_builder.py <post_id> [en|ru]")
        sys.exit(1)
    pid = int(sys.argv[1])
    lng = sys.argv[2] if len(sys.argv) > 2 else "en"
    import database as db
    p = db.get_scheduled_post(pid)
    if not p:
        print(f"Post #{pid} not found")
        sys.exit(1)
    paths = asyncio.run(build_slides(p, lng, progress=lambda *a: print("→", *a)))
    print(f"Built {len(paths)} slides:")
    for s in paths:
        print(" -", s)
