"""
TikTok video uploader via browser automation (Playwright).

Uses a saved persistent Chromium profile (tiktok_session/) so TikTok stays
logged in across uploads. No official API required.

First-time setup:
  pip install playwright
  playwright install chromium          (or .venv/bin/playwright install chromium)
  python get_tiktok_session.py        # log in once, saves session

Limitations:
  - TikTok may detect automation and block uploads — keep headless=False if issues arise.
  - Session expires when TikTok forces re-login; re-run get_tiktok_session.py to refresh.
  - Caption limit: 2 200 characters including hashtags.
"""

import asyncio
import logging
import os

from config import TIKTOK_SESSION_DIR

logger = logging.getLogger(__name__)

# Candidate upload page URLs (TikTok changes these occasionally)
_UPLOAD_URLS = [
    "https://www.tiktok.com/tiktokstudio/upload",
    "https://www.tiktok.com/upload",
]


def is_configured() -> bool:
    """Return True if a saved browser session exists."""
    return os.path.isdir(TIKTOK_SESSION_DIR)


async def _dismiss_overlay(page) -> None:
    """Close TikTok's onboarding/tutorial overlay (react-joyride) if visible."""
    try:
        # Remove overlay element from DOM entirely via JS
        removed = await page.evaluate("""
            () => {
                const overlay = document.querySelector(
                    '[data-test-id="overlay"], .react-joyride__overlay'
                );
                if (overlay) {
                    overlay.remove();
                    return true;
                }
                const portal = document.getElementById('react-joyride-portal');
                if (portal) {
                    portal.remove();
                    return true;
                }
                return false;
            }
        """)
        if removed:
            logger.info("TikTok: dismissed onboarding overlay")
            await page.wait_for_timeout(300)
    except Exception as exc:
        logger.debug("TikTok: overlay dismiss: %s", exc)


async def _do_upload(video_path: str, caption: str) -> None:
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise RuntimeError(
            "Playwright not installed.\n"
            "  .venv/bin/pip install playwright\n"
            "  .venv/bin/playwright install chromium"
        )

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=TIKTOK_SESSION_DIR,
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        try:
            # Try upload page URLs in order
            opened = False
            for url in _UPLOAD_URLS:
                try:
                    logger.info("TikTok: navigating to %s", url)
                    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    await page.wait_for_timeout(2_000)
                    opened = True
                    break
                except Exception as exc:
                    logger.warning("TikTok: %s failed: %s", url, exc)
            if not opened:
                raise RuntimeError("Could not open any TikTok upload page")

            # Check session is valid
            if any(kw in page.url.lower() for kw in ("login", "passport", "signup")):
                raise RuntimeError(
                    "TikTok session expired.\nRun: python get_tiktok_session.py"
                )

            # Find file input — may be hidden, possibly inside an iframe
            file_input = None
            target = page   # default: main page frame

            # Check main page first
            try:
                el = page.locator('input[type="file"]').first
                await el.wait_for(state="attached", timeout=5_000)
                file_input = el
            except Exception:
                pass

            # Fall back to scanning iframes
            if file_input is None:
                for frame in page.frames:
                    if frame is page.main_frame:
                        continue
                    try:
                        el = frame.locator('input[type="file"]').first
                        await el.wait_for(state="attached", timeout=2_000)
                        file_input = el
                        target = frame
                        break
                    except Exception:
                        continue

            if file_input is None:
                raise RuntimeError(
                    "Could not find file upload input on TikTok page.\n"
                    "TikTok may have changed their UI. Try updating selectors."
                )

            logger.info("TikTok: uploading %s", os.path.basename(video_path))
            await file_input.set_input_files(video_path)

            # Wait for video to finish processing (preview or progress bar)
            logger.info("TikTok: waiting for video to process...")
            try:
                await target.locator(
                    'video[src], [data-e2e="video-preview"], [class*="video-preview"]'
                ).first.wait_for(state="visible", timeout=180_000)
            except Exception:
                # Some versions show a spinner then jump straight to editor
                await page.wait_for_timeout(10_000)

            await page.wait_for_timeout(2_000)

            # Check for upload failure before proceeding
            upload_failed = await page.evaluate("""
                () => {
                    const texts = ['upload failed', 'couldn\'t upload', 'upload error'];
                    const body = document.body.innerText.toLowerCase();
                    return texts.some(t => body.includes(t));
                }
            """)
            if upload_failed:
                raise RuntimeError(
                    "TikTok rejected the video file (\"Upload failed\"). "
                    "Check file format, size, or duration."
                )

            # Dismiss onboarding/tutorial overlay if present (react-joyride)
            await _dismiss_overlay(page)

            # Fill caption
            cap_selector = (
                '[data-e2e="caption-input"], '
                '.public-DraftEditor-content, '
                '[contenteditable="true"]'
            )
            try:
                cap_el = target.locator(cap_selector).first
                await cap_el.wait_for(state="visible", timeout=8_000)
                # Use JS click to bypass any remaining overlay
                await cap_el.evaluate("el => el.click()")
                await page.wait_for_timeout(300)
                # Select all existing text and replace
                await page.keyboard.press("Meta+a")
                await cap_el.type(caption[:2_200], delay=10)
                logger.info("TikTok: caption filled")
            except Exception as exc:
                logger.warning("TikTok: could not fill caption (%s) — continuing", exc)

            await page.wait_for_timeout(1_000)

            # Dismiss overlay again in case it reappeared
            await _dismiss_overlay(page)

            # Click Post button via JS to bypass any overlay.
            # Use progressively more specific selectors — TikTok uses different
            # markup depending on language/region. We deliberately exclude
            # sidebar/nav buttons by scoping to the editor area first.
            post_btn = None
            post_selectors = [
                # Most reliable: data attribute on the submit button
                '[data-e2e="post-button"]',
                # Editor-area specific submit (scoped away from sidebar)
                '.editor-btn-post button, [class*="btn-post"]',
                # Text-based, exact match to avoid sidebar buttons
                'button[class*="submit"], button[class*="publish"]',
                # Fallback text match
                'button:has-text("Post")',
                'button:has-text("Publish")',
                'button:has-text("Опубликовать")',
            ]
            for sel in post_selectors:
                try:
                    el = target.locator(sel).last  # last = rightmost/bottom = action button
                    await el.wait_for(state="visible", timeout=3_000)
                    post_btn = el
                    logger.info("TikTok: Post button found via '%s'", sel)
                    break
                except Exception:
                    continue

            if post_btn is None:
                raise RuntimeError("Could not find Post button on TikTok upload page")

            await post_btn.evaluate("el => el.click()")
            logger.info("TikTok: clicked Post, waiting for confirmation...")

            # Take a screenshot right after clicking to aid debugging
            try:
                shot_path = os.path.join(os.path.dirname(video_path), "tiktok_post_click.png")
                await page.screenshot(path=shot_path)
                logger.info("TikTok: post-click screenshot: %s", shot_path)
            except Exception:
                pass

            # Wait for success (redirect or confirmation element)
            try:
                await target.locator(
                    '[data-e2e="upload-success"], [class*="success"]'
                ).first.wait_for(state="visible", timeout=30_000)
            except Exception:
                # Some versions just redirect to profile — give it a few seconds
                await page.wait_for_timeout(6_000)

            # Check for post-click error dialog
            post_error = await page.evaluate("""
                () => {
                    const texts = ["couldn't upload", 'upload failed', 'something went wrong'];
                    const body = document.body.innerText.toLowerCase();
                    return texts.find(t => body.includes(t)) || null;
                }
            """)
            if post_error:
                raise RuntimeError(
                    f'TikTok post failed: "{post_error}" dialog appeared after clicking Post. '
                    'Check the tiktok_post_click.png screenshot for details.'
                )

            logger.info("TikTok: upload complete")

        except Exception:
            # Save screenshot for debugging before closing
            try:
                shot_path = os.path.join(os.path.dirname(video_path), "tiktok_debug.png")
                await page.screenshot(path=shot_path)
                logger.info("TikTok debug screenshot: %s", shot_path)
            except Exception:
                pass
            raise
        finally:
            await ctx.close()


async def upload_video(video_path: str, caption: str, retries: int = 2) -> None:
    """
    Upload *video_path* to TikTok with *caption*.
    Retries up to *retries* times on failure before raising.
    """
    if not is_configured():
        raise RuntimeError(
            "TikTok session not found.\nRun: python get_tiktok_session.py"
        )
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, retries + 2):
        try:
            if attempt > 1:
                wait = 30 * attempt
                logger.info("TikTok: retry %d/%d in %ds...", attempt - 1, retries, wait)
                await asyncio.sleep(wait)
            await _do_upload(video_path, caption)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning("TikTok attempt %d failed: %s", attempt, exc)
    raise last_exc
