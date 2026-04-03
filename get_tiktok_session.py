#!/usr/bin/env python3
"""
One-shot script: opens a visible Chrome browser so you can log into TikTok.
The session (cookies + localStorage) is saved to tiktok_session/ and reused
by tiktok_publisher.py for headless uploads.

Steps:
  1. pip install playwright
     playwright install chromium          (or: .venv/bin/playwright install chromium)
  2. python get_tiktok_session.py
  3. Log into TikTok in the opened browser window
  4. Press Enter in this terminal to save the session and close the browser
"""
import asyncio
import os
import sys

SESSION_DIR = os.path.join(os.path.dirname(__file__), "tiktok_session")


async def main() -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  .venv/bin/pip install playwright")
        print("  .venv/bin/playwright install chromium")
        sys.exit(1)

    print(f"Session will be saved to: {SESSION_DIR}")
    print("Opening browser — log into TikTok, then press Enter here.\n")

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=SESSION_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://www.tiktok.com/login", wait_until="domcontentloaded")

        print("Browser opened. Log into TikTok with your account.")
        print("After you see your TikTok feed, press Enter to save session...")
        await asyncio.get_event_loop().run_in_executor(None, input)

        print("Saving session...")
        await ctx.close()

    print(f"\nSession saved to: {SESSION_DIR}")
    print("Now tiktok_publisher.py can upload videos headlessly.")


if __name__ == "__main__":
    asyncio.run(main())
