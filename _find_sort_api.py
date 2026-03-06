import asyncio
import aiohttp
import ssl
import certifi
from bs4 import BeautifulSoup

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://www.playground.ru/news", headers=HEADERS, ssl=SSL_CONTEXT) as r:
            html = await r.text()
        soup = BeautifulSoup(html, "html.parser")

        # Look for inline JS mentioning sort/creation
        for s in soup.find_all("script", src=False):
            txt = s.get_text()
            if "creation_date" in txt or "js-post-sort" in txt:
                print("=== INLINE JS ===")
                lines = txt.split("\n")
                for i, l in enumerate(lines):
                    if any(k in l for k in ["sort", "creation", "url", "ajax", "fetch", "request"]):
                        print(f"  {i}: {l.strip()[:120]}")
                break

        # Also print all external JS filenames
        print("\n=== JS SCRIPTS ===")
        for s in soup.find_all("script", src=True):
            print(" ", s["src"][:100])

asyncio.run(main())
