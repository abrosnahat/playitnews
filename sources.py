"""Pluggable per-project news sources.

Each project (see projects.json) declares a ``source`` name. This module maps
that name to an object exposing two coroutines:

    async def get_latest_links(session) -> list[{"url", "title"}]
    async def scrape_article(session, url) -> scraper.Article | None

``playground`` delegates to the original single-site implementation in
``scraper.py``; ``fighttime`` parses fighttime.ru MMA/UFC/boxing news.
"""
import logging
import re
from typing import Optional, Protocol
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

import scraper
from scraper import Article, _is_valid_image_url, fetch

logger = logging.getLogger(__name__)


class NewsSource(Protocol):
    async def get_latest_links(self, session: aiohttp.ClientSession) -> list[dict]: ...
    async def scrape_article(self, session: aiohttp.ClientSession, url: str) -> Optional[Article]: ...


class PlaygroundSource:
    """playground.ru — delegates to the original scraper implementation."""

    async def get_latest_links(self, session: aiohttp.ClientSession) -> list[dict]:
        return await scraper.get_latest_article_links(session)

    async def scrape_article(self, session: aiohttp.ClientSession, url: str) -> Optional[Article]:
        return await scraper.scrape_article(session, url)


class FightTimeSource:
    """fighttime.ru — MMA/UFC/boxing news listing + article scraper (Joomla/K2 site)."""

    BASE = "https://fighttime.ru"
    LISTING_URL = "https://fighttime.ru/news.html"

    async def get_latest_links(self, session: aiohttp.ClientSession) -> list[dict]:
        html = await fetch(session, self.LISTING_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")

        articles: list[dict] = []
        seen: set[str] = set()
        for a in soup.select("a.story-item__title[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            full = urljoin(self.BASE, href)
            title = a.get_text(strip=True)
            if title and full not in seen:
                seen.add(full)
                articles.append({"url": full, "title": title})

        logger.info("fighttime: unique=%d", len(articles))
        return articles

    async def scrape_article(self, session: aiohttp.ClientSession, url: str) -> Optional[Article]:
        html = await fetch(session, url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.select_one("h1.story-title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else url

        body = soup.select_one("div.itemFullText")
        if body:
            for tag in body.select("script, style"):
                tag.decompose()
            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in body.find_all(["p", "li", "blockquote"])
            ]
            text = "\n".join(p for p in paragraphs if len(p) > 20)
        else:
            text = soup.get_text(separator="\n", strip=True)[:3000]

        # Hero image: the featured image, fall back to og:image.
        image_urls: list[str] = []
        photo = soup.select_one("#feat-img-reg img.itemImage")
        if photo:
            src = (
                photo.get("src")
                or photo.get("data-src")
                or photo.get("data-lazy-src")
                or ""
            )
            if src:
                hero = urljoin(self.BASE, src.strip())
                if _is_valid_image_url(hero):
                    image_urls.append(hero)

        if not image_urls:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                hero = urljoin(self.BASE, og["content"].strip())
                if _is_valid_image_url(hero):
                    image_urls.append(hero)

        # fighttime.ru doesn't embed playground/championat video widgets.
        return Article(url=url, title=title, text=text, image_urls=image_urls, pg_embeds=[])


_SOURCES: dict[str, NewsSource] = {
    "playground": PlaygroundSource(),
    "fighttime": FightTimeSource(),
}


def get_source(name: str | None) -> NewsSource:
    """Return the source implementation for a project, defaulting to playground."""
    return _SOURCES.get(name or "", _SOURCES["playground"])
