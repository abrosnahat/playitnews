"""Pluggable per-project news sources.

Each project (see projects.json) declares a ``source`` name. This module maps
that name to an object exposing two coroutines:

    async def get_latest_links(session) -> list[{"url", "title"}]
    async def scrape_article(session, url) -> scraper.Article | None

``playground`` delegates to the original single-site implementation in
``scraper.py``; ``championat`` parses championat.ru UFC news.
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


class ChampionatSource:
    """championat.ru UFC — reads the 'Обсуждаемые' tab of the top-news block."""

    BASE = "https://www.championat.ru"
    LISTING_URL = "https://www.championat.ru/news/boxing/_ufc/1.html"

    async def get_latest_links(self, session: aiohttp.ClientSession) -> list[dict]:
        html = await fetch(session, self.LISTING_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")

        # The "Обсуждаемые" (discussed) tab content of the top-news widget.
        tab = (
            soup.select_one("div.tabs-content._discussed[data-type='discussed']")
            or soup.select_one("[data-type='discussed']")
        )
        if tab is None:
            logger.warning("championat: вкладка 'Обсуждаемые' не найдена")
            return []

        articles: list[dict] = []
        seen: set[str] = set()
        for a in tab.select("a.news-item__title[href]"):
            href = a.get("href", "").strip()
            if not href or "#comments" in href:
                continue
            full = urljoin(self.BASE, href)
            if full in seen:
                continue
            title = a.get_text(strip=True)
            if not title:
                continue
            seen.add(full)
            articles.append({"url": full, "title": title})

        logger.info("championat: найдено обсуждаемых новостей: %d", len(articles))
        return articles

    async def scrape_article(self, session: aiohttp.ClientSession, url: str) -> Optional[Article]:
        html = await fetch(session, url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        # Drop "external article" recommendation blocks entirely so their
        # images are never picked up as the article hero.
        for ext in soup.select(".external-article, .external-article__item"):
            ext.decompose()

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else url

        body = (
            soup.select_one("div.article-content")
            or soup.select_one("article")
            or soup.select_one("main")
        )
        if body:
            for tag in body.select(
                "script, style, nav, aside, .banner, .advertisement, "
                ".external-article, .related-articles, .share, .comments"
            ):
                tag.decompose()
            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in body.find_all(["p", "li", "blockquote"])
            ]
            text = "\n".join(p for p in paragraphs if len(p) > 20)
        else:
            text = soup.get_text(separator="\n", strip=True)[:3000]

        # Hero image: prefer the article-head photo, fall back to og:image.
        # championat serves a small crop under /s/<WxH>/; swapping the /s/
        # prefix for /b/ yields the full-res image.
        def _normalize(src: str) -> str:
            return re.sub(r"(img\.championat\.ru)/s/", r"\1/b/", src.strip())

        image_urls: list[str] = []
        photo = soup.select_one(".article-head__photo img")
        if photo:
            src = (
                photo.get("src")
                or photo.get("data-src")
                or photo.get("data-original")
                or ""
            )
            if src:
                hero = _normalize(src)
                if _is_valid_image_url(hero):
                    image_urls.append(hero)

        if not image_urls:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                hero = _normalize(og["content"])
                if _is_valid_image_url(hero):
                    image_urls.append(hero)

        # championat embeds video via a custom widget we don't parse → no pg_embeds.
        return Article(url=url, title=title, text=text, image_urls=image_urls, pg_embeds=[])


_SOURCES: dict[str, NewsSource] = {
    "playground": PlaygroundSource(),
    "championat": ChampionatSource(),
}


def get_source(name: str | None) -> NewsSource:
    """Return the source implementation for a project, defaulting to playground."""
    return _SOURCES.get(name or "", _SOURCES["playground"])
