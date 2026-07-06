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


def _extract_championat_video_embeds(scope, referrer: str) -> list[dict]:
    """Find championat's Rambler Video Platform widgets: ``<div class="video-wrapper
    js-player" data-id="record::<uuid>">`` (the ``_inited`` class is added client-side
    by their JS and isn't present in the raw HTML)."""
    embeds: list[dict] = []
    for wrapper in scope.select("div.video-wrapper[data-id]"):
        data_id = wrapper.get("data-id", "").strip()
        if data_id.startswith("record::"):
            embeds.append({"type": "championat", "id": data_id, "referrer": referrer})
    return embeds


class ChampionatSource:
    """championat.ru UFC — reads the 'Обсуждаемые' tab of the top-news block."""

    BASE = "https://www.championat.ru"
    LISTING_URL = "https://www.championat.ru/news/boxing/1.html"

    async def get_latest_links(self, session: aiohttp.ClientSession) -> list[dict]:
        html = await fetch(session, self.LISTING_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")

        def _links_from_tab(tab) -> list[dict]:
            result: list[dict] = []
            if tab is None:
                return result
            for a in tab.select("a.news-item__title[href]"):
                href = a.get("href", "").strip()
                if not href or "#comments" in href:
                    continue
                full = urljoin(self.BASE, href)
                title = a.get_text(strip=True)
                if title:
                    result.append({"url": full, "title": title})
            return result

        # «Обсуждаемые» tab (5 articles)
        discussed = (
            soup.select_one("div.tabs-content._discussed[data-type='discussed']")
            or soup.select_one("[data-type='discussed']")
        )
        # «Главные новости» tab (5 articles)
        main = (
            soup.select_one("div.tabs-content._main[data-type='main']")
            or soup.select_one("[data-type='main']")
        )

        articles: list[dict] = []
        seen: set[str] = set()
        for item in _links_from_tab(discussed)[:5] + _links_from_tab(main)[:5]:
            if item["url"] not in seen:
                seen.add(item["url"])
                articles.append(item)

        logger.info(
            "championat: discussed=%d main=%d unique=%d",
            len(_links_from_tab(discussed)),
            len(_links_from_tab(main)),
            len(articles),
        )
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
        pg_embeds: list[dict] = []
        if body:
            pg_embeds = _extract_championat_video_embeds(body, url)
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

        # championat embeds video via the Rambler Video Platform widget — resolved
        # and downloaded through scraper.download_videos (type "championat").
        return Article(url=url, title=title, text=text, image_urls=image_urls, pg_embeds=pg_embeds)


_SOURCES: dict[str, NewsSource] = {
    "playground": PlaygroundSource(),
    "championat": ChampionatSource(),
}


def get_source(name: str | None) -> NewsSource:
    """Return the source implementation for a project, defaulting to playground."""
    return _SOURCES.get(name or "", _SOURCES["playground"])
