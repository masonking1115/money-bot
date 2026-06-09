"""RSS news provider (free, phase-1).

Fetches an RSS feed for a query and normalizes <item> entries into NewsItem
models. Defaults to Google News search RSS, which needs no API key. The feed
URL is a template with a {query} placeholder.
"""

from __future__ import annotations

from datetime import date, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import httpx

from moneybot.models import NewsItem

DEFAULT_TEMPLATE = "https://news.google.com/rss/search?q={query}"


class RssNewsProvider:
    def __init__(
        self, feed_url_template: str = DEFAULT_TEMPLATE, source: str = "google_news"
    ) -> None:
        self.feed_url_template = feed_url_template
        self.source = source

    def _fetch(self, url: str) -> str:
        # Network seam — patched in tests so no request is made.
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.text

    def get_news(
        self,
        query: str,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[NewsItem]:
        url = self.feed_url_template.format(query=quote_plus(query))
        xml = self._fetch(url)
        root = ET.fromstring(xml)

        items: list[NewsItem] = []
        for item in root.iter("item"):
            pub = item.findtext("pubDate")
            if not pub:
                continue
            published = parsedate_to_datetime(pub)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if since is not None and published.date() < since:
                continue
            if as_of is not None and published.date() > as_of:
                continue
            items.append(
                NewsItem(
                    title=(item.findtext("title") or "").strip(),
                    url=(item.findtext("link") or "").strip(),
                    published_at=published,
                    source=self.source,
                    summary=item.findtext("description"),
                )
            )
        items.sort(key=lambda n: n.published_at)
        return items
