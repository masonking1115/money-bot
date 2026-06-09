from datetime import date

from moneybot.providers import NewsProvider
from moneybot.providers.rss_news import RssNewsProvider

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>NVDA beats estimates</title>
    <link>https://news/1</link>
    <pubDate>Mon, 08 Jun 2026 13:00:00 GMT</pubDate>
    <description>Strong quarter</description>
  </item>
  <item>
    <title>NVDA announces product</title>
    <link>https://news/2</link>
    <pubDate>Tue, 09 Jun 2026 09:00:00 GMT</pubDate>
    <description>New chip</description>
  </item>
</channel></rss>
"""


def _provider():
    prov = RssNewsProvider()
    prov._fetch = lambda url: SAMPLE_RSS
    return prov


def test_satisfies_protocol():
    assert isinstance(RssNewsProvider(), NewsProvider)


def test_parses_items_oldest_first():
    items = _provider().get_news("NVDA")
    assert [i.title for i in items] == ["NVDA beats estimates", "NVDA announces product"]
    assert items[0].url == "https://news/1"
    assert items[0].source == "google_news"
    assert items[0].summary == "Strong quarter"
    assert items[0].published_at.tzinfo is not None


def test_as_of_excludes_future_items():
    items = _provider().get_news("NVDA", as_of=date(2026, 6, 8))
    assert [i.title for i in items] == ["NVDA beats estimates"]


def test_since_excludes_old_items():
    items = _provider().get_news("NVDA", since=date(2026, 6, 9))
    assert [i.title for i in items] == ["NVDA announces product"]


def test_query_is_url_encoded_into_template():
    captured = {}

    prov = RssNewsProvider(feed_url_template="https://x/?q={query}")
    prov._fetch = lambda url: captured.setdefault("url", url) or SAMPLE_RSS
    prov.get_news("NV DA")
    assert captured["url"] == "https://x/?q=NV+DA"
