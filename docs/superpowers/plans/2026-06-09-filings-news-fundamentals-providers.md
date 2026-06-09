# Filings, News & Fundamentals Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the data layer by adding SEC EDGAR filings, RSS news, and yfinance fundamentals providers behind clean protocols, integrated into the `DataLayer` with caching and point-in-time discipline.

**Architecture:** Three new providers implement three new protocols (`FilingsProvider`, `NewsProvider`, `FundamentalsProvider`), each isolating its network call in a `_fetch*` seam so unit tests run offline. The `DataLayer` facade gains `get_filings`, `get_news`, and `get_fundamentals` — universe-bounded, cached for live calls, and bypassing the cache for point-in-time (`as_of`) calls, exactly like `get_bars`. EDGAR needs a CIK per ticker, sourced from `universe.yaml` (manual override, consistent with the earnings-date approach).

**Tech Stack:** Python 3.11+, uv, pytest, pydantic v2, httpx (new), pandas, yfinance. Tests mock all network seams.

---

## Context for the implementer

Plan 1 is merged on `main`. Existing modules:
- `src/moneybot/models.py` — `Bar`, `Filing(ticker, form_type, filed_at: date, accession_no, url, raw_text|None, content_hash[computed])`, `NewsItem(ticker|None, title, url, published_at: datetime, source, summary|None, url_hash[computed])`, `Fundamentals(ticker, as_of: date, market_cap|None, pe_ratio|None, revenue|None, extra: dict)`.
- `src/moneybot/config.py` — `Settings(BaseSettings)` (env prefix `MONEYBOT_`), `TickerMeta(symbol, market_cap_tier|None, earnings_date|None)`, `Universe(sector, benchmark, tickers)` with `.symbols`/`.get(symbol)`, `load_universe(path)`.
- `src/moneybot/cache.py` — `Cache(root)` with `set_json/get_json` and `set_dataframe/get_dataframe`.
- `src/moneybot/providers/__init__.py` — `PriceProvider` protocol (`@runtime_checkable`).
- `src/moneybot/data_layer.py` — `DataLayer(universe, price_provider, cache)` with `_require_in_universe`, `get_bars`.

Pydantic round-trip for caching: dump with `model_dump(mode="json")` (ISO dates → strings, JSON-safe) and read back with `Model.model_validate(d)` (computed fields like `content_hash` are ignored on input). Use `cache.set_json` / `cache.get_json` for lists of these dumps.

All tests must be network-free: monkeypatch the provider `_fetch*` methods.

---

## File Structure

- Modify: `pyproject.toml` (add `httpx`)
- Modify: `src/moneybot/config.py` (add `cik` to `TickerMeta`; add `sec_user_agent` to `Settings`)
- Modify: `src/moneybot/providers/__init__.py` (add `FilingsProvider`, `NewsProvider`, `FundamentalsProvider` protocols)
- Create: `src/moneybot/providers/edgar_filings.py` — `EdgarFilingsProvider`
- Create: `src/moneybot/providers/rss_news.py` — `RssNewsProvider`
- Create: `src/moneybot/providers/yfinance_fundamentals.py` — `YFinanceFundamentalsProvider`
- Modify: `src/moneybot/data_layer.py` (add `get_filings`, `get_news`, `get_fundamentals`)
- Modify: `universe.example.yaml` (show `cik`), `.env.example` (show `MONEYBOT_SEC_USER_AGENT`)
- Modify: `README.md` (status bump)
- Tests: `tests/providers/test_edgar_filings.py`, `tests/providers/test_rss_news.py`, `tests/providers/test_yfinance_fundamentals.py`, and additions to `tests/test_data_layer.py`, `tests/test_config.py`

---

## Task 1: Add httpx, extend protocols and config

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/moneybot/providers/__init__.py`
- Modify: `src/moneybot/config.py`
- Modify: `universe.example.yaml`, `.env.example`
- Test: `tests/test_config.py` (add cases)

- [ ] **Step 1: Add `httpx` to `pyproject.toml` dependencies**

In the `[project]` `dependencies` list, add `"httpx>=0.27"` (keep the list otherwise unchanged). Then run `uv sync`.

Run: `uv sync`
Expected: httpx installed, no errors.

- [ ] **Step 2: Write failing config tests — append to `tests/test_config.py`**

```python
def test_ticker_meta_accepts_cik(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text(
        "sector: s\nbenchmark: B\ntickers:\n  - symbol: NVDA\n    cik: \"0001045810\"\n"
    )
    uni = load_universe(path)
    assert uni.get("NVDA").cik == "0001045810"


def test_settings_has_sec_user_agent_default(monkeypatch):
    monkeypatch.delenv("MONEYBOT_SEC_USER_AGENT", raising=False)
    settings = Settings()
    assert "moneybot" in settings.sec_user_agent.lower()
```

- [ ] **Step 3: Run the new config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "cik or sec_user_agent" -v`
Expected: FAIL (`cik` attribute missing / `sec_user_agent` missing).

- [ ] **Step 4: Extend `TickerMeta` and `Settings` in `src/moneybot/config.py`**

In `TickerMeta`, add the field:
```python
    cik: str | None = None
```

In `Settings`, add the field (after the alpaca credentials, before the model-tier fields):
```python
    sec_user_agent: str = "moneybot mason@voltai.com"
```

- [ ] **Step 5: Add the new protocols to `src/moneybot/providers/__init__.py`**

Append (keep the existing `PriceProvider` and imports; add `Filing`, `NewsItem`, `Fundamentals` to the moneybot imports):

```python
from moneybot.models import Filing, Fundamentals, NewsItem


@runtime_checkable
class FilingsProvider(Protocol):
    def get_recent_filings(
        self,
        ticker: str,
        types: list[str] | None = None,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[Filing]:
        """Return filings for a ticker, oldest first.

        types filters by form (e.g. ["10-K", "8-K"]); since drops filings before
        that date; if as_of is set, no filing with filed_at > as_of is returned.
        """
        ...


@runtime_checkable
class NewsProvider(Protocol):
    def get_news(
        self,
        query: str,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[NewsItem]:
        """Return news items for a query (ticker or sector term), oldest first.

        If as_of is set, no item with published_at.date() > as_of is returned.
        """
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    def get_fundamentals(self, ticker: str, as_of: date | None = None) -> Fundamentals:
        """Return a fundamentals snapshot. Phase-1 sources are current-only;
        as_of stamps the record but does not retrieve historical fundamentals."""
        ...
```

- [ ] **Step 6: Update `universe.example.yaml` and `.env.example`**

In `universe.example.yaml`, add a `cik` line under NVDA (CIKs are found at https://www.sec.gov/cgi-bin/browse-edgar — 10-digit, zero-padded):
```yaml
  - symbol: NVDA
    market_cap_tier: mega
    earnings_date: 2026-08-27   # manual override; verify each quarter
    cik: "0001045810"           # SEC CIK (zero-padded); needed for EDGAR filings
```

In `.env.example`, add:
```bash
MONEYBOT_SEC_USER_AGENT=moneybot you@example.com
```

- [ ] **Step 7: Run config tests + full suite to verify pass**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check src tests`
Expected: all config tests pass (including the 2 new); ruff clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/moneybot/providers/__init__.py src/moneybot/config.py universe.example.yaml .env.example tests/test_config.py
git commit -m "feat: add httpx, filings/news/fundamentals protocols, cik + sec_user_agent config"
```

---

## Task 2: EDGAR filings provider

**Files:**
- Create: `src/moneybot/providers/edgar_filings.py`
- Test: `tests/providers/test_edgar_filings.py`

- [ ] **Step 1: Write the failing test `tests/providers/test_edgar_filings.py`**

```python
from datetime import date

import pytest

from moneybot.providers import FilingsProvider
from moneybot.providers.edgar_filings import EdgarFilingsProvider


def _fake_submissions():
    # Mimics data.sec.gov submissions JSON (parallel arrays under filings.recent)
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "4"],
                "filingDate": ["2026-02-01", "2026-05-10", "2026-05-12"],
                "accessionNumber": [
                    "0001045810-26-000010",
                    "0001045810-26-000031",
                    "0001045810-26-000033",
                ],
                "primaryDocument": ["nvda-10k.htm", "nvda-8k.htm", "form4.xml"],
            }
        }
    }


def _provider():
    prov = EdgarFilingsProvider(cik_map={"NVDA": "0001045810"}, user_agent="test ua")
    prov._fetch_submissions = lambda cik10: _fake_submissions()
    return prov


def test_satisfies_protocol():
    assert isinstance(EdgarFilingsProvider(cik_map={}, user_agent="x"), FilingsProvider)


def test_parses_filings_oldest_first_with_urls():
    filings = _provider().get_recent_filings("NVDA")
    assert [f.form_type for f in filings] == ["10-K", "8-K", "4"]
    assert filings[0].filed_at == date(2026, 2, 1)
    # URL: archives path uses CIK without leading zeros and accession w/o dashes
    assert filings[0].url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000010/nvda-10k.htm"
    )


def test_filters_by_form_type():
    filings = _provider().get_recent_filings("NVDA", types=["10-K", "8-K"])
    assert [f.form_type for f in filings] == ["10-K", "8-K"]


def test_filters_by_since():
    filings = _provider().get_recent_filings("NVDA", since=date(2026, 5, 1))
    assert [f.form_type for f in filings] == ["8-K", "4"]


def test_as_of_excludes_future_filings():
    filings = _provider().get_recent_filings("NVDA", as_of=date(2026, 5, 11))
    assert [f.filed_at for f in filings] == [date(2026, 2, 1), date(2026, 5, 10)]


def test_unknown_cik_raises():
    prov = EdgarFilingsProvider(cik_map={}, user_agent="x")
    with pytest.raises(ValueError, match="no CIK"):
        prov.get_recent_filings("TSLA")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/providers/test_edgar_filings.py -v`
Expected: FAIL (`ModuleNotFoundError: ... edgar_filings`).

- [ ] **Step 3: Write `src/moneybot/providers/edgar_filings.py`**

```python
"""SEC EDGAR filings provider (free, phase-1).

Resolves a ticker to a CIK via the injected cik_map, fetches the company's
recent-submissions JSON from data.sec.gov, and normalizes it into Filing models.
"""

from __future__ import annotations

from datetime import date

import httpx

from moneybot.models import Filing

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


class EdgarFilingsProvider:
    def __init__(self, cik_map: dict[str, str], user_agent: str) -> None:
        self.cik_map = cik_map
        self.user_agent = user_agent

    def _fetch_submissions(self, cik10: str) -> dict:
        # Network seam — patched in tests so no request is made.
        resp = httpx.get(
            SUBMISSIONS_URL.format(cik10=cik10),
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_recent_filings(
        self,
        ticker: str,
        types: list[str] | None = None,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[Filing]:
        if ticker not in self.cik_map:
            raise ValueError(f"no CIK configured for {ticker} (add it to universe.yaml)")

        cik_raw = str(self.cik_map[ticker])
        cik10 = cik_raw.zfill(10)
        cik_digits = cik_raw.lstrip("0") or "0"

        data = self._fetch_submissions(cik10)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        out: list[Filing] = []
        for form, filed_str, acc, doc in zip(forms, dates, accs, docs):
            filed = date.fromisoformat(filed_str)
            if types is not None and form not in types:
                continue
            if since is not None and filed < since:
                continue
            if as_of is not None and filed > as_of:
                continue
            url = ARCHIVES_URL.format(
                cik=cik_digits, acc_nodash=acc.replace("-", ""), doc=doc
            )
            out.append(
                Filing(
                    ticker=ticker,
                    form_type=form,
                    filed_at=filed,
                    accession_no=acc,
                    url=url,
                )
            )
        out.sort(key=lambda f: f.filed_at)
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/providers/test_edgar_filings.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/providers/edgar_filings.py tests/providers/test_edgar_filings.py
git commit -m "feat: EDGAR filings provider with type/since/as_of filtering"
```

---

## Task 3: RSS news provider

**Files:**
- Create: `src/moneybot/providers/rss_news.py`
- Test: `tests/providers/test_rss_news.py`

- [ ] **Step 1: Write the failing test `tests/providers/test_rss_news.py`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/providers/test_rss_news.py -v`
Expected: FAIL (`ModuleNotFoundError: ... rss_news`).

- [ ] **Step 3: Write `src/moneybot/providers/rss_news.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/providers/test_rss_news.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/providers/rss_news.py tests/providers/test_rss_news.py
git commit -m "feat: RSS news provider with since/as_of filtering"
```

---

## Task 4: yfinance fundamentals provider

**Files:**
- Create: `src/moneybot/providers/yfinance_fundamentals.py`
- Test: `tests/providers/test_yfinance_fundamentals.py`

- [ ] **Step 1: Write the failing test `tests/providers/test_yfinance_fundamentals.py`**

```python
from datetime import date

from moneybot.providers import FundamentalsProvider
from moneybot.providers.yfinance_fundamentals import YFinanceFundamentalsProvider


def _provider():
    prov = YFinanceFundamentalsProvider()
    prov._fetch_info = lambda ticker: {
        "marketCap": 3_000_000_000_000,
        "trailingPE": 55.5,
        "totalRevenue": 130_000_000_000,
        "irrelevant": "ignored",
    }
    return prov


def test_satisfies_protocol():
    assert isinstance(YFinanceFundamentalsProvider(), FundamentalsProvider)


def test_maps_info_fields():
    fund = _provider().get_fundamentals("NVDA", as_of=date(2026, 6, 9))
    assert fund.ticker == "NVDA"
    assert fund.as_of == date(2026, 6, 9)
    assert fund.market_cap == 3_000_000_000_000
    assert fund.pe_ratio == 55.5
    assert fund.revenue == 130_000_000_000


def test_missing_fields_become_none():
    prov = YFinanceFundamentalsProvider()
    prov._fetch_info = lambda ticker: {}
    fund = prov.get_fundamentals("AMD", as_of=date(2026, 6, 9))
    assert fund.market_cap is None
    assert fund.pe_ratio is None
    assert fund.revenue is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/providers/test_yfinance_fundamentals.py -v`
Expected: FAIL (`ModuleNotFoundError: ... yfinance_fundamentals`).

- [ ] **Step 3: Write `src/moneybot/providers/yfinance_fundamentals.py`**

```python
"""yfinance fundamentals provider (free, phase-1).

NOTE: yfinance exposes only the CURRENT fundamentals snapshot — there is no
historical point-in-time fundamentals here. `as_of` stamps the returned record
but does not retrieve as-of-date values. Treat fundamentals cautiously in
backtests until a point-in-time fundamentals feed is added.
"""

from __future__ import annotations

from datetime import date

import yfinance as yf

from moneybot.models import Fundamentals


class YFinanceFundamentalsProvider:
    def _fetch_info(self, ticker: str) -> dict:
        # Network seam — patched in tests so no request is made.
        return dict(yf.Ticker(ticker).info)

    def get_fundamentals(self, ticker: str, as_of: date | None = None) -> Fundamentals:
        info = self._fetch_info(ticker)
        return Fundamentals(
            ticker=ticker,
            as_of=as_of or date.today(),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            revenue=info.get("totalRevenue"),
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/providers/test_yfinance_fundamentals.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/providers/yfinance_fundamentals.py tests/providers/test_yfinance_fundamentals.py
git commit -m "feat: yfinance fundamentals provider (current-snapshot, phase-1)"
```

---

## Task 5: DataLayer integration

**Files:**
- Modify: `src/moneybot/data_layer.py`
- Test: `tests/test_data_layer.py` (add cases)

- [ ] **Step 1: Write the failing tests — append to `tests/test_data_layer.py`**

Add these imports at the top of the file if not already present: `from moneybot.models import Filing, Fundamentals, NewsItem`.

```python
class StubFilings:
    def __init__(self):
        self.calls = 0

    def get_recent_filings(self, ticker, types=None, since=None, as_of=None):
        self.calls += 1
        f = Filing(ticker=ticker, form_type="8-K", filed_at=date(2026, 6, 9),
                   accession_no="a-1", url="https://x/1")
        return [f]


class StubNews:
    def __init__(self):
        self.calls = 0

    def get_news(self, query, since=None, as_of=None):
        self.calls += 1
        from datetime import datetime, timezone
        return [NewsItem(title="t", url="https://n/1",
                         published_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                         source="stub")]


class StubFundamentals:
    def __init__(self):
        self.calls = 0

    def get_fundamentals(self, ticker, as_of=None):
        self.calls += 1
        return Fundamentals(ticker=ticker, as_of=as_of or date(2026, 6, 9),
                            market_cap=1.0)


def test_get_filings_returns_and_caches(tmp_path):
    filings = StubFilings()
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path),
                   filings_provider=filings)
    out = dl.get_filings("NVDA")
    assert [f.form_type for f in out] == ["8-K"]
    assert isinstance(out[0], Filing)
    dl.get_filings("NVDA")  # second call served from cache
    assert filings.calls == 1


def test_get_filings_requires_provider(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path))
    with pytest.raises(ValueError, match="no filings provider"):
        dl.get_filings("NVDA")


def test_get_filings_outside_universe_rejected(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path),
                   filings_provider=StubFilings())
    with pytest.raises(ValueError, match="not in universe"):
        dl.get_filings("TSLA")


def test_get_filings_as_of_bypasses_cache(tmp_path):
    filings = StubFilings()
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path),
                   filings_provider=filings)
    dl.get_filings("NVDA", as_of=date(2026, 6, 9))
    dl.get_filings("NVDA", as_of=date(2026, 6, 9))
    assert filings.calls == 2


def test_get_news_returns_and_caches(tmp_path):
    news = StubNews()
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path),
                   news_provider=news)
    out = dl.get_news("NVDA")
    assert isinstance(out[0], NewsItem)
    dl.get_news("NVDA")
    assert news.calls == 1


def test_get_fundamentals_returns_and_caches(tmp_path):
    fund = StubFundamentals()
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path),
                   fundamentals_provider=fund)
    out = dl.get_fundamentals("NVDA")
    assert out.market_cap == 1.0
    dl.get_fundamentals("NVDA")
    assert fund.calls == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_layer.py -k "filings or news or fundamentals" -v`
Expected: FAIL (`DataLayer` has no `filings_provider` kwarg / no `get_filings`).

- [ ] **Step 3: Update `src/moneybot/data_layer.py`**

Replace the imports and class with this (preserves existing `get_bars` behavior, adds optional providers + three methods):

```python
"""DataLayer facade: bounds access to the universe, caches live data,
and enforces point-in-time access for backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.models import Filing, Fundamentals, NewsItem
from moneybot.providers import (
    FilingsProvider,
    FundamentalsProvider,
    NewsProvider,
    PriceProvider,
)


class DataLayer:
    def __init__(
        self,
        universe: Universe,
        price_provider: PriceProvider,
        cache: Cache,
        *,
        filings_provider: FilingsProvider | None = None,
        news_provider: NewsProvider | None = None,
        fundamentals_provider: FundamentalsProvider | None = None,
    ) -> None:
        self.universe = universe
        self.prices = price_provider
        self.cache = cache
        self.filings = filings_provider
        self.news = news_provider
        self.fundamentals = fundamentals_provider

    def _require_in_universe(self, ticker: str) -> None:
        if ticker not in self.universe.symbols and ticker != self.universe.benchmark:
            raise ValueError(f"{ticker} not in universe")

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        self._require_in_universe(ticker)

        # Point-in-time requests bypass the cache: backtest correctness beats reuse,
        # and a cached "live" frame may contain bars newer than as_of.
        if as_of is not None:
            df = self.prices.get_bars(ticker, timeframe, lookback, as_of=as_of)
            if not df.empty and df["ts"].dt.date.max() > as_of:
                raise ValueError(
                    f"provider returned bars after as_of={as_of} (point-in-time violation)"
                )
            return df

        key = f"bars:{ticker}:{timeframe}:{lookback}"
        cached = self.cache.get_dataframe(key)
        if cached is not None and not cached.empty:
            return cached
        df = self.prices.get_bars(ticker, timeframe, lookback)
        if not df.empty:
            self.cache.set_dataframe(key, df)
        return df

    def get_filings(
        self,
        ticker: str,
        types: list[str] | None = None,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[Filing]:
        self._require_in_universe(ticker)
        if self.filings is None:
            raise ValueError("no filings provider configured")

        if as_of is not None:
            filings = self.filings.get_recent_filings(
                ticker, types=types, since=since, as_of=as_of
            )
            for f in filings:
                if f.filed_at > as_of:
                    raise ValueError(
                        f"provider returned a filing after as_of={as_of}"
                    )
            return filings

        type_key = ",".join(sorted(types)) if types else "all"
        key = f"filings:{ticker}:{type_key}:{since or 'none'}"
        cached = self.cache.get_json(key)
        if cached is not None:
            return [Filing.model_validate(d) for d in cached]
        filings = self.filings.get_recent_filings(ticker, types=types, since=since)
        self.cache.set_json(key, [f.model_dump(mode="json") for f in filings])
        return filings

    def get_news(
        self,
        ticker: str,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[NewsItem]:
        self._require_in_universe(ticker)
        if self.news is None:
            raise ValueError("no news provider configured")

        if as_of is not None:
            return self.news.get_news(ticker, since=since, as_of=as_of)

        key = f"news:{ticker}:{since or 'none'}"
        cached = self.cache.get_json(key)
        if cached is not None:
            return [NewsItem.model_validate(d) for d in cached]
        items = self.news.get_news(ticker, since=since)
        self.cache.set_json(key, [n.model_dump(mode="json") for n in items])
        return items

    def get_fundamentals(
        self, ticker: str, as_of: date | None = None
    ) -> Fundamentals:
        self._require_in_universe(ticker)
        if self.fundamentals is None:
            raise ValueError("no fundamentals provider configured")

        if as_of is not None:
            return self.fundamentals.get_fundamentals(ticker, as_of=as_of)

        key = f"fundamentals:{ticker}"
        cached = self.cache.get_json(key)
        if cached is not None:
            return Fundamentals.model_validate(cached)
        fund = self.fundamentals.get_fundamentals(ticker)
        self.cache.set_json(key, fund.model_dump(mode="json"))
        return fund
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_layer.py -v`
Expected: all pass (the original 6 plus the 6 new).

- [ ] **Step 5: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/data_layer.py tests/test_data_layer.py
git commit -m "feat: DataLayer get_filings/get_news/get_fundamentals with caching + point-in-time"
```

---

## Task 6: README status bump

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section of `README.md`**

Replace the `## Status` section body with:
```markdown
## Status

- Phase 1: foundation + point-in-time price data layer.
- Phase 2: filings (SEC EDGAR), news (RSS), and fundamentals (yfinance) providers,
  integrated into the DataLayer with caching and point-in-time access.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README status — phase 2 data providers"
```

---

## Self-Review Notes

- **Spec coverage (Plan 2 scope):** filings provider (EDGAR) ✓ Task 2; news provider (RSS) ✓ Task 3; fundamentals provider ✓ Task 4; pluggable behind common protocols ✓ Task 1; caching + point-in-time + universe bounding integrated into DataLayer ✓ Task 5; cached-by-params keys mirror the price approach. Structured *agent extraction* of filings (caching by content_hash) is Plan 4, not here — Plan 2 provides filing metadata + URL only (raw_text stays None).
- **Type consistency:** `FilingsProvider.get_recent_filings`, `NewsProvider.get_news`, `FundamentalsProvider.get_fundamentals` signatures match between the protocols (Task 1), the implementations (Tasks 2–4), the stubs, and the `DataLayer` calls (Task 5). Pydantic round-trip uses `model_dump(mode="json")` ↔ `model_validate`.
- **Network isolation:** every provider isolates its call in `_fetch*` and all tests monkeypatch it — no test hits the network.
- **Placeholder scan:** every step contains complete, runnable code and exact commands.
- **Known phase-1 limitations (documented in code):** fundamentals are current-snapshot only (no historical point-in-time); filing full-text fetch deferred to Plan 4.
