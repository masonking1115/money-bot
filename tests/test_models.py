from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from moneybot.models import Bar, Filing, Fundamentals, NewsItem


def test_bar_roundtrips_fields():
    bar = Bar(
        ts=datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc),
        open=10.0, high=11.0, low=9.5, close=10.5, volume=1000,
    )
    assert bar.close == 10.5
    assert bar.volume == 1000


def test_bar_rejects_negative_volume():
    with pytest.raises(ValidationError):
        Bar(
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
            open=1, high=1, low=1, close=1, volume=-5,
        )


def test_filing_computes_stable_content_hash():
    f1 = Filing(ticker="NVDA", form_type="10-K", filed_at=date(2026, 2, 1),
                accession_no="0001-26-000001", url="https://x/1", raw_text="hello")
    f2 = Filing(ticker="NVDA", form_type="10-K", filed_at=date(2026, 2, 1),
                accession_no="0001-26-000001", url="https://x/1", raw_text="hello")
    assert f1.content_hash == f2.content_hash
    assert len(f1.content_hash) == 64  # sha256 hex


def test_filing_hash_changes_with_text():
    base = dict(ticker="NVDA", form_type="8-K", filed_at=date(2026, 2, 1),
                accession_no="a", url="https://x")
    assert Filing(**base, raw_text="a").content_hash != Filing(**base, raw_text="b").content_hash


def test_newsitem_url_hash_is_deterministic():
    n = NewsItem(title="t", url="https://news/abc", published_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                 source="rss")
    assert n.url_hash == NewsItem(title="other", url="https://news/abc",
                                  published_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                                  source="rss").url_hash


def test_fundamentals_allows_optional_fields():
    fund = Fundamentals(ticker="AMD", as_of=date(2026, 6, 9))
    assert fund.market_cap is None
    assert fund.extra == {}
