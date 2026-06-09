"""Data provider protocols. Implementations are swappable behind these interfaces."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from moneybot.models import Filing, Fundamentals, NewsItem


@runtime_checkable
class PriceProvider(Protocol):
    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV bars with a tz-aware 'ts' column, oldest first.

        If as_of is given, no bar with ts.date() > as_of may be returned
        (point-in-time discipline — prevents lookahead).
        """
        ...


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
