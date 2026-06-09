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

        type_key = "|".join(sorted(types)) if types else "all"
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
            items = self.news.get_news(query=ticker, since=since, as_of=as_of)
            for n in items:
                if n.published_at.date() > as_of:
                    raise ValueError(
                        f"provider returned news after as_of={as_of} (point-in-time violation)"
                    )
            return items

        key = f"news:{ticker}:{since or 'none'}"
        cached = self.cache.get_json(key)
        if cached is not None:
            return [NewsItem.model_validate(d) for d in cached]
        items = self.news.get_news(query=ticker, since=since)
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
