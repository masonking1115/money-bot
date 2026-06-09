"""DataLayer facade: bounds access to the universe, caches live data,
and enforces point-in-time access for backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.providers import PriceProvider


class DataLayer:
    def __init__(self, universe: Universe, price_provider: PriceProvider, cache: Cache) -> None:
        self.universe = universe
        self.prices = price_provider
        self.cache = cache

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
            return self.prices.get_bars(ticker, timeframe, lookback, as_of=as_of)

        key = f"bars:{ticker}:{timeframe}:{lookback}"
        cached = self.cache.get_dataframe(key)
        if cached is not None:
            return cached

        df = self.prices.get_bars(ticker, timeframe, lookback)
        self.cache.set_dataframe(key, df)
        return df
