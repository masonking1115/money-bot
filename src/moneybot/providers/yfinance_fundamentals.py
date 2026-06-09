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
