"""Data provider protocols. Implementations are swappable behind these interfaces."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


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
