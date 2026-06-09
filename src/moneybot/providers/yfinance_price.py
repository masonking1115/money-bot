"""yfinance-backed price provider (free, phase-1)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


class YFinancePriceProvider:
    """Fetches OHLCV bars from yfinance and normalizes them to the house schema."""

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # Isolated for testing — patched in unit tests so no network is hit.
        return yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False,
        )

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        period = f"{max(lookback, 1)}d" if timeframe.endswith("d") else f"{max(lookback, 1)}d"
        raw = self._download(ticker, period=period, interval=timeframe)
        if raw.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        # yfinance can return a column MultiIndex for single tickers; flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        reset = raw.reset_index()
        first_col = reset.columns[0]  # the datetime index becomes the first column
        df = reset.rename(
            columns={
                first_col: "ts",
                "Datetime": "ts",
                "Date": "ts",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["ts", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("ts").reset_index(drop=True)

        if as_of is not None:
            df = df[df["ts"].dt.date <= as_of].reset_index(drop=True)
        return df
