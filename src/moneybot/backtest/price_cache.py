"""Persisted point-in-time price cache, keyed by (ticker, timeframe, lookback, as_of).

Wraps any PriceProvider. record mode populates on miss; replay mode requires a hit.
Bars are stored as JSON (ts serialized as ISO-8601 with offset) so reads reconstruct
a tz-aware 'ts' column identical to the live provider's contract."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from moneybot.providers import PriceProvider

_BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class CachingPriceProvider:
    def __init__(self, inner: PriceProvider, *, root: str | Path, mode: Literal["record", "replay"]) -> None:
        self._inner = inner
        self._dir = Path(root) / "prices"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode

    def _path(self, ticker: str, timeframe: str, lookback: int, as_of: date | None) -> Path:
        key = f"{ticker}__{timeframe}__{lookback}__{as_of}"
        return self._dir / f"{key}.json"

    def get_bars(
        self, ticker: str, timeframe: str, lookback: int, as_of: date | None = None
    ) -> pd.DataFrame:
        path = self._path(ticker, timeframe, lookback, as_of)
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(records, columns=_BAR_COLUMNS)
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            return df
        if self._mode == "replay":
            raise RuntimeError(
                f"cache miss in replay mode for {ticker} {timeframe} {lookback} as_of={as_of}; "
                "run a record pass first"
            )
        df = self._inner.get_bars(ticker, timeframe, lookback, as_of=as_of)
        out = df.copy()
        if not out.empty:
            out["ts"] = pd.to_datetime(out["ts"], utc=True).map(lambda t: t.isoformat())
        records = out.to_dict(orient="records")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records), encoding="utf-8")
        tmp.replace(path)
        return df
