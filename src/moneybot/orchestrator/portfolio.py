"""Build the Risk Engine's PortfolioState from the broker, marked to market.

SodEquityStore remembers the first equity seen each trading day so the Risk
Engine's daily-loss circuit breaker has a day_pnl_pct to read. build_portfolio_state
translates broker positions + account into a PortfolioState, marking each holding
to its current price via the data layer (falling back to cost when a price is
missing, and never asking the data layer about a non-universe ticker).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneybot.data_layer import DataLayer


class SodEquityStore:
    """Anchors start-of-day equity (JSON) to compute intraday P&L percentage."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "sod_equity.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def day_pnl_pct(self, equity: float, today: date) -> float:
        anchor = self._read(today)
        if anchor is None:
            self._write(today, equity)
            return 0.0
        if anchor <= 0:
            return 0.0
        return (equity - anchor) / anchor

    def _read(self, today: date) -> float | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("date") != today.isoformat():
            return None
        return float(data["equity"])

    def _write(self, today: date, equity: float) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"date": today.isoformat(), "equity": equity}), encoding="utf-8"
        )
        tmp.replace(self.path)


def last_finite_close(closes: list) -> float | None:
    return next((c for c in reversed(closes) if c is not None and math.isfinite(c)), None)


def mark_price(
    *,
    data_layer: DataLayer,
    ticker: str,
    timeframe: str,
    lookback: int,
    as_of: date | None,
) -> float | None:
    """Most recent finite close for a ticker, or None. Caller must ensure the
    ticker is in the universe (or is the benchmark) before calling."""
    bars = data_layer.get_bars(ticker, timeframe, lookback, as_of=as_of)
    closes = [] if bars.empty else bars["close"].tolist()
    return last_finite_close(closes)
