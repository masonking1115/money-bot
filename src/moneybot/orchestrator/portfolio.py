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

from moneybot.risk.models import PortfolioState, Position

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.execution.broker import Broker


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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date") != today.isoformat():
                return None
            return float(data["equity"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Corrupt/partial file -> treat as no anchor; the next write re-anchors.
            # A missing anchor yields 0% day P&L, which safely won't trip the breaker.
            return None

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


def build_portfolio_state(
    *,
    broker: Broker,
    data_layer: DataLayer,
    settings: Settings,
    as_of: date | None,
    day_pnl_pct: float,
) -> PortfolioState:
    """Translate the broker snapshot into the Risk Engine's PortfolioState.

    Each holding is marked to its current price (cost-basis fallback when a price
    is missing or the ticker is not markable). Equity comes from the broker; if the
    broker reports a non-positive equity, fall back to cash + marked value so the
    PortfolioState's gt=0 constraint holds.
    """
    account = broker.get_account()
    universe = data_layer.universe
    markable = set(universe.symbols) | {universe.benchmark}

    positions: list[Position] = []
    for rec in broker.get_positions():
        price: float | None = None
        if rec.ticker in markable:
            price = mark_price(
                data_layer=data_layer,
                ticker=rec.ticker,
                timeframe=settings.risk_timeframe,
                lookback=settings.risk_lookback_days,
                as_of=as_of,
            )
        if price is None:
            price = rec.avg_price  # mark-to-cost fallback
        positions.append(
            Position(ticker=rec.ticker, shares=rec.qty, market_value=rec.qty * price)
        )

    equity = account.equity
    if equity <= 0:
        equity = account.cash + sum(p.market_value for p in positions)
    if equity <= 0:
        # A non-positive account cannot be sized against; fail loudly rather than
        # raising an opaque ValidationError inside PortfolioState (or trading on a lie).
        raise ValueError(
            f"cannot build PortfolioState: equity is non-positive ({equity}); "
            "broker reported non-positive equity and cash + marked positions is also <= 0"
        )

    return PortfolioState(
        equity=equity, cash=account.cash, positions=positions, day_pnl_pct=day_pnl_pct
    )
