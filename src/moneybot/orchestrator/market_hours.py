"""A simple US-equity market-hours check.

Phase-1: regular session only — Mon-Fri, 09:30-16:00 America/New_York. It does
NOT know about market holidays or half-days; that (or a broker-clock check) can
replace this predicate later — the orchestrator takes it as an injectable, so
nothing else changes. Naive datetimes are assumed UTC.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)


def is_market_open(now: datetime) -> bool:
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC"))
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _OPEN <= et.time() <= _CLOSE
