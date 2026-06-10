"""A mutable clock for replay. The harness advances it to each simulated day;
the orchestrator reads it for cycle_id + market-hours gating. The fixed 10:00 ET
intraday time keeps cycle_id unique per day and inside normal market hours."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_CYCLE_TIME = time(10, 0)


class SimClock:
    def __init__(self) -> None:
        self._day: date | None = None

    def set_day(self, day: date) -> None:
        self._day = day

    def __call__(self) -> datetime:
        if self._day is None:
            raise RuntimeError("SimClock used before set_day()")
        return datetime.combine(self._day, _CYCLE_TIME, tzinfo=_ET)
