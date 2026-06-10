from datetime import date, datetime
from zoneinfo import ZoneInfo

from moneybot.backtest.clock import SimClock

ET = ZoneInfo("America/New_York")


def test_returns_fixed_intraday_time_for_set_day():
    clock = SimClock()
    clock.set_day(date(2024, 3, 1))
    now = clock()
    assert now == datetime(2024, 3, 1, 10, 0, tzinfo=ET)
    # cycle_id derivation (as the orchestrator does it) is unique per day
    assert now.strftime("%Y-%m-%dT%H") == "2024-03-01T10"


def test_advancing_changes_the_returned_day():
    clock = SimClock()
    clock.set_day(date(2024, 3, 1))
    clock.set_day(date(2024, 3, 4))
    assert clock().date() == date(2024, 3, 4)


def test_call_before_set_raises():
    import pytest

    with pytest.raises(RuntimeError):
        SimClock()()
