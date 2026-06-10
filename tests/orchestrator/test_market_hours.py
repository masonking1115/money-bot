from datetime import datetime
from zoneinfo import ZoneInfo

from moneybot.orchestrator.market_hours import is_market_open

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_open_midday_weekday():
    # Wednesday 2026-06-10, 10:00 ET
    assert is_market_open(datetime(2026, 6, 10, 10, 0, tzinfo=ET)) is True


def test_closed_before_open():
    assert is_market_open(datetime(2026, 6, 10, 9, 0, tzinfo=ET)) is False


def test_open_at_930():
    assert is_market_open(datetime(2026, 6, 10, 9, 30, tzinfo=ET)) is True


def test_closed_after_4pm():
    assert is_market_open(datetime(2026, 6, 10, 16, 1, tzinfo=ET)) is False


def test_closed_on_saturday():
    # Saturday 2026-06-13, midday
    assert is_market_open(datetime(2026, 6, 13, 12, 0, tzinfo=ET)) is False


def test_naive_or_utc_is_converted():
    # 14:00 UTC on a weekday == 10:00 ET (summer, EDT) -> open
    assert is_market_open(datetime(2026, 6, 10, 14, 0, tzinfo=UTC)) is True
