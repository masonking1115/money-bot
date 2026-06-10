from datetime import date

import pandas as pd

from moneybot.backtest.calendar import trading_days_from_bars


def _bars(days):
    ts = pd.to_datetime([f"{d} 00:00:00+00:00" for d in days])
    return pd.DataFrame({"ts": ts, "close": [1.0] * len(days)})


def test_filters_to_range_and_sorts():
    bars = _bars(["2024-01-02", "2024-01-05", "2024-01-03", "2023-12-29", "2024-02-01"])
    out = trading_days_from_bars(bars, date(2024, 1, 1), date(2024, 1, 31))
    assert out == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]


def test_empty_bars_yields_no_days():
    assert trading_days_from_bars(pd.DataFrame({"ts": [], "close": []}), date(2024, 1, 1), date(2024, 1, 31)) == []
