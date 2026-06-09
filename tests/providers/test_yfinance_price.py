from datetime import date, datetime, timezone

import pandas as pd

from moneybot.providers import PriceProvider
from moneybot.providers.yfinance_price import YFinancePriceProvider


def _fake_yf_frame():
    idx = pd.DatetimeIndex(
        [datetime(2026, 6, 7, tzinfo=timezone.utc),
         datetime(2026, 6, 8, tzinfo=timezone.utc),
         datetime(2026, 6, 9, tzinfo=timezone.utc)],
        name="Datetime",
    )
    return pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
         "Close": [10.0, 11.0, 12.0], "Volume": [100, 200, 300]},
        index=idx,
    )


def test_satisfies_protocol():
    assert isinstance(YFinancePriceProvider(), PriceProvider)


def test_get_bars_normalizes_columns(monkeypatch):
    prov = YFinancePriceProvider()
    monkeypatch.setattr(prov, "_download", lambda *a, **k: _fake_yf_frame())
    df = prov.get_bars("NVDA", timeframe="1d", lookback=5)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.0, 11.0, 12.0]
    assert df["ts"].is_monotonic_increasing


def test_as_of_filters_future_bars(monkeypatch):
    prov = YFinancePriceProvider()
    monkeypatch.setattr(prov, "_download", lambda *a, **k: _fake_yf_frame())
    df = prov.get_bars("NVDA", timeframe="1d", lookback=5, as_of=date(2026, 6, 8))
    assert df["ts"].dt.date.max() == date(2026, 6, 8)
    assert len(df) == 2


# Fix C — Robust ts column when the index has no name
def test_get_bars_handles_unnamed_index(monkeypatch):
    idx = pd.DatetimeIndex(
        [datetime(2026, 6, 8, tzinfo=timezone.utc), datetime(2026, 6, 9, tzinfo=timezone.utc)],
        name=None,
    )
    frame = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [10.0, 11.0], "Volume": [1, 2]},
        index=idx,
    )
    prov = YFinancePriceProvider()
    monkeypatch.setattr(prov, "_download", lambda *a, **k: frame)
    df = prov.get_bars("NVDA", timeframe="1d", lookback=5)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.0, 11.0]
