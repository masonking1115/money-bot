from datetime import date

import pandas as pd
import pytest

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.data_layer import DataLayer


class StubPriceProvider:
    def __init__(self):
        self.calls = 0

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.calls += 1
        df = pd.DataFrame({
            "ts": pd.to_datetime(
                ["2026-06-07", "2026-06-08", "2026-06-09"], utc=True
            ),
            "open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
            "close": [10.0, 11.0, 12.0], "volume": [100, 200, 300],
        })
        if as_of is not None:
            df = df[df["ts"].dt.date <= as_of].reset_index(drop=True)
        return df


def _universe():
    return Universe(sector="semis", benchmark="SMH",
                    tickers=[{"symbol": "NVDA"}, {"symbol": "AMD"}])


def test_get_bars_returns_provider_data(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path))
    df = dl.get_bars("NVDA", "1d", 5)
    assert df["close"].tolist() == [10.0, 11.0, 12.0]


def test_rejects_ticker_outside_universe(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path))
    with pytest.raises(ValueError, match="not in universe"):
        dl.get_bars("TSLA", "1d", 5)


def test_caches_live_bars_and_avoids_refetch(tmp_path):
    provider = StubPriceProvider()
    dl = DataLayer(_universe(), provider, Cache(tmp_path))
    dl.get_bars("NVDA", "1d", 5)
    dl.get_bars("NVDA", "1d", 5)
    assert provider.calls == 1  # second call served from cache


def test_point_in_time_requests_are_not_cached_and_filter(tmp_path):
    provider = StubPriceProvider()
    dl = DataLayer(_universe(), provider, Cache(tmp_path))
    df = dl.get_bars("NVDA", "1d", 5, as_of=date(2026, 6, 8))
    assert df["ts"].dt.date.max() == date(2026, 6, 8)
    # as_of requests always re-fetch (backtest correctness over cache reuse)
    dl.get_bars("NVDA", "1d", 5, as_of=date(2026, 6, 8))
    assert provider.calls == 2
