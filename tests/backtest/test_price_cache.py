from datetime import date

import pandas as pd
import pytest

from moneybot.backtest.price_cache import CachingPriceProvider


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.calls += 1
        ts = pd.to_datetime([f"{as_of} 00:00:00+00:00"])
        return pd.DataFrame({"ts": ts, "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]})


def test_record_then_hit_does_not_recall_inner(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="record")
    a = cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    b = cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert inner.calls == 1                      # second call served from cache
    assert list(a["close"]) == list(b["close"])
    assert str(b["ts"].dt.date.iloc[0]) == "2024-03-01"  # ts round-trips as tz-aware


def test_cache_key_separates_args(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="record")
    cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 2))  # different as_of
    cache.get_bars("AMD", "1d", 20, as_of=date(2024, 3, 1))   # different ticker
    assert inner.calls == 3


def test_replay_miss_raises(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="replay")
    with pytest.raises(RuntimeError, match="cache miss"):
        cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert inner.calls == 0


def test_persists_across_instances(tmp_path):
    CachingPriceProvider(CountingProvider(), root=tmp_path, mode="record").get_bars(
        "NVDA", "1d", 20, as_of=date(2024, 3, 1)
    )
    reopened = CachingPriceProvider(CountingProvider(), root=tmp_path, mode="replay")
    df = reopened.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert df["close"].iloc[0] == 1.5  # read back from disk, inner never called
