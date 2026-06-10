from datetime import date

import pandas as pd

from moneybot.backtest.engine import run_backtest
from moneybot.backtest.models import BacktestConfig
from moneybot.cache import Cache
from moneybot.config import Settings, TickerMeta, Universe
from moneybot.data_layer import DataLayer
from moneybot.memory.models import MemoryContext


class StubPrices:
    """Deterministic rising price for NVDA; flat for SMH. Honors as_of (point-in-time)."""

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        base = {"NVDA": 100.0, "SMH": 200.0}.get(ticker, 50.0)
        # a single bar dated at as_of (enough for marking; rising via day number)
        bump = as_of.day if as_of else 1
        price = base + (bump if ticker == "NVDA" else 0)
        ts = pd.to_datetime([f"{as_of} 00:00:00+00:00"])
        return pd.DataFrame(
            {"ts": ts, "open": [price], "high": [price], "low": [price], "close": [price], "volume": [1_000_000]}
        )


class StubFilings:
    def get_recent_filings(self, ticker, types=None, since=None, as_of=None):
        return []


class StubNews:
    def get_news(self, query, since=None, as_of=None):
        return []


class StubRetriever:
    def retrieve(self, tickers, sector):
        return MemoryContext()


class FakeLLM:
    """Never produces a confirmed plan -> no entries; exercises the full no-trade path."""

    def complete_json(self, *, model, system, user, schema):
        return {}


def _universe():
    return Universe(
        sector="semiconductors", benchmark="SMH",
        tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
    )


def _data_layer(tmp_path):
    return DataLayer(
        _universe(), StubPrices(), Cache(str(tmp_path / "cache")),
        filings_provider=StubFilings(), news_provider=StubNews(),
    )


def _settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "run"), paper_starting_cash=100_000.0)


def test_run_backtest_produces_curve_and_metrics(tmp_path):
    # Benchmark bars define the trading calendar; supply 3 weekdays.
    bench = pd.DataFrame({
        "ts": pd.to_datetime(["2024-03-04 00:00:00+00:00", "2024-03-05 00:00:00+00:00", "2024-03-06 00:00:00+00:00"]),
        "close": [200.0, 200.0, 200.0],
    })

    report = run_backtest(
        settings=_settings(tmp_path),
        data_layer=_data_layer(tmp_path),
        llm=FakeLLM(),
        retriever=StubRetriever(),
        config=BacktestConfig(start=date(2024, 3, 4), end=date(2024, 3, 6)),
        cache_root=tmp_path / "bt_cache",
        benchmark_bars=bench,
    )

    assert len(report.equity_curve) == 3
    assert report.equity_curve[0].day == date(2024, 3, 4)
    # No confirmed plans -> no trades, equity stays at starting cash.
    assert report.metrics.n_trades == 0
    assert report.metrics.final_equity == 100_000.0
    assert report.metrics.benchmark_return == 0.0
    assert any("daily-loss breaker" in n.lower() for n in report.notes)


def test_run_backtest_no_trading_days_is_safe(tmp_path):
    empty = pd.DataFrame({"ts": [], "close": []})
    report = run_backtest(
        settings=_settings(tmp_path),
        data_layer=_data_layer(tmp_path),
        llm=FakeLLM(),
        retriever=StubRetriever(),
        config=BacktestConfig(start=date(2024, 3, 4), end=date(2024, 3, 6)),
        cache_root=tmp_path / "bt_cache",
        benchmark_bars=empty,
    )
    assert report.equity_curve == []
    assert report.metrics.final_equity == 100_000.0
