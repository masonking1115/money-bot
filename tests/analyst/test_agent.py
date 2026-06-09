import pandas as pd
import pytest

from moneybot.analyst.agent import AnalystAgent
from moneybot.config import Settings, TickerMeta, Universe
from moneybot.memory.models import MemoryContext


class ScriptedLLM:
    """A fake LLMClient: returns queued responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def complete_json(self, *, model, system, user, schema):
        self.requests.append(
            {"model": model, "system": system, "user": user, "schema": schema}
        )
        return self._responses.pop(0)


def _bars(closes):
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [f"2026-06-{i + 1:02d}" for i in range(len(closes))], utc=True
            ),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * len(closes),
        }
    )


class FakeData:
    """Minimal DataLayer stand-in: serves canned bars and exposes a universe."""

    def __init__(self, bars_by_ticker):
        self._bars = bars_by_ticker
        self.universe = Universe(
            sector="semiconductors",
            benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
        )
        self.bars_calls = []

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.bars_calls.append((ticker, timeframe, lookback, as_of))
        empty = pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]
        )
        return self._bars.get(ticker, empty)


def _settings():
    return Settings(model_analyst="claude-opus-4-8", rs_lookback_days=20, rs_timeframe="1d")


def test_relative_strength_computes_excess_return_per_universe_name():
    data = FakeData(
        {
            "SMH": _bars([100.0, 110.0]),   # benchmark +10%
            "NVDA": _bars([100.0, 130.0]),  # +30% -> excess +20%
            "AMD": _bars([100.0, 105.0]),   # +5%  -> excess -5%
        }
    )
    agent = AnalystAgent(
        data_layer=data, strategy=object(), llm=ScriptedLLM([]), settings=_settings()
    )
    rs = agent._relative_strength()
    assert rs["NVDA"] == pytest.approx(0.20)
    assert rs["AMD"] == pytest.approx(-0.05)
    assert set(rs) == {"NVDA", "AMD"}  # benchmark itself is not a candidate


def test_relative_strength_threads_as_of_into_get_bars():
    import datetime

    data = FakeData({"SMH": _bars([100.0, 110.0]), "NVDA": _bars([100.0, 120.0])})
    agent = AnalystAgent(
        data_layer=data, strategy=object(), llm=ScriptedLLM([]), settings=_settings()
    )
    as_of = datetime.date(2026, 6, 5)
    agent._relative_strength(as_of=as_of)
    # every get_bars call carried the as_of and the configured timeframe/lookback
    assert all(c[3] == as_of for c in data.bars_calls)
    assert all(c[1] == "1d" and c[2] == 20 for c in data.bars_calls)


def test_memory_context_empty_when_no_retriever():
    data = FakeData({})
    agent = AnalystAgent(
        data_layer=data, strategy=object(), llm=ScriptedLLM([]), settings=_settings()
    )
    ctx = agent._memory_context("NVDA")
    assert isinstance(ctx, MemoryContext)
    assert ctx.dossiers == [] and ctx.lessons == []


def test_memory_context_uses_retriever_with_ticker_and_sector():
    data = FakeData({})

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, tickers, sector):
            self.calls.append((tickers, sector))
            return MemoryContext()

    retriever = FakeRetriever()
    agent = AnalystAgent(
        data_layer=data,
        strategy=object(),
        llm=ScriptedLLM([]),
        settings=_settings(),
        retriever=retriever,
    )
    agent._memory_context("NVDA")
    assert retriever.calls == [(["NVDA"], "semiconductors")]
