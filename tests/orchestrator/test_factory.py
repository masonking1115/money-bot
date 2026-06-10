from datetime import datetime, timezone

import pandas as pd

from moneybot.config import Settings, TickerMeta, Universe
from moneybot.memory.models import MemoryContext
from moneybot.orchestrator.engine import Orchestrator
from moneybot.orchestrator.factory import build_orchestrator


class FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semis", benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA")],
        )

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame({"close": [100.0]})


class FakeRetriever:
    def retrieve(self, tickers, sector):
        return MemoryContext()


class FakeLLM:
    def complete_json(self, *, model, system, user, schema):
        return {}


def test_build_orchestrator_wires_everything(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path))
    orch = build_orchestrator(
        settings=settings,
        data_layer=FakeData(),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc),
    )
    assert isinstance(orch, Orchestrator)
    assert orch.research is not None and orch.analyst is not None
    assert orch.risk is not None and orch.execution is not None
    assert orch.journal is not None and orch.sod_equity is not None
    assert orch.strategy is not None
    assert orch.journal.path == tmp_path / "journal.jsonl"
    assert orch.sod_equity.path == tmp_path / "sod_equity.json"


def test_market_open_defaults_to_real_predicate(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path))
    orch = build_orchestrator(
        settings=settings, data_layer=FakeData(), retriever=FakeRetriever(),
        llm=FakeLLM(),
    )
    from datetime import datetime as dt
    from zoneinfo import ZoneInfo
    assert orch._market_open(dt(2026, 6, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))) is False


def test_build_orchestrator_uses_injected_research_and_analyst(tmp_path):
    # When research/analyst are supplied, the factory must use them verbatim
    # (not build its own from the LLM) — this is the backtest's injection seam.
    settings = Settings(mode="paper", data_dir=str(tmp_path))

    sentinel_research = object()
    sentinel_analyst = object()

    orch = build_orchestrator(
        settings=settings,
        data_layer=FakeData(),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        research=sentinel_research,
        analyst=sentinel_analyst,
    )

    assert orch.research is sentinel_research
    assert orch.analyst is sentinel_analyst
