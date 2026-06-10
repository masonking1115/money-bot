from datetime import date

import pytest

from moneybot.analyst.models import TradePlan
from moneybot.backtest.agent_cache import CachingAnalyst, CachingResearch
from moneybot.strategies.models import CatalystSignal, Evidence, ExitPlan


def _signal(ticker):
    return CatalystSignal(
        ticker=ticker, category="demand", direction="bullish", materiality=0.8,
        freshness_days=1, conviction=0.7,
        evidence=[Evidence(source="8-K", quote="q", url="http://x")], thesis="t",
        signal_id="sig1",
    )


def _plan(ticker):
    return TradePlan(
        ticker=ticker, action="buy", conviction=0.6, thesis="t", score=1.0,
        signal_ref="sig1",
        exit_plan=ExitPlan(max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.2, thesis_check_guidance="n/a"),
        analyst_note="ok",
    )


class FakeResearch:
    def __init__(self):
        self.calls = 0

    def research_universe(self, as_of=None):
        self.calls += 1
        return {"NVDA": [_signal("NVDA")]}


class FakeAnalyst:
    def __init__(self):
        self.calls = 0

    def analyze(self, research, as_of=None):
        self.calls += 1
        return [_plan("NVDA")]


def test_research_records_then_replays_without_recalling(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    a = cache.research_universe(as_of=date(2024, 3, 1))
    b = cache.research_universe(as_of=date(2024, 3, 1))
    assert inner.calls == 1
    assert isinstance(b["NVDA"][0], CatalystSignal)
    assert b["NVDA"][0].ticker == "NVDA"
    assert a["NVDA"][0].signal_id == "sig1"


def test_analyst_records_then_replays_without_recalling(tmp_path):
    inner = FakeAnalyst()
    cache = CachingAnalyst(inner, root=tmp_path, mode="record")
    cache.analyze({"NVDA": [_signal("NVDA")]}, as_of=date(2024, 3, 1))
    plans = cache.analyze({"NVDA": [_signal("NVDA")]}, as_of=date(2024, 3, 1))
    assert inner.calls == 1
    assert isinstance(plans[0], TradePlan)
    assert plans[0].ticker == "NVDA"


def test_distinct_days_recompute(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    cache.research_universe(as_of=date(2024, 3, 1))
    cache.research_universe(as_of=date(2024, 3, 2))
    assert inner.calls == 2


def test_replay_miss_raises(tmp_path):
    cache = CachingAnalyst(FakeAnalyst(), root=tmp_path, mode="replay")
    with pytest.raises(RuntimeError, match="cache miss"):
        cache.analyze({}, as_of=date(2024, 3, 1))


def test_research_persists_across_instances(tmp_path):
    CachingResearch(FakeResearch(), root=tmp_path, mode="record").research_universe(as_of=date(2024, 3, 1))
    inner = FakeResearch()
    reopened = CachingResearch(inner, root=tmp_path, mode="replay")
    out = reopened.research_universe(as_of=date(2024, 3, 1))
    assert inner.calls == 0
    assert out["NVDA"][0].ticker == "NVDA"


def test_as_of_none_uses_stable_key(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    cache.research_universe(as_of=None)
    cache.research_universe(as_of=None)
    assert inner.calls == 1
