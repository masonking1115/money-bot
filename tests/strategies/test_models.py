import pytest
from pydantic import ValidationError

from moneybot.strategies.models import (
    CatalystSignal,
    Evidence,
    ExitPlan,
    Proposal,
    StrategyParams,
)


def _signal(**over):
    base = dict(
        ticker="NVDA", category="guidance", direction="bullish",
        materiality=0.8, freshness_days=1, conviction=0.7,
        evidence=[Evidence(source="edgar", quote="raised FY guide", url="https://x/1")],
        thesis="guidance raised",
    )
    base.update(over)
    return CatalystSignal(**base)


def test_catalyst_signal_roundtrips():
    s = _signal()
    assert CatalystSignal.model_validate_json(s.model_dump_json()) == s
    assert s.signal_id is None


def test_catalyst_signal_rejects_out_of_range_conviction():
    with pytest.raises(ValidationError):
        _signal(conviction=1.5)


def test_proposal_defaults():
    p = Proposal(ticker="NVDA", action="buy", conviction=0.7, thesis="t", score=0.5)
    assert p.signal_ref is None


def test_strategy_params_defaults_match_spec():
    p = StrategyParams()
    assert p.freshness_window_days == 5
    assert p.max_hold_days == 10
    assert p.stop_loss_pct == 0.08
    assert p.profit_target_pct == 0.20
    assert p.hedge_enabled is False


def test_exit_plan_fields():
    e = ExitPlan(max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
                 thesis_check_guidance="check guidance held")
    assert e.max_hold_days == 10
