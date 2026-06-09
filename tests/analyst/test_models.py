import pytest
from pydantic import ValidationError

from moneybot.analyst.models import ConfirmationVerdict, TradePlan
from moneybot.strategies.models import ExitPlan


def _exit_plan():
    return ExitPlan(
        max_hold_days=10,
        stop_loss_pct=0.08,
        profit_target_pct=0.20,
        thesis_check_guidance="re-read the name",
    )


def test_confirmation_verdict_defaults_risk_flags_to_empty():
    v = ConfirmationVerdict(confirmed=True, adjusted_conviction=0.7, reasoning="ok")
    assert v.risk_flags == []


def test_confirmation_verdict_rejects_out_of_range_conviction():
    with pytest.raises(ValidationError):
        ConfirmationVerdict(confirmed=True, adjusted_conviction=1.5, reasoning="bad")


def test_trade_plan_carries_exit_plan_and_signal_ref():
    plan = TradePlan(
        ticker="NVDA",
        action="buy",
        conviction=0.8,
        thesis="guidance raised",
        score=0.42,
        signal_ref="abc123",
        exit_plan=_exit_plan(),
        analyst_note="confirmed: demand durable",
    )
    assert plan.action == "buy"
    assert plan.exit_plan.max_hold_days == 10
    assert plan.signal_ref == "abc123"
    assert plan.risk_flags == []


def test_trade_plan_action_must_be_buy():
    with pytest.raises(ValidationError):
        TradePlan(
            ticker="NVDA",
            action="sell",  # long-only strategy: only "buy" is valid
            conviction=0.8,
            thesis="t",
            score=0.1,
            signal_ref=None,
            exit_plan=_exit_plan(),
            analyst_note="n",
        )
