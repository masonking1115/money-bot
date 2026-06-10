import pytest
from pydantic import ValidationError

from moneybot.orchestrator.models import CycleResult, ExitSignal


def test_exit_signal_fields():
    s = ExitSignal(ticker="NVDA", shares=10, reason="stop_loss", reference_price=90.0)
    assert s.ticker == "NVDA" and s.shares == 10 and s.reason == "stop_loss"


def test_exit_signal_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        ExitSignal(ticker="NVDA", shares=1, reason="vibes", reference_price=1.0)


def test_exit_signal_shares_must_be_positive():
    with pytest.raises(ValidationError):
        ExitSignal(ticker="NVDA", shares=0, reason="time_stop", reference_price=1.0)


def test_cycle_result_defaults():
    r = CycleResult(status="completed", cycle_id="2026-06-09T10")
    assert r.reason == ""
    assert r.entry_fills == [] and r.exit_fills == []
    assert r.plans_proposed == 0
    assert r.halted_by_risk is False
    assert r.reconciliation is None


def test_cycle_result_status_validated():
    with pytest.raises(ValidationError):
        CycleResult(status="exploded")
