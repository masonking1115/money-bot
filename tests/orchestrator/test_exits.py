from datetime import date

from moneybot.execution.models import PositionRecord
from moneybot.orchestrator.exits import evaluate_exits
from moneybot.strategies.models import ExitPlan


def _plan(stop=0.08, target=0.20, max_hold=10):
    return ExitPlan(
        max_hold_days=max_hold,
        stop_loss_pct=stop,
        profit_target_pct=target,
        thesis_check_guidance="n/a",
    )


def _long(ticker="NVDA", qty=10.0, avg=100.0):
    return PositionRecord(ticker=ticker, qty=qty, avg_price=avg)


AS_OF = date(2026, 6, 20)
ENTRY = date(2026, 6, 18)  # 2 days before AS_OF


def test_no_trigger_in_band():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 105.0},  # +5%: below +20% target, above -8% stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_stop_loss_triggers():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 92.0},  # -8% exactly -> stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert len(sigs) == 1
    assert sigs[0].reason == "stop_loss" and sigs[0].shares == 10
    assert sigs[0].reference_price == 92.0


def test_profit_target_triggers():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 120.0},  # +20% -> target
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "profit_target"


def test_time_stop_triggers_when_in_band_but_held_too_long():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": date(2026, 6, 1)},  # 19 days before AS_OF >= 10
        current_prices={"NVDA": 105.0},  # in band
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "time_stop"


def test_stop_loss_takes_precedence_over_time_stop():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": date(2026, 6, 1)},  # also past max hold
        current_prices={"NVDA": 80.0},  # also below stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "stop_loss"


def test_unknown_entry_date_skips_time_stop_only():
    # in band, no entry date -> cannot time-stop, so no signal
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={},
        current_prices={"NVDA": 105.0},
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_missing_price_skips_position():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={},  # no mark available
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_shorts_are_ignored():
    sigs = evaluate_exits(
        positions=[PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)],
        entry_dates={"SMH": ENTRY},
        current_prices={"SMH": 100.0},
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []  # phase-1 exit loop manages longs only
