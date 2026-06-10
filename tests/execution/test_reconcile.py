from moneybot.execution.models import PositionRecord
from moneybot.execution.reconcile import reconcile


def _p(ticker, qty):
    return PositionRecord(ticker=ticker, qty=qty, avg_price=100.0)


def test_in_sync_when_quantities_match():
    stored = [_p("NVDA", 10.0), _p("AMD", 5.0)]
    broker = [_p("AMD", 5.0), _p("NVDA", 10.0)]  # order-independent
    result = reconcile(stored, broker)
    assert result.in_sync
    assert result.discrepancies == []


def test_quantity_mismatch_is_a_discrepancy():
    result = reconcile([_p("NVDA", 10.0)], [_p("NVDA", 8.0)])
    assert not result.in_sync
    assert len(result.discrepancies) == 1
    d = result.discrepancies[0]
    assert d.ticker == "NVDA" and d.stored_qty == 10.0 and d.broker_qty == 8.0


def test_position_missing_at_broker():
    result = reconcile([_p("NVDA", 10.0)], [])
    assert not result.in_sync
    assert result.discrepancies[0].broker_qty == 0.0


def test_unexpected_position_at_broker():
    result = reconcile([], [_p("NVDA", 4.0)])
    assert not result.in_sync
    assert result.discrepancies[0].stored_qty == 0.0


def test_tiny_float_drift_is_tolerated():
    result = reconcile([_p("NVDA", 10.0)], [_p("NVDA", 10.0 + 1e-9)])
    assert result.in_sync


def test_discrepancies_sorted_by_ticker():
    result = reconcile([_p("NVDA", 1.0), _p("AMD", 1.0)], [])
    assert [d.ticker for d in result.discrepancies] == ["AMD", "NVDA"]
