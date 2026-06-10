from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from moneybot.execution.models import (
    AccountSnapshot,
    Discrepancy,
    Fill,
    OrderRequest,
    PositionRecord,
    ReconciliationResult,
)


def test_order_request_defaults_to_market():
    o = OrderRequest(
        client_order_id="c1:NVDA:buy", ticker="NVDA", side="buy", quantity=10
    )
    assert o.order_type == "market"
    assert o.reference_price is None


def test_order_request_quantity_must_be_positive():
    with pytest.raises(ValidationError):
        OrderRequest(client_order_id="c1", ticker="NVDA", side="buy", quantity=0)


def test_order_request_rejects_unknown_side():
    with pytest.raises(ValidationError):
        OrderRequest(client_order_id="c1", ticker="NVDA", side="hodl", quantity=1)


def test_fill_construction():
    f = Fill(
        client_order_id="c1:NVDA:buy",
        broker_order_id="paper-1",
        ticker="NVDA",
        side="buy",
        status="filled",
        filled_qty=10,
        avg_price=100.0,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    assert f.status == "filled"
    assert f.reason == ""


def test_position_record_signed_qty():
    long = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    short = PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)
    assert long.qty > 0 and short.qty < 0


def test_account_snapshot():
    a = AccountSnapshot(equity=100_000.0, cash=40_000.0)
    assert a.equity == 100_000.0


def test_reconciliation_result_in_sync():
    r = ReconciliationResult(in_sync=True, discrepancies=[])
    assert r.in_sync and r.discrepancies == []


def test_discrepancy_fields():
    d = Discrepancy(ticker="NVDA", stored_qty=10.0, broker_qty=8.0)
    assert d.ticker == "NVDA"
