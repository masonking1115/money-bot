from datetime import datetime, timezone

from moneybot.execution.models import OrderRequest
from moneybot.execution.paper import PaperBroker


def _clock():
    return datetime(2026, 6, 9, tzinfo=timezone.utc)


def _broker(cash=100_000.0):
    return PaperBroker(starting_cash=cash, clock=_clock)


def _buy(ticker="NVDA", qty=10, price=100.0, oid="c1:NVDA:buy"):
    return OrderRequest(
        client_order_id=oid, ticker=ticker, side="buy", quantity=qty, reference_price=price
    )


def test_buy_fills_and_decrements_cash():
    b = _broker()
    fill = b.place_order(_buy())
    assert fill.status == "filled"
    assert fill.filled_qty == 10 and fill.avg_price == 100.0
    assert fill.broker_order_id == "paper-1"
    assert fill.ts == _clock()
    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "NVDA" and positions[0].qty == 10.0
    acct = b.get_account()
    assert acct.cash == 100_000.0 - 1_000.0
    assert acct.equity == 100_000.0  # mark-to-cost: cash + qty*avg_price


def test_missing_reference_price_is_rejected():
    b = _broker()
    order = OrderRequest(client_order_id="c2", ticker="NVDA", side="buy", quantity=5)
    fill = b.place_order(order)
    assert fill.status == "rejected"
    assert fill.filled_qty == 0
    assert "reference price" in fill.reason
    assert b.get_positions() == []


def test_idempotent_on_client_order_id():
    b = _broker()
    first = b.place_order(_buy())
    second = b.place_order(_buy())  # same client_order_id
    assert second.broker_order_id == first.broker_order_id
    assert b.get_account().cash == 100_000.0 - 1_000.0  # applied once only
    assert b.get_positions()[0].qty == 10.0


def test_short_adds_cash_and_makes_negative_position():
    b = _broker()
    order = OrderRequest(
        client_order_id="c3:SMH:short",
        ticker="SMH",
        side="short",
        quantity=5,
        reference_price=200.0,
    )
    fill = b.place_order(order)
    assert fill.status == "filled"
    assert b.get_account().cash == 100_000.0 + 1_000.0
    pos = b.get_positions()[0]
    assert pos.ticker == "SMH" and pos.qty == -5.0


def test_flat_positions_excluded():
    b = _broker()
    b.place_order(_buy(qty=10, oid="o-buy"))
    sell = OrderRequest(
        client_order_id="o-sell",
        ticker="NVDA",
        side="sell",
        quantity=10,
        reference_price=110.0,
    )
    b.place_order(sell)
    assert b.get_positions() == []


# H2 regression: order reason propagates to filled Fill ----------------------

def test_filled_fill_carries_order_reason():
    """H2: OrderRequest.reason is echoed in the filled Fill.reason."""
    b = _broker()
    order = OrderRequest(
        client_order_id="c:NVDA:exit",
        ticker="NVDA",
        side="sell",
        quantity=5,
        reference_price=90.0,
        reason="stop_loss",
    )
    fill = b.place_order(order)
    assert fill.status == "filled"
    assert fill.reason == "stop_loss"


def test_rejection_reason_not_overridden_by_order_reason():
    """H2: rejection fills keep 'no reference price' reason, not order.reason."""
    b = _broker()
    order = OrderRequest(
        client_order_id="c:NVDA:exit",
        ticker="NVDA",
        side="sell",
        quantity=5,
        reason="stop_loss",
    )
    fill = b.place_order(order)
    assert fill.status == "rejected"
    assert "reference price" in fill.reason  # rejection reason preserved
