from datetime import datetime, timezone

from moneybot.execution.alpaca import AlpacaBroker
from moneybot.execution.models import OrderRequest


def _clock():
    return datetime(2026, 6, 9, tzinfo=timezone.utc)


def _broker():
    return AlpacaBroker(key_id="k", secret_key="s", paper=True, clock=_clock)


def test_buy_maps_to_alpaca_buy_and_parses_fill(monkeypatch):
    b = _broker()
    captured = {}

    def fake_submit(symbol, qty, side, client_order_id):
        captured.update(symbol=symbol, qty=qty, side=side, client_order_id=client_order_id)
        return {
            "id": "abc-123",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "101.5",
        }

    monkeypatch.setattr(b, "_submit_raw", fake_submit)
    order = OrderRequest(
        client_order_id="c1:NVDA:buy",
        ticker="NVDA",
        side="buy",
        quantity=10,
        reference_price=100.0,
    )
    fill = b.place_order(order)
    assert captured == {
        "symbol": "NVDA",
        "qty": 10,
        "side": "buy",
        "client_order_id": "c1:NVDA:buy",
    }
    assert fill.status == "filled"
    assert fill.broker_order_id == "abc-123"
    assert fill.filled_qty == 10 and fill.avg_price == 101.5
    assert fill.ts == _clock()


def test_short_maps_to_alpaca_sell(monkeypatch):
    b = _broker()
    captured = {}

    def fake_submit(symbol, qty, side, client_order_id):
        captured["side"] = side
        return {"id": "x", "status": "accepted", "filled_qty": "0", "filled_avg_price": None}

    monkeypatch.setattr(b, "_submit_raw", fake_submit)
    order = OrderRequest(
        client_order_id="c2:SMH:short",
        ticker="SMH",
        side="short",
        quantity=5,
        reference_price=200.0,
    )
    fill = b.place_order(order)
    assert captured["side"] == "sell"
    assert fill.status == "accepted"
    assert fill.filled_qty == 0 and fill.avg_price == 0.0


def test_rejected_status_maps_through(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_submit_raw",
        lambda symbol, qty, side, client_order_id: {
            "id": "x",
            "status": "rejected",
            "filled_qty": "0",
            "filled_avg_price": None,
        },
    )
    order = OrderRequest(
        client_order_id="c3", ticker="NVDA", side="buy", quantity=1, reference_price=10.0
    )
    assert b.place_order(order).status == "rejected"


def test_get_positions_parses_signed_qty(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_positions_raw",
        lambda: [
            {"symbol": "NVDA", "qty": "10", "avg_entry_price": "100.0"},
            {"symbol": "SMH", "qty": "-5", "avg_entry_price": "200.0"},
        ],
    )
    positions = {p.ticker: p for p in b.get_positions()}
    assert positions["NVDA"].qty == 10.0
    assert positions["SMH"].qty == -5.0 and positions["SMH"].avg_price == 200.0


def test_get_account_parses_equity_and_cash(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b, "_account_raw", lambda: {"equity": "123456.78", "cash": "5000.00"}
    )
    acct = b.get_account()
    assert acct.equity == 123456.78 and acct.cash == 5000.0


def test_construction_does_not_touch_sdk():
    # Building the adapter must not import the SDK or open a connection — the
    # client is created lazily only inside the _*_raw methods (never called here).
    b = AlpacaBroker(key_id="k", secret_key="s", paper=True)
    assert b._client is None
