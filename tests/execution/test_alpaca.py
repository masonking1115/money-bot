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


def _order(oid="c", side="buy"):
    return OrderRequest(
        client_order_id=oid, ticker="NVDA", side=side, quantity=2, reference_price=10.0
    )


def test_canceled_and_expired_map_to_rejected(monkeypatch):
    b = _broker()
    for raw_status in ("canceled", "expired"):
        monkeypatch.setattr(
            b,
            "_submit_raw",
            lambda symbol, qty, side, client_order_id, s=raw_status: {
                "id": "x",
                "status": s,
                "filled_qty": "0",
                "filled_avg_price": None,
            },
        )
        fill = b.place_order(_order())
        assert fill.status == "rejected" and fill.reason == raw_status


def test_duplicate_client_order_id_returns_existing_order(monkeypatch):
    # A re-run trips Alpaca's duplicate-client_order_id rejection; the broker must
    # fall back to the already-placed order and return it (idempotent, no re-trade).
    from moneybot.execution.alpaca import _DuplicateClientOrderId

    b = _broker()
    submit_calls = []

    def fake_submit(symbol, qty, side, client_order_id):
        submit_calls.append(client_order_id)
        raise _DuplicateClientOrderId(client_order_id)

    def fake_fetch(client_order_id):
        return {
            "id": "existing-1",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "100.0",
        }

    monkeypatch.setattr(b, "_submit_raw", fake_submit)
    monkeypatch.setattr(b, "_get_by_client_id_raw", fake_fetch)
    fill = b.place_order(_order(oid="c1:NVDA:buy"))
    assert fill.broker_order_id == "existing-1"
    assert fill.status == "filled" and fill.filled_qty == 10
    assert submit_calls == ["c1:NVDA:buy"]  # submitted once, then fell back to fetch


def test_partially_filled_maps_to_accepted(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_submit_raw",
        lambda symbol, qty, side, client_order_id: {
            "id": "x",
            "status": "partially_filled",
            "filled_qty": "1",
            "filled_avg_price": "10.0",
        },
    )
    fill = b.place_order(_order())
    assert fill.status == "accepted"
    assert fill.filled_qty == 1 and fill.avg_price == 10.0


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
