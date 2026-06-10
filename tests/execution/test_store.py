from datetime import datetime, timezone

from moneybot.execution.models import Fill
from moneybot.execution.store import PositionStore


def _fill(side, qty, price, ticker="NVDA", oid="c1"):
    return Fill(
        client_order_id=oid,
        broker_order_id="b1",
        ticker=ticker,
        side=side,
        status="filled",
        filled_qty=qty,
        avg_price=price,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )


def test_empty_store_returns_nothing(tmp_path):
    store = PositionStore(tmp_path)
    assert store.get_all() == []


def test_apply_fill_creates_position(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    positions = store.get_all()
    assert len(positions) == 1
    assert positions[0].ticker == "NVDA" and positions[0].qty == 10.0


def test_persists_across_instances(tmp_path):
    PositionStore(tmp_path).apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    reopened = PositionStore(tmp_path)
    assert reopened.get_all()[0].qty == 10.0


def test_apply_fill_is_idempotent_on_client_order_id(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="dup"))
    store.apply_fill(_fill("buy", 10, 100.0, oid="dup"))  # same id -> no double count
    assert store.get_all()[0].qty == 10.0


def test_distinct_ids_accumulate(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    store.apply_fill(_fill("buy", 10, 120.0, oid="o2"))
    pos = store.get_all()[0]
    assert pos.qty == 20.0 and pos.avg_price == 110.0


def test_closing_removes_position(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    store.apply_fill(_fill("sell", 10, 130.0, oid="o2"))
    assert store.get_all() == []
