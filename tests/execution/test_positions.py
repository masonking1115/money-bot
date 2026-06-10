from datetime import datetime, timezone

from moneybot.execution.models import Fill, PositionRecord
from moneybot.execution.positions import apply_fill


def _fill(side, qty, price, ticker="NVDA"):
    return Fill(
        client_order_id=f"c:{ticker}:{side}",
        broker_order_id="b1",
        ticker=ticker,
        side=side,
        status="filled",
        filled_qty=qty,
        avg_price=price,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )


def test_open_long_from_flat():
    r = apply_fill(None, _fill("buy", 10, 100.0))
    assert r.qty == 10.0 and r.avg_price == 100.0 and r.ticker == "NVDA"


def test_add_to_long_weighted_average():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("buy", 10, 120.0))
    assert r.qty == 20.0 and r.avg_price == 110.0  # (10*100 + 10*120)/20


def test_reduce_long_keeps_cost_basis():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 4, 130.0))
    assert r.qty == 6.0 and r.avg_price == 100.0  # cost basis unchanged on a partial exit


def test_fully_closing_returns_none():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 10, 130.0))
    assert r is None


def test_open_short_from_flat():
    r = apply_fill(None, _fill("short", 5, 200.0))
    assert r.qty == -5.0 and r.avg_price == 200.0


def test_cover_reduces_short():
    start = PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)
    r = apply_fill(start, _fill("cover", 2, 180.0, ticker="SMH"))
    assert r.qty == -3.0 and r.avg_price == 200.0


def test_crossing_through_zero_resets_avg_to_fill_price():
    # long 10, sell 15 -> net short 5 at the fill price
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 15, 130.0))
    assert r.qty == -5.0 and r.avg_price == 130.0


def test_add_to_short_weighted_average():
    # short 5 @ 200, short 3 more @ 190 -> short 8 at (5*200 + 3*190)/8 = 196.25
    start = PositionRecord(ticker="NVDA", qty=-5.0, avg_price=200.0)
    r = apply_fill(start, _fill("short", 3, 190.0))
    assert r.qty == -8.0
    assert abs(r.avg_price - 196.25) < 1e-9


def test_crossing_through_zero_short_to_long():
    # short 5, buy 8 -> net long 3 at the fill price
    start = PositionRecord(ticker="NVDA", qty=-5.0, avg_price=200.0)
    r = apply_fill(start, _fill("buy", 8, 185.0))
    assert r.qty == 3.0 and r.avg_price == 185.0
