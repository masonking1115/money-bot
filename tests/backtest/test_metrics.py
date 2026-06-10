from datetime import date, datetime, timezone

import pytest

from moneybot.backtest.metrics import (
    build_trade_log,
    compute_metrics,
    max_drawdown,
    sharpe,
)
from moneybot.backtest.models import EquityPoint
from moneybot.execution.models import Fill


def _fill(ticker, side, qty, price, day):
    return Fill(
        client_order_id=f"{ticker}:{side}:{day}", broker_order_id="b", ticker=ticker,
        side=side, status="filled", filled_qty=qty, avg_price=price,
        ts=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
    )


def test_max_drawdown_simple():
    # 100 -> 120 -> 90 -> 110 : worst peak-to-trough is 120 -> 90 = -25%
    eq = [100.0, 120.0, 90.0, 110.0]
    assert max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotonic_increase_is_zero():
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_sharpe_zero_when_no_variance():
    assert sharpe([0.01, 0.01, 0.01]) == 0.0  # std == 0 -> defined as 0


def test_sharpe_positive_for_positive_mean():
    assert sharpe([0.01, -0.005, 0.02, 0.0]) > 0


def test_build_trade_log_fifo_realized_pnl():
    fills = [
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),
        _fill("NVDA", "sell", 10, 110.0, date(2024, 1, 10)),
    ]
    trades = build_trade_log(fills)
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "NVDA" and t.qty == 10
    assert t.entry_price == 100.0 and t.exit_price == 110.0
    assert t.pnl == pytest.approx(100.0)
    assert t.pnl_pct == pytest.approx(0.10)


def test_build_trade_log_partial_then_full_exit():
    fills = [
        _fill("AMD", "buy", 10, 50.0, date(2024, 1, 2)),
        _fill("AMD", "sell", 4, 55.0, date(2024, 1, 5)),
        _fill("AMD", "sell", 6, 45.0, date(2024, 1, 9)),
    ]
    trades = build_trade_log(fills)
    assert len(trades) == 2
    assert trades[0].qty == 4 and trades[0].pnl == pytest.approx(20.0)
    assert trades[1].qty == 6 and trades[1].pnl == pytest.approx(-30.0)


def test_build_trade_log_ignores_rejected_and_unmatched():
    fills = [
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),  # still open at end -> no trade
    ]
    fills[0] = fills[0].model_copy(update={"status": "rejected"})
    assert build_trade_log(fills) == []


def test_compute_metrics_end_to_end():
    curve = [
        EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=0.0, n_positions=1),
        EquityPoint(day=date(2024, 1, 3), equity=110_000.0, cash=0.0, n_positions=1),
    ]
    trades = build_trade_log([
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),
        _fill("NVDA", "sell", 10, 110.0, date(2024, 1, 3)),
    ])
    m = compute_metrics(
        equity_curve=curve, trades=trades, starting_cash=100_000.0,
        benchmark_closes=[200.0, 204.0],
    )
    assert m.total_return == pytest.approx(0.10)
    assert m.final_equity == 110_000.0
    assert m.win_rate == 1.0
    assert m.n_trades == 1
    assert m.benchmark_return == pytest.approx(0.02)
    assert m.benchmark_final_equity == pytest.approx(102_000.0)


def test_compute_metrics_empty_curve_is_safe():
    m = compute_metrics(equity_curve=[], trades=[], starting_cash=100_000.0, benchmark_closes=[])
    assert m.total_return == 0.0 and m.n_trades == 0 and m.final_equity == 100_000.0
