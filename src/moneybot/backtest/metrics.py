"""Pure performance math: trade log from Fills, plus headline metrics."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from moneybot.backtest.models import EquityPoint, PerformanceMetrics, TradeRecord

if TYPE_CHECKING:
    from moneybot.execution.models import Fill

_TRADING_DAYS = 252


def max_drawdown(equities: list[float]) -> float:
    """Worst peak-to-trough decline as a positive fraction (0.25 == -25%)."""
    peak = float("-inf")
    worst = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, (e - peak) / peak)
    return abs(worst)


def _daily_returns(equities: list[float]) -> list[float]:
    out = []
    for prev, cur in zip(equities, equities[1:]):
        if prev != 0:
            out.append((cur - prev) / prev)
    return out


def sharpe(returns: list[float]) -> float:
    """Annualized Sharpe (risk-free = 0). Zero if <2 points or no variance."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return 0.0
    std = var ** 0.5
    return (mean / std) * (_TRADING_DAYS ** 0.5)


def build_trade_log(fills: list[Fill]) -> list[TradeRecord]:
    """FIFO-match buys to sells per ticker; one TradeRecord per closed lot.

    Long-only (phase 1): 'buy' opens, 'sell' closes; rejected fills and shorts/covers
    are ignored. Open lots with no matching sell at the end are not reported."""
    open_lots: dict[str, deque] = {}
    trades: list[TradeRecord] = []
    for f in fills:
        if f.status != "filled" or f.filled_qty <= 0:
            continue
        if f.side == "buy":
            open_lots.setdefault(f.ticker, deque()).append(
                {"qty": f.filled_qty, "price": f.avg_price, "date": f.ts.date()}
            )
        elif f.side == "sell":
            remaining = f.filled_qty
            lots = open_lots.get(f.ticker)
            while remaining > 0 and lots:
                lot = lots[0]
                matched = min(remaining, lot["qty"])
                pnl = (f.avg_price - lot["price"]) * matched
                pnl_pct = (f.avg_price - lot["price"]) / lot["price"] if lot["price"] else 0.0
                trades.append(
                    TradeRecord(
                        ticker=f.ticker, qty=matched, entry_date=lot["date"], exit_date=f.ts.date(),
                        entry_price=lot["price"], exit_price=f.avg_price, pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason=f.reason,
                    )
                )
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] == 0:
                    lots.popleft()
    return trades


def compute_metrics(
    *,
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    starting_cash: float,
    benchmark_closes: list[float],
) -> PerformanceMetrics:
    equities = [p.equity for p in equity_curve]
    final_equity = equities[-1] if equities else starting_cash
    total_return = (final_equity - starting_cash) / starting_cash if starting_cash else 0.0

    returns = _daily_returns(equities)
    n_periods = len(returns)
    if n_periods > 0 and starting_cash > 0 and final_equity > 0:
        cagr = (final_equity / starting_cash) ** (_TRADING_DAYS / n_periods) - 1
    else:
        cagr = 0.0

    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / n_trades if n_trades else 0.0

    if benchmark_closes:
        b0, bn = benchmark_closes[0], benchmark_closes[-1]
        benchmark_return = (bn - b0) / b0 if b0 else 0.0
        benchmark_final_equity = starting_cash * (1 + benchmark_return)
    else:
        benchmark_return = 0.0
        benchmark_final_equity = starting_cash

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown(equities),
        sharpe=sharpe(returns),
        win_rate=win_rate,
        n_trades=n_trades,
        final_equity=final_equity,
        benchmark_return=benchmark_return,
        benchmark_final_equity=benchmark_final_equity,
    )
