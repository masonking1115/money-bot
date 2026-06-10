"""Drive the live orchestrator across historical days and assemble a report.

run_backtest is the single entry point. It is given an already-built data layer,
LLM, and retriever (the composition root wires real-or-cached providers); it builds
the orchestrator with a SimClock + always-open market gate + caching research/analyst,
replays each trading day through run_cycle(as_of=day), marks the portfolio to market
for the equity curve, and computes metrics vs the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from moneybot.analyst.factory import build_analyst_agent
from moneybot.backtest.agent_cache import CachingAnalyst, CachingResearch
from moneybot.backtest.calendar import trading_days_from_bars
from moneybot.backtest.clock import SimClock
from moneybot.backtest.metrics import build_trade_log, compute_metrics
from moneybot.backtest.models import BacktestReport, EquityPoint
from moneybot.orchestrator.factory import build_orchestrator
from moneybot.orchestrator.portfolio import build_portfolio_state
from moneybot.research.factory import build_research_agent

if TYPE_CHECKING:
    from moneybot.backtest.models import BacktestConfig
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever

_DAILY_BREAKER_NOTE = (
    "Daily-loss breaker is inert in a daily backtest: one cycle per day means "
    "start-of-day equity equals current equity, so day P&L is ~0. The breaker is "
    "an intraday protection; daily cadence cannot exercise it."
)


def run_backtest(
    *,
    settings: Settings,
    data_layer: DataLayer,
    llm: LLMClient,
    retriever: MemoryRetriever,
    config: BacktestConfig,
    cache_root: str | Path,
    benchmark_bars: pd.DataFrame,
) -> BacktestReport:
    cache_root = Path(cache_root)
    days = trading_days_from_bars(benchmark_bars, config.start, config.end)

    clock = SimClock()

    # Build (and cache-wrap) the AI layer; build_orchestrator uses these verbatim.
    if not config.use_agents:
        research = _NoResearch()
        analyst = _NoAnalyst()
    elif config.mode == "replay":
        # Replay must be truly offline: never construct an Anthropic client.
        # _NeverCalled raises if the cache misses, making a cache miss obvious.
        _sentinel = _NeverCalled()
        research = CachingResearch(_sentinel, root=cache_root, mode="replay")
        analyst = CachingAnalyst(_sentinel, root=cache_root, mode="replay")
    else:
        # record mode: build the real agents so live LLM calls can be cached.
        research = CachingResearch(
            build_research_agent(
                settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
            ),
            root=cache_root,
            mode=config.mode,
        )
        analyst = CachingAnalyst(
            build_analyst_agent(
                settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
            ),
            root=cache_root,
            mode=config.mode,
        )

    orch = build_orchestrator(
        settings=settings,
        data_layer=data_layer,
        retriever=retriever,
        llm=llm,
        clock=clock,
        market_open=lambda _now: True,  # calendar already restricts to real trading days
        research=research,
        analyst=analyst,
    )

    equity_curve: list[EquityPoint] = []
    fills = []
    for day in days:
        clock.set_day(day)
        result = orch.run_cycle(as_of=day)
        fills.extend(result.entry_fills)
        fills.extend(result.exit_fills)
        point = _mark_equity(orch=orch, settings=settings, data_layer=data_layer, day=day)
        equity_curve.append(point)

    benchmark_closes = _benchmark_closes(benchmark_bars, days)
    trades = build_trade_log(fills)
    metrics = compute_metrics(
        equity_curve=equity_curve,
        trades=trades,
        starting_cash=config.starting_cash,
        benchmark_closes=benchmark_closes,
    )
    return BacktestReport(
        config=config,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        notes=[_DAILY_BREAKER_NOTE],
    )


def _mark_equity(*, orch, settings, data_layer, day) -> EquityPoint:
    """Marked-to-market equity for the day (reuses the live point-in-time marker)."""
    broker = orch.execution.broker
    try:
        state = build_portfolio_state(
            broker=broker, data_layer=data_layer, settings=settings, as_of=day, day_pnl_pct=0.0
        )
        equity = state.equity
    except ValueError:
        # Non-positive equity (e.g. blown account): fall back to broker cash.
        equity = broker.get_account().cash
    account = broker.get_account()
    positions = [p for p in broker.get_positions() if p.qty != 0]
    return EquityPoint(day=day, equity=equity, cash=account.cash, n_positions=len(positions))


def _benchmark_closes(benchmark_bars: pd.DataFrame, days: list) -> list[float]:
    if benchmark_bars.empty or not days:
        return []
    df = benchmark_bars.copy()
    df["_day"] = pd.to_datetime(df["ts"]).dt.date
    by_day = dict(zip(df["_day"], df["close"]))
    return [by_day[d] for d in days if d in by_day]


class _NoResearch:
    def research_universe(self, as_of=None):
        return {}


class _NoAnalyst:
    def analyze(self, research, as_of=None):
        return []


class _NeverCalled:
    """Sentinel inner for CachingResearch/CachingAnalyst in replay mode.

    The caching wrappers serve every call from disk; if they somehow miss and
    fall through to the inner, this raises rather than constructing a real
    Anthropic client (which would fail without ANTHROPIC_API_KEY and break
    offline replay).
    """

    def research_universe(self, as_of=None):
        raise RuntimeError("inner agent called in replay mode (cache miss)")

    def analyze(self, research, as_of=None):
        raise RuntimeError("inner agent called in replay mode (cache miss)")
