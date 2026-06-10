"""Orchestrator: run one full trading cycle end-to-end.

Order of operations each cycle: kill-switch gate -> market-hours gate -> mechanical
exits (close triggered longs) -> research -> analyst -> portfolio snapshot -> risk
engine -> entry execution -> reconcile, journaling each step. Every collaborator is
injected, so tests use fakes and nothing hits the network or an LLM. The clock is
injected (no fabricated time); the cycle_id is derived from it so a re-run within
the same hour is idempotent at the broker/store.

This task implements construction + the two global gates; exits and the entry
pipeline are added next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from moneybot.execution.models import OrderRequest
from moneybot.orchestrator.exits import evaluate_exits
from moneybot.orchestrator.models import CycleResult
from moneybot.orchestrator.portfolio import build_portfolio_state, mark_price
from moneybot.risk.kill_switch import kill_switch_active

if TYPE_CHECKING:
    from datetime import date, datetime

    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.orchestrator.portfolio import SodEquityStore
    from moneybot.strategies.base import Strategy


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        data_layer: DataLayer,
        research,
        analyst,
        risk,
        execution,
        journal,
        sod_equity: SodEquityStore,
        strategy: Strategy,
        clock: Callable[[], datetime],
        market_open: Callable[[datetime], bool],
    ) -> None:
        self.settings = settings
        self.data = data_layer
        self.research = research
        self.analyst = analyst
        self.risk = risk
        self.execution = execution
        self.journal = journal
        self.sod_equity = sod_equity
        self.strategy = strategy
        self._clock = clock
        self._market_open = market_open

    def run_cycle(self, as_of=None) -> CycleResult:
        now = self._clock()
        cycle_id = now.strftime("%Y-%m-%dT%H")

        if kill_switch_active(self.settings):
            self.journal.append("halt", None, {"reason": "kill_switch"})
            return CycleResult(status="halted", reason="kill_switch", cycle_id=cycle_id)

        if not self._market_open(now):
            self.journal.append("skip", None, {"reason": "market_closed"})
            return CycleResult(status="skipped", reason="market_closed", cycle_id=cycle_id)

        as_of_date = as_of if as_of is not None else now.date()

        # 1. Mechanical exits on existing positions (before taking on new ones).
        exit_fills = self._run_exits(cycle_id=cycle_id, as_of=as_of, as_of_date=as_of_date)

        # 2. Research -> Analyst.
        research = self.research.research_universe(as_of=as_of)
        plans = self.analyst.analyze(research, as_of=as_of)
        self.journal.append("plans", None, {"count": len(plans)})

        # 3. Portfolio snapshot (marked to market) with today's P&L for the breaker.
        account = self.execution.broker.get_account()
        day_pnl = self.sod_equity.day_pnl_pct(account.equity, as_of_date)
        try:
            portfolio = build_portfolio_state(
                broker=self.execution.broker,
                data_layer=self.data,
                settings=self.settings,
                as_of=as_of,
                day_pnl_pct=day_pnl,
            )
        except ValueError as exc:
            # A non-positive-equity account cannot be sized against. Exits already
            # ran (capital-preserving); record why we stop and skip new entries.
            self.journal.append("halt", None, {"reason": "portfolio_unbuildable", "detail": str(exc)})
            return CycleResult(
                status="halted",
                reason="portfolio_unbuildable",
                cycle_id=cycle_id,
                exit_fills=exit_fills,
            )

        # 4. Risk Engine -> entry execution.
        assessment = self.risk.assess(plans, portfolio, as_of=as_of)
        if assessment.halted:
            entry_fills = []
        else:
            entry_fills = self.execution.execute(assessment, cycle_id=cycle_id)

        # Journal buy fills with their exit plan so the exit loop can time-stop later.
        plan_by_ticker = {p.ticker: p for p in plans}
        for fill in entry_fills:
            if fill.status == "filled" and fill.side == "buy":
                plan = plan_by_ticker.get(fill.ticker)
                self.journal.append(
                    "fill",
                    fill.ticker,
                    {
                        "side": "buy",
                        "shares": fill.filled_qty,
                        "price": fill.avg_price,
                        "exit_plan": plan.exit_plan.model_dump() if plan else None,
                    },
                )

        reconciliation = self.execution.reconcile()
        return CycleResult(
            status="completed",
            cycle_id=cycle_id,
            plans_proposed=len(plans),
            entry_fills=entry_fills,
            exit_fills=exit_fills,
            halted_by_risk=assessment.halted,
            reconciliation=reconciliation,
        )

    def _markable(self, ticker: str) -> bool:
        u = self.data.universe
        return ticker in u.symbols or ticker == u.benchmark

    def _mark(self, ticker: str, as_of: date | None) -> float | None:
        if not self._markable(ticker):
            return None
        return mark_price(
            data_layer=self.data,
            ticker=ticker,
            timeframe=self.settings.risk_timeframe,
            lookback=self.settings.risk_lookback_days,
            as_of=as_of,
        )

    def _entry_dates(self, tickers: list[str]) -> dict[str, date]:
        """Most recent buy-fill date per ticker, from the journal (source of truth)."""
        dates: dict[str, date] = {}
        for ticker in tickers:
            buys = [
                e
                for e in self.journal.read(ticker=ticker, kind="fill")
                if e.payload.get("side") == "buy"
            ]
            if buys:
                dates[ticker] = max(e.ts for e in buys).date()
        return dates

    def _run_exits(self, *, cycle_id: str, as_of: date | None, as_of_date: date) -> list:
        longs = [p for p in self.execution.broker.get_positions() if p.qty > 0]
        if not longs:
            return []
        current_prices: dict[str, float] = {}
        for p in longs:
            # Thread the cycle's as_of so a backtest marks point-in-time, not live.
            price = self._mark(p.ticker, as_of)
            if price is not None:
                current_prices[p.ticker] = price
        entry_dates = self._entry_dates([p.ticker for p in longs])
        signals = evaluate_exits(
            positions=longs,
            entry_dates=entry_dates,
            current_prices=current_prices,
            exit_plan=self.strategy.exit_plan(),
            as_of=as_of_date,
        )
        fills = []
        for sig in signals:
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{sig.ticker}:exit",
                ticker=sig.ticker,
                side="sell",
                quantity=sig.shares,
                reference_price=sig.reference_price,
                reason=sig.reason,
            )
            fill = self.execution.place(order)
            self.journal.append(
                "exit",
                sig.ticker,
                {"reason": sig.reason, "shares": sig.shares, "status": fill.status},
            )
            fills.append(fill)
        return fills
