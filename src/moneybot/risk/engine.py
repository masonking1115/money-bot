"""RiskEngine: the deterministic layer the agents cannot bypass.

Pure Python, no LLM. Given the Analyst's TradePlans and a portfolio snapshot, it
runs a fixed rule pipeline and emits a RiskAssessment. Two GLOBAL gates can stop
the whole cycle (kill switch, daily-loss circuit breaker); the remaining rules
are per-plan (Task 7) and an optional hedge is computed last (Task 8). Per-name
and sector caps come from the active strategy's parameters; operational limits
come from Settings. Every decision records the rule(s) that fired.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from moneybot.risk.kill_switch import kill_switch_active
from moneybot.risk.metrics import average_dollar_volume, realized_volatility
from moneybot.risk.models import HedgeOrder, RiskAssessment, RiskDecision
from moneybot.risk.sizing import target_weight

if TYPE_CHECKING:
    from moneybot.analyst.models import TradePlan
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.risk.models import PortfolioState
    from moneybot.strategies.base import Strategy


class RiskEngine:
    def __init__(
        self,
        *,
        data_layer: DataLayer,
        strategy: Strategy,
        settings: Settings,
    ) -> None:
        self.data = data_layer
        self.strategy = strategy
        self.settings = settings

    @staticmethod
    def _veto(plan: TradePlan, rule: str, reasoning: str) -> RiskDecision:
        return RiskDecision(
            ticker=plan.ticker, approved=False, rules_fired=[rule], reasoning=reasoning
        )

    def assess(
        self,
        plans: list[TradePlan],
        portfolio: PortfolioState,
        as_of: date | None = None,
    ) -> RiskAssessment:
        """Run the rule pipeline over the cycle's plans and return verdicts."""
        if kill_switch_active(self.settings):
            return RiskAssessment(
                decisions=[
                    self._veto(p, "kill_switch", "kill switch engaged") for p in plans
                ],
                halted=True,
            )

        if portfolio.day_pnl_pct <= -self.settings.daily_loss_limit_pct:
            return RiskAssessment(
                decisions=[
                    self._veto(
                        p,
                        "daily_loss_circuit_breaker",
                        f"day P&L {portfolio.day_pnl_pct:.2%} at/under the "
                        f"-{self.settings.daily_loss_limit_pct:.0%} floor",
                    )
                    for p in plans
                ],
                halted=True,
            )

        params = self.strategy.parameters()
        held = {pos.ticker for pos in portfolio.positions}
        running_gross = portfolio.gross_exposure_pct

        decisions: list[RiskDecision] = []
        for plan in plans:
            decision = self._assess_plan(
                plan, portfolio, params, held, running_gross, as_of
            )
            decisions.append(decision)
            if decision.approved:
                running_gross += decision.target_weight
                held.add(plan.ticker)

        hedge = None
        if params.hedge_enabled:
            hedge = self._hedge(portfolio, decisions, as_of)
        return RiskAssessment(decisions=decisions, halted=False, hedge=hedge)

    def _in_earnings_blackout(self, ticker: str, as_of: date | None) -> bool:
        """True when a known earnings date is today..N days ahead of as_of.

        Without an as_of we cannot measure proximity, and we never fabricate a
        clock — so the blackout simply cannot fire in that case.
        """
        if as_of is None:
            return False
        try:
            meta = self.data.universe.get(ticker)
        except KeyError:
            return False
        earnings = meta.earnings_date
        if earnings is None:
            return False
        days = (earnings - as_of).days
        return 0 <= days <= self.settings.earnings_blackout_days

    def _assess_plan(
        self,
        plan: TradePlan,
        portfolio: PortfolioState,
        params,  # StrategyParams
        held: set[str],
        running_gross: float,
        as_of: date | None,
    ) -> RiskDecision:
        if plan.ticker not in self.data.universe.symbols:
            return self._veto(plan, "sanity", "not a tradeable name")

        if plan.ticker in held:
            return self._veto(plan, "already_held", "position already open; no pyramiding")

        if self._in_earnings_blackout(plan.ticker, as_of):
            return self._veto(
                plan, "earnings_blackout", "within the earnings blackout window"
            )

        bars = self.data.get_bars(
            plan.ticker,
            self.settings.risk_timeframe,
            self.settings.risk_lookback_days,
            as_of=as_of,
        )
        closes = [] if bars.empty else bars["close"].tolist()
        volumes = [] if bars.empty else bars["volume"].tolist()
        price = closes[-1] if closes else None
        if price is None or price <= 0:
            return self._veto(plan, "sanity", "no valid reference price")

        adv = average_dollar_volume(closes, volumes)
        if adv is None or adv < self.settings.min_dollar_volume:
            return self._veto(plan, "liquidity", "below the minimum $-volume floor")

        weight = target_weight(
            conviction=plan.conviction,
            volatility=realized_volatility(closes),
            max_position_pct=params.max_position_pct,
            target_volatility=self.settings.target_volatility,
        )
        if weight <= 0:
            return self._veto(plan, "sizing", "computed a zero position size")

        rules: list[str] = []
        target_dollars = weight * portfolio.equity

        sector_headroom = round(
            (params.max_sector_exposure_pct - running_gross) * portfolio.equity, 2
        )
        if sector_headroom <= 0:
            return self._veto(plan, "sector_exposure_cap", "no sector exposure headroom")
        if sector_headroom < target_dollars:
            target_dollars = sector_headroom
            rules.append("sector_exposure_cap")

        if portfolio.cash < target_dollars:
            target_dollars = portfolio.cash
            rules.append("insufficient_cash")

        shares = int(target_dollars // price)
        if shares <= 0:
            return RiskDecision(
                ticker=plan.ticker,
                approved=False,
                rules_fired=rules or ["sanity"],
                reasoning="rounds to zero shares",
            )

        actual_dollars = shares * price
        return RiskDecision(
            ticker=plan.ticker,
            approved=True,
            target_weight=actual_dollars / portfolio.equity,
            target_dollars=actual_dollars,
            shares=shares,
            reference_price=price,
            rules_fired=rules,
            reasoning="approved within limits",
        )

    def _hedge(
        self,
        portfolio: PortfolioState,
        decisions: list[RiskDecision],
        as_of: date | None,
    ) -> HedgeOrder | None:
        """Short the benchmark to offset a fraction of gross long exposure.

        Gross long = existing long market value + newly approved dollars. Returns
        None when there is nothing to hedge or the benchmark cannot be priced.
        """
        new_long = sum(d.target_dollars for d in decisions if d.approved)
        gross_long = portfolio.long_market_value + new_long
        if gross_long <= 0:
            return None

        benchmark = self.data.universe.benchmark
        bars = self.data.get_bars(
            benchmark, self.settings.risk_timeframe, self.settings.risk_lookback_days, as_of=as_of
        )
        closes = [] if bars.empty else bars["close"].tolist()
        price = closes[-1] if closes else None
        if price is None or price <= 0:
            return None

        hedge_dollars = gross_long * self.settings.hedge_ratio
        shares = int(hedge_dollars // price)
        if shares <= 0:
            return None
        return HedgeOrder(
            ticker=benchmark, side="short", shares=shares, dollars=shares * price
        )
