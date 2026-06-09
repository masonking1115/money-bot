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
from moneybot.risk.models import RiskAssessment, RiskDecision

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

        # Per-plan pipeline and hedge are added in Tasks 7-8.
        return RiskAssessment(decisions=[], halted=False)
